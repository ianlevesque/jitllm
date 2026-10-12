"""Check the Java checkpoint reader against independent on-disk fixtures."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "model.language_model.layers.1.ple.ple_embedding"
JAVA_CLASS = "org.beehive.jitllm.integration.cli.FlashNextCheckpointProbe"


def write_tensors(path, tensors):
    offset = 0
    header = {}
    data = bytearray()
    for name, dtype, shape, value in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(value)]}
        data.extend(value)
        offset += len(value)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


class JitllmCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("javac"):
            raise unittest.SkipTest("JDK required")
        cls.compiled = tempfile.TemporaryDirectory(prefix="jitllm-checkpoint-classes-")
        cls.classes = cls.compiled.name
        java = ROOT / "src/main/java/org/beehive/jitllm"
        sources = list((java / "format/safetensors").glob("*.java"))
        sources += [java / "model/qwen4/PleHash.java", java / "model/loader/Qwen4PleTable.java",
                    java / "integration/cli/FlashNextCheckpointProbe.java"]
        sources.append(ROOT / "src/main/java/org/beehive/jitllm/server/Json.java")
        if not sources[-1].exists():
            raise unittest.SkipTest("JITLLM source checkout required")
        subprocess.run(["javac", "-d", cls.classes, *map(str, sources)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.compiled.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jitllm-checkpoint-")
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        self.tensors = [
            (PREFIX + ".layer_multipliers", "I64", [3], struct.pack("<3q", 92233720368547757, 81234567890123457, 71234567890123457)),
            (PREFIX + ".ngram_heads_vocab_sizes", "I64", [4], struct.pack("<4q", 7, 11, 13, 17)),
            (PREFIX + ".ngram_heads_offsets", "I64", [4], struct.pack("<4q", 0, 7, 18, 31)),
            (PREFIX + ".ngram_embedding.weight_scale", "BF16", [1], struct.pack("<H", 0x3f00)),
        ]
        self.payload = bytes([0x38, 0x40, 0x80, 0x7e, 0xb8, 0x01, 0x3c, 0x00] * 12)
        # Unequal shards make arithmetic based on one assumed shard width fail.
        self.tensors += [
            (PREFIX + ".ngram_embedding.shard_1.weight", "F8_E4M3", [31, 2], self.payload[34:]),
            (PREFIX + ".ngram_embedding.shard_0.weight", "F8_E4M3", [17, 2], self.payload[:34]),
        ]
        write_tensors(self.model / "model.safetensors", self.tensors)

    def probe(self, request, expected_ok=True):
        path = self.model / "request.json"
        path.write_text(json.dumps({"cache_bytes": 128, "page_bytes": 32, **request}))
        process = subprocess.run(["java", "-cp", self.classes, JAVA_CLASS, str(self.model), str(path)], capture_output=True, text=True)
        if expected_ok:
            self.assertEqual(process.returncode, 0, process.stderr)
            return json.loads(process.stdout)
        self.assertNotEqual(process.returncode, 0)
        return process.stderr

    @staticmethod
    def expected_float(byte):
        exp, mant = (byte >> 3) & 15, byte & 7
        value = mant * 2**-9 if exp == 0 else (1 + mant / 8) * 2**(exp - 7)
        return (-value if byte & 128 else value) * .5

    def test_staging_across_pages_and_shards(self):
        rows = [0, 16, 17, 47, 17, 0]
        result = self.probe({"rows": rows})
        expected = [self.expected_float(b) for row in rows for b in self.payload[row * 2:row * 2 + 2]]
        self.assertEqual(result["gather"], expected)
        self.assertEqual(result["rows"], 48)
        self.assertLessEqual(result["cache"]["resident_bytes"], 128)
        self.assertGreater(result["cache"]["hits"], 0)

    def test_signed_overflow_hash_and_eos(self):
        windows = [[248044, 248044, 101], [103, 248044, 101], [248044, 103, 101], [247000, 201000, 198000], [101, 103, 248044]]
        result = self.probe({"windows": windows})
        multipliers = [92233720368547757, 81234567890123457, 71234567890123457]
        def signed(value):
            value &= (1 << 64) - 1
            return value - (1 << 64) if value >= 1 << 63 else value
        for window, batch in zip(windows, result["batches"]):
            mix = signed(window[-1] * multipliers[0])
            crossed_eos = False
            ids = []
            for pos in (1, 2):
                previous = window[-1 - pos]
                crossed_eos |= previous == 248044
                mix = signed(mix ^ signed((248044 if crossed_eos else previous) * multipliers[pos]))
                for size, offset in ([ (7, 0), (11, 7)] if pos == 1 else [(13, 18), (17, 31)]):
                    ids.append(mix % size + offset)
            self.assertEqual(batch["ids"], ids)

    def test_indexed_multifile(self):
        write_tensors(self.model / "metadata.safetensors", self.tensors[:4])
        write_tensors(self.model / "table.safetensors", self.tensors[4:])
        mapping = {t[0]: "metadata.safetensors" if i < 4 else "table.safetensors" for i, t in enumerate(self.tensors)}
        (self.model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        self.assertEqual(self.probe({"rows": [47]})["tensor_count"], 6)

    def test_read_past_end_and_negative_rows_fail(self):
        for row in [-1, 48]:
            self.assertIn("PLE row", self.probe({"rows": [row]}, False))

    def test_missing_shard_fails(self):
        tensors = [t for t in self.tensors if "shard_0" not in t[0]]
        write_tensors(self.model / "model.safetensors", tensors)
        self.assertIn("Missing PLE shard", self.probe({}, False))

    def test_truncated_payload_fails(self):
        path = self.model / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        self.assertIn("Invalid tensor extent", self.probe({}, False))

    def test_shape_byte_mismatch_fails(self):
        self.tensors[-1] = (*self.tensors[-1][:2], [18, 2], self.tensors[-1][3])
        write_tensors(self.model / "model.safetensors", self.tensors)
        self.assertIn("byte count", self.probe({}, False))

    def test_index_path_traversal_fails(self):
        (self.model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "../escape.safetensors"}}))
        self.assertIn("escapes model directory", self.probe({}, False))

    def test_too_small_cache_budget_fails(self):
        self.assertIn("at least one page", self.probe({"cache_bytes": 16}, False))

    def test_wrong_hash_history_width_fails(self):
        self.assertIn("history width", self.probe({"windows": [[1, 2]]}, False))

    def test_nvfp4_low_nibble_first_and_two_level_scales(self):
        prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj"
        self.tensors += [
            (prefix + ".weight", "U8", [2, 8], bytes(range(0x10, 0x18)) + bytes(range(0x98, 0xa0))),
            (prefix + ".weight_scale", "F8_E4M3", [2, 1], bytes([0x38, 0x40])),
            (prefix + ".weight_scale_2", "F32", [], struct.pack("<f", .25)),
        ]
        write_tensors(self.model / "model.safetensors", self.tensors)
        result = self.probe({"matrix_prefix": prefix, "matrix_rows": [0, 1]})
        values = [0, .5, 1, 1.5, 2, 3, 4, 6]
        expected = []
        for row, raw in enumerate([range(0x10, 0x18), range(0x98, 0xa0)]):
            decoded = []
            for byte in raw:
                for nibble in (byte & 15, byte >> 4):
                    decoded.append(values[nibble & 7] * (-1 if nibble & 8 else 1) * (row + 1) * .25)
            expected.append(decoded)
        self.assertEqual(result["matrix"], expected)

    def test_fp8_mtp_128_wide_block_scales(self):
        prefix = "mtp.layers.0.mlp.experts.0.gate_proj"
        self.tensors += [
            (prefix + ".weight", "F8_E4M3", [129, 129], bytes([0x38]) * (129 * 129)),
            (prefix + ".weight_scale_inv", "BF16", [2, 2], struct.pack("<4H", 0x3f80, 0x4000, 0x4040, 0x4080)),
        ]
        write_tensors(self.model / "model.safetensors", self.tensors)
        result = self.probe({"matrix_prefix": prefix, "matrix_rows": [0, 127, 128]})
        self.assertEqual(result["matrix"], [[1.] * 128 + [2.], [1.] * 128 + [2.], [3.] * 128 + [4.]])

    def test_dense_bf16_matrix(self):
        prefix = "model.language_model.layers.0.mlp.gate"
        self.tensors += [(prefix + ".weight", "BF16", [2, 2], struct.pack("<4H", 0x3f80, 0xc000, 0x0000, 0x3f00))]
        write_tensors(self.model / "model.safetensors", self.tensors)
        result = self.probe({"matrix_prefix": prefix, "matrix_rows": [1, 0]})
        self.assertEqual(result["matrix"], [[0., .5], [1., -2.]])

    def test_malformed_nvfp4_scales_rejected(self):
        prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj"
        self.tensors += [
            (prefix + ".weight", "U8", [1, 8], bytes(8)),
            (prefix + ".weight_scale", "F8_E4M3", [1, 2], bytes(2)),
            (prefix + ".weight_scale_2", "F32", [], struct.pack("<f", 1.)),
        ]
        write_tensors(self.model / "model.safetensors", self.tensors)
        self.assertIn("Invalid NVFP4 block scales", self.probe({"matrix_prefix": prefix, "matrix_rows": [0]}, False))


if __name__ == "__main__":
    unittest.main()
