"""Compare Java NVMe PLE staging with PyTorch using the actual NVIDIA checkpoint.

Run with .venv-qwen-flash-next/bin/python verify_jitllm_checkpoint.py MODEL_DIR.
Only requested rows are read. The 51 GB PLE table is never materialized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import struct
import subprocess
import tempfile

import torch

ROOT = Path(__file__).resolve().parent
PREFIX = "model.language_model.layers.1.ple.ple_embedding"


def verify(model: Path) -> dict:
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}

    def entry(name):
        file = model / index[name]
        if file not in headers:
            with file.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                headers[file] = (8 + length, json.loads(stream.read(length)))
        base, header = headers[file]
        return file, base, header[name]

    def read(name, offset=0, count=None):
        file, base, metadata = entry(name)
        begin, end = metadata["data_offsets"]
        if count is None:
            count = end - begin
        if offset < 0 or offset + count > end - begin:
            raise ValueError("Oracle read exceeds tensor")
        with file.open("rb") as stream:
            stream.seek(base + begin + offset)
            data = stream.read(count)
        if len(data) != count:
            raise ValueError("Truncated oracle read")
        return bytearray(data)

    multipliers = torch.frombuffer(read(PREFIX + ".layer_multipliers"), dtype=torch.int64).clone()
    sizes = torch.frombuffer(read(PREFIX + ".ngram_heads_vocab_sizes"), dtype=torch.int64).clone()
    offsets = torch.frombuffer(read(PREFIX + ".ngram_heads_offsets"), dtype=torch.int64).clone()
    scale = torch.frombuffer(read(PREFIX + ".ngram_embedding.weight_scale"), dtype=torch.bfloat16).clone()
    shards = sorted((name for name in index if name.startswith(PREFIX + ".ngram_embedding.shard_") and name.endswith(".weight")),
                    key=lambda name: int(re.search(r"shard_(\d+)", name)[1]))
    ends = []
    total = 0
    for name in shards:
        total += entry(name)[2]["shape"][0]
        ends.append(total)
    width = entry(shards[0])[2]["shape"][1]
    rng = random.Random(20261012)
    rows = [0, total - 1] + [row for end in ends[:-1] for row in [end - 1, end]]
    rows += [rng.randrange(total) for _ in range(256)]
    windows = [[248044, 248044, 101], [104, 248044, 101], [104, 105, 248044]]
    windows += [[rng.randrange(248320) for _ in range(3)] for _ in range(64)]

    def oracle_hash(window):
        contexts = torch.tensor([window], dtype=torch.int64)
        shifted = [contexts]
        positions = torch.arange(contexts.shape[1])
        eos_positions = torch.where(contexts == 248044, positions, -1)
        previous_eos = torch.cat([torch.full((1, 1), -1, dtype=torch.int64), torch.cummax(eos_positions, dim=1).values[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        for shift in (1, 2):
            source = positions - shift
            values = contexts.gather(1, source.clamp_min(0).unsqueeze(0))
            shifted.append(torch.where((positions - segment_start >= shift) & (source >= 0), values, 248044))
        mixed = shifted[0] * multipliers[0]
        ids = []
        for pos in (1, 2):
            mixed = torch.bitwise_xor(mixed, shifted[pos] * multipliers[pos])
            start, end = (pos - 1) * 8, pos * 8
            ids += (torch.remainder(mixed[:, -1:].unsqueeze(-1), sizes[start:end]) + offsets[start:end]).flatten().tolist()
        return ids

    def oracle_gather(ids):
        import bisect
        values = []
        for row in ids:
            shard = bisect.bisect_right(ends, row)
            begin = 0 if shard == 0 else ends[shard - 1]
            raw = read(shards[shard], (row - begin) * width, width)
            values.append((torch.frombuffer(raw, dtype=torch.float8_e4m3fn).to(torch.bfloat16) * scale).float())
        return torch.cat(values)

    def oracle_matrix_row(prefix, row):
        metadata = entry(prefix + ".weight")[2]
        columns = metadata["shape"][1]
        if metadata["dtype"] == "U8":
            raw = torch.frombuffer(read(prefix + ".weight", row * columns, columns), dtype=torch.uint8)
            nibbles = torch.stack([raw & 15, raw >> 4], dim=-1).flatten().long()
            magnitudes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
            decoded = magnitudes[nibbles & 7] * torch.where(nibbles & 8 != 0, -1., 1.)
            scale_columns = columns * 2 // 16
            scales = torch.frombuffer(read(prefix + ".weight_scale", row * scale_columns, scale_columns), dtype=torch.float8_e4m3fn).float()
            global_scale = torch.frombuffer(read(prefix + ".weight_scale_2"), dtype=torch.float32)[0]
            return decoded * scales.repeat_interleave(16) * global_scale
        if metadata["dtype"] == "F8_E4M3":
            decoded = torch.frombuffer(read(prefix + ".weight", row * columns, columns), dtype=torch.float8_e4m3fn).float()
            scale_columns = (columns + 127) // 128
            scales = torch.frombuffer(read(prefix + ".weight_scale_inv", (row // 128) * scale_columns * 2, scale_columns * 2), dtype=torch.bfloat16).float()
            return decoded * scales.repeat_interleave(128)[:columns]
        raise ValueError("Unsupported oracle matrix dtype")

    with tempfile.TemporaryDirectory(prefix="jitllm-real-checkpoint-") as temp:
        compiled = Path(temp) / "classes"
        java = ROOT / "extensions/jitllm/src/main/java/org/beehive/jitllm"
        sources = list((java / "format/safetensors").glob("*.java"))
        sources += [java / "model/qwen4/PleHash.java", java / "model/loader/Qwen4PleTable.java",
                    java / "integration/cli/FlashNextCheckpointProbe.java"]
        sources.append(ROOT / ".local/jitllm/src/main/java/org/beehive/jitllm/server/Json.java")
        subprocess.run(["javac", "-d", str(compiled), *map(str, sources)], check=True)
        request = Path(temp) / "request.json"
        request.write_text(json.dumps({"windows": windows, "rows": rows, "cache_bytes": 1048576, "page_bytes": 4096}))
        result = subprocess.run(["java", "-Xmx128m", "-cp", str(compiled), "org.beehive.jitllm.integration.cli.FlashNextCheckpointProbe", str(model), str(request)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        actual = json.loads(result.stdout)

        matrix_reports = []
        for base in ["model.language_model.layers.0", "mtp.layers.0"]:
            for projection in ["gate_proj", "up_proj", "down_proj"]:
                prefix = base + ".mlp.experts.0." + projection
                matrix_rows = [0, 1, 127, 128, entry(prefix + ".weight")[2]["shape"][0] - 1]
                request.write_text(json.dumps({"matrix_prefix": prefix, "matrix_rows": matrix_rows, "cache_bytes": 1048576, "page_bytes": 4096}))
                result = subprocess.run(["java", "-Xmx128m", "-cp", str(compiled), "org.beehive.jitllm.integration.cli.FlashNextCheckpointProbe", str(model), str(request)],
                                        capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                decoded = json.loads(result.stdout)
                for row, observed in zip(matrix_rows, decoded["matrix"], strict=True):
                    torch.testing.assert_close(torch.tensor(observed), oracle_matrix_row(prefix, row), rtol=0, atol=0)
                matrix_reports.append({"prefix": prefix, "dtype": entry(prefix + ".weight")[2]["dtype"], "rows_checked": matrix_rows, "max_absolute_error": 0.})

    maximum_error = 0.
    for window, batch in zip(windows, actual["batches"], strict=True):
        ids = oracle_hash(window)
        assert ids == batch["ids"], (window, ids, batch["ids"])
        expected = oracle_gather(ids)
        observed = torch.tensor(batch["values"])
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        maximum_error = max(maximum_error, (expected - observed).abs().max().item())
    expected = oracle_gather(rows)
    torch.testing.assert_close(torch.tensor(actual["gather"]), expected, rtol=0, atol=0)
    assert actual["cache"]["resident_bytes"] <= 1048576
    return {"checkpoint": str(model.resolve()), "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            "tensor_count": actual["tensor_count"], "shards": len(shards), "table_rows": total, "row_width": width,
            "hash_windows": len(windows), "gather_rows": len(rows) + len(windows) * len(sizes),
            "max_absolute_error": maximum_error, "cache": actual["cache"], "java_heap_limit_mib": 128, "matrices": matrix_reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.model), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    print(report)
