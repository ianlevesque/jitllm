"""Validate native Flash-Next configuration and tokenization against the HF runtime."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import tempfile

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]


def verify(model):
    oracle = Tokenizer.from_file(str(model / "tokenizer.json"))
    texts = ["Hello world!", "你好世界", "e\u0301 cafe\u0301", "日本語 한국어 العربية हिन्दी", "1234567890", "   \n\t\r\n", "😀🐝🤖🧠", "Can't won't I'LL they'd I'Ve", "",
             "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n", "<tool_call>\n<function=test>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>", "a<|im_end|><|im_start|>b", "<think>xyz</think>", "<|vision_start|><|image_pad|><|vision_end|>"]
    pieces = [" hi", "\n", "\t", "I'm", "e\u0301", "你好", "1", ".", "😀", "مرحبا", "世界", "ab", "<think>", "</think>", "<|im_end|>"]
    rng = random.Random(20261012)
    texts += ["".join(rng.choice(pieces) for _ in range(rng.randrange(1, 35))) for _ in range(200)]
    texts += [token["content"] for token in json.loads((model / "tokenizer.json").read_text())["added_tokens"]]
    with tempfile.TemporaryDirectory(prefix="jitllm-metadata-") as temp:
        request = Path(temp) / "request.json"
        request.write_text(json.dumps({"texts": texts}))
        process = subprocess.run(["java", "-Xmx1g", "-cp", str(ROOT / "target/classes"), "org.beehive.jitllm.integration.cli.FlashNextMetadataProbe", str(model), str(request)], capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(process.stderr)
        actual = json.loads(process.stdout)
    for text, result in zip(texts, actual["texts"], strict=True):
        expected = oracle.encode(text).ids
        if expected != result["ids"]:
            raise AssertionError((text, expected, result["ids"]))
        decoded = oracle.decode(expected, skip_special_tokens=False)
        if decoded != result["decoded"]:
            raise AssertionError((text, decoded, result["decoded"]))
    config = json.loads((model / "config.json").read_text())["text_config"]
    dims = actual["dimensions"]
    assert dims["layers"] == config["num_hidden_layers"] == 48
    assert dims["kv_layers"] == config["layer_types"].count("full_attention") == 12
    assert dims["head_size"] == config["head_dim"] == 256
    assert dims["hyper_size"] == config["hidden_size"] * config["hc_count"] == 10240
    assert dims["rotary_dim"] == config["head_dim"] * config["partial_rotary_factor"] == 64
    assert dims["context"] == config["max_position_embeddings"] == 262144
    return {"checkpoint": str(model.resolve()), "texts_checked": len(texts), "all_token_ids_exact": True, "all_decodes_exact": True, "dimensions": dims}


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
