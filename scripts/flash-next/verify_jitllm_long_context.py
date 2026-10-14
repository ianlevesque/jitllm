"""Qualify JITLLM Flash-Next at long context with a needle-in-haystack ladder.

Drives the serve HTTP API (``jitllm_runtime.py serve``) rather than the gpu-api probe:
the probe pins ``-Dtornado.device.memory=24GB`` and a 4096-token request default, while
serve takes an explicit ``--ctx-size`` and an 88GB device budget, which is the shape a
262144-token qualification has to run in. Start the server first, for example:

    .venv-qwen-flash-next/bin/python jitllm_runtime.py serve MODEL_DIR \
        --context 262144 --port 30020 --device 0 --device-memory 88GB

(jitllm_runtime's serve ``--context`` still defaults to 4096; pass it explicitly. Its
``--device`` is a TornadoVM ordinal — 0 is the RTX PRO 6000 on this host per the serve
help — while this script's ``--gpu`` is an nvidia-smi index: 1 is the RTX PRO 6000.)

Each rung builds one chat prompt of an exact tokenizer-measured length: deterministic
filler, one planted passcode fact at a stated depth, and a closing question. A rung
passes when the model returns the planted code. ``--dry-run`` builds and tokenizes the
prompts on CPU only and asserts the exact token length of every rung.

Nominal rung lengths are prompt-token targets. The top rung cannot literally be 262144
prompt tokens in a 262144 context (the session needs room to generate, and the facade
returns CONTEXT_FULL once prompt tokens reach the context), so every target is clamped
to ``context - max_tokens - reserve`` and the exact figure is recorded per rung.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
DEFAULT_LENGTHS = (4096, 32768, 131072, 262144)
DEFAULT_DEPTHS = (0.1, 0.5, 0.9)
# One token per repetition for the checkpoint tokenizer (" filler" -> a single id,
# verified in tests); the builder re-tokenizes and fails loudly if that ever stops
# holding, rather than silently landing near the target.
PAD_UNIT = " filler"
FILLER_SENTENCES = (
    "The quarterly facilities report describes routine maintenance schedules for the northern "
    "warehouse, listing ordinary inventory counts for shelving, lighting fixtures, loading "
    "equipment, and the delivery vehicles assigned to regional distribution routes.",
    "A separate section of the memorandum summarises seasonal staffing rotations, noting that "
    "supervisors reviewed holiday coverage, training records, safety inspections, and the "
    "standard procedures used when new contractors begin work at the depot.",
    "The appendix records monthly utility readings for water, heating, and electricity across "
    "each building, together with brief remarks about insulation repairs, gutter cleaning, "
    "car park resurfacing, and the replacement timetable for ageing ventilation units.",
    "Meeting notes from the operations committee mention procurement timelines for office "
    "supplies, archive storage arrangements, visitor badge policies, and the annual review of "
    "insurance documentation held by the administrative office.",
)
INSTRUCTION = (
    "Read the document below carefully. It contains one specific fact: a passcode for a named "
    "project. After the document you will be asked for that passcode.\n\nDOCUMENT:\n"
)
QUESTION = "\n\nWhat is the passcode for project {codename}? Answer with only the passcode."


def needle_text(codename: str, code: str) -> str:
    return f"The passcode for project {codename} is {code}."


def rung_identity(seed: int, nominal_length: int, depth: float) -> tuple[str, str]:
    """Deterministic codename/passcode for one rung, distinct across rungs."""
    tag = (seed * 1_000_003 + nominal_length) * 100 + round(depth * 100)
    return f"LANTERN-{nominal_length}", f"KF-{tag % 900000 + 100000}-{tag % 89 + 10}"


class TokenCounter:
    """Counts tokens of the exact chat-templated text the server is expected to ingest."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def templated(self, user_text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)

    def count(self, user_text: str) -> int:
        return len(self.tokenizer(self.templated(user_text))["input_ids"])

    def count_raw(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])


def build_prompt(counter: TokenCounter, target_tokens: int, depth: float,
                 codename: str, code: str) -> dict:
    """One user message of exactly ``target_tokens`` templated tokens.

    The needle sits at ``depth`` of the *filler span* (0 = just after the instruction,
    1 = just before the question). Length is hit by repeating whole filler sentences to
    just under the target, then padding the shortfall with PAD_UNIT repetitions inside
    the tail filler; the final text is re-tokenized and equality asserted, so a tokenizer
    whose merges change at the joins fails here instead of at the model.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")
    needle = needle_text(codename, code)
    question = QUESTION.format(codename=codename)
    fixed = INSTRUCTION + needle + question
    fixed_tokens = counter.count(fixed)
    filler_budget = target_tokens - fixed_tokens
    if filler_budget < 0:
        raise ValueError(
            f"target {target_tokens} tokens is below the {fixed_tokens}-token fixed frame")
    head_budget = round(filler_budget * depth)

    def filler_span(budget: int) -> str:
        pieces, used, i = [], 0, 0
        while True:
            sentence = FILLER_SENTENCES[i % len(FILLER_SENTENCES)]
            cost = counter.count_raw((" " if pieces else "") + sentence)
            if used + cost > budget:
                break
            pieces.append(sentence)
            used += cost
            i += 1
        return " ".join(pieces)

    head = filler_span(head_budget)
    tail = filler_span(filler_budget - head_budget)
    parts = [INSTRUCTION, head, (" " if head else "") + needle, (" " if tail else "") + tail, question]
    text = "".join(parts)
    actual = counter.count(text)
    pad_tokens = counter.count_raw(PAD_UNIT)
    if pad_tokens != 1:
        raise AssertionError(f"PAD_UNIT {PAD_UNIT!r} tokenizes to {pad_tokens} tokens, expected 1")
    for _ in range(8):
        missing = target_tokens - actual
        if missing == 0:
            break
        if missing < 0:
            raise AssertionError(
                f"prompt overshot target by {-missing} tokens (sentence-granularity overflow)")
        text = text[: -len(question)] + PAD_UNIT * missing + question
        actual = counter.count(text)
    if actual != target_tokens:
        raise AssertionError(
            f"prompt is {actual} tokens, target was {target_tokens}; pad adjustment did not converge")
    needle_char = text.index(needle)
    return {
        "text": text,
        "prompt_tokens": actual,
        "prompt_chars": len(text),
        "needle": needle,
        "needle_char_offset": needle_char,
        "needle_depth_fraction": round(needle_char / len(text), 4) if text else 0.0,
        # OpenAIServer.validationError estimates prompt tokens as chars/4 and rejects at
        # estimated + max_tokens >= context; surface that number before any request is sent.
        "server_estimated_prompt_tokens": len(text) // 4,
    }


def target_for(nominal: int, context: int, max_tokens: int, reserve: int) -> int:
    return min(nominal, context - max_tokens - reserve)


class GpuSampler:
    """Polls nvidia-smi memory for one GPU index while a request runs."""

    def __init__(self, gpu_index: int, interval: float = 0.5):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_mib(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                 "-i", str(self.gpu_index)],
                capture_output=True, text=True, timeout=10, check=True)
            return int(out.stdout.strip().splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def __enter__(self):
        baseline = self._read_mib()
        self.baseline_mib = baseline
        if baseline is not None:
            self.samples.append(baseline)

        def loop():
            while not self._stop.wait(self.interval):
                value = self._read_mib()
                if value is not None:
                    self.samples.append(value)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False

    def report(self) -> dict:
        if not self.samples:
            return {"gpu_index": self.gpu_index, "peak_mib": None, "baseline_mib": None,
                    "peak_delta_mib": None, "samples": 0,
                    "note": "nvidia-smi unavailable or no reading on this index"}
        peak = max(self.samples)
        return {"gpu_index": self.gpu_index, "peak_mib": peak, "baseline_mib": self.samples[0],
                "peak_delta_mib": peak - self.samples[0], "samples": len(self.samples),
                "note": "absolute device memory; other processes on this GPU are included"}


def http_json(url: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as failure:
        raise RuntimeError(
            f"HTTP {failure.code} from {url}: {failure.read().decode(errors='replace')[:500]}") from failure


def run_rung_http(base_url: str, model: str | None, prompt: dict, max_tokens: int,
                  timeout: float, stream: bool) -> dict:
    """One chat completion; returns text, server/local token counts, and timing."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt["text"]}],
        "temperature": 0,
        "seed": 1234,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    started = time.perf_counter()
    if not stream:
        body = http_json(url, payload, timeout)
        elapsed = time.perf_counter() - started
        choice = body["choices"][0]
        message = choice.get("message") or {}
        text = (message.get("reasoning_content") or "") + (message.get("content") or "")
        usage = body.get("usage") or {}
        return {"text": text, "elapsed_seconds": elapsed, "ttft_seconds": None,
                "decode_seconds": None, "completion_tokens": usage.get("completion_tokens"),
                "server_prompt_tokens": usage.get("prompt_tokens"),
                "finish_reason": choice.get("finish_reason"), "streamed_chunks": None}
    payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    pieces, usage, finish, chunks = [], {}, None, 0
    first_token_at = None
    try:
        response_context = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as failure:
        raise RuntimeError(
            f"HTTP {failure.code} from {url}: {failure.read().decode(errors='replace')[:500]}") from failure
    with response_context as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(piece)
                    chunks += 1
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    ended = time.perf_counter()
    ttft = None if first_token_at is None else first_token_at - started
    return {"text": "".join(pieces), "elapsed_seconds": ended - started,
            "ttft_seconds": ttft,
            "decode_seconds": None if first_token_at is None else ended - first_token_at,
            "completion_tokens": usage.get("completion_tokens"),
            "server_prompt_tokens": usage.get("prompt_tokens"),
            "finish_reason": finish, "streamed_chunks": chunks}


def served_model(base_url: str, timeout: float) -> dict:
    body = http_json(base_url.rstrip("/") + "/v1/models", None, timeout)
    entry = (body.get("data") or [{}])[0]
    return {"id": entry.get("id"), "context_length": entry.get("context_length")}


def load_counter(model_dir: str) -> TokenCounter:
    from transformers import AutoTokenizer
    return TokenCounter(AutoTokenizer.from_pretrained(model_dir))


def resolve_model_dir(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("JITLLM_FLASH_NEXT_MODEL")
    if env:
        return env
    snapshots = sorted(Path.home().glob(
        ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/*"))
    if snapshots:
        return str(snapshots[-1])
    raise SystemExit("no checkpoint tokenizer found: pass --model-dir (or set JITLLM_FLASH_NEXT_MODEL)")


def parse_lengths(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"invalid lengths: {text!r}")
    return values


def parse_depths(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.split(",") if part.strip())
    if not values or any(not 0.0 <= v <= 1.0 for v in values):
        raise argparse.ArgumentTypeError(f"invalid depths: {text!r}")
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", help="checkpoint directory (tokenizer source)")
    parser.add_argument("--base-url", default="http://127.0.0.1:30020",
                        help="serve base URL (jitllm_runtime serve default port)")
    parser.add_argument("--model", help="served model id; default: read from /v1/models")
    parser.add_argument("--context", type=int, default=262144,
                        help="context the server was started with (--ctx-size)")
    parser.add_argument("--lengths", type=parse_lengths, default=DEFAULT_LENGTHS,
                        help="comma-separated nominal prompt-token rungs")
    parser.add_argument("--depths", type=parse_depths, default=DEFAULT_DEPTHS,
                        help="comma-separated needle depths in [0,1]")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reserve", type=int, default=16,
                        help="extra tokens held back below context - max_tokens")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu", type=int, default=1, help="nvidia-smi GPU index to sample")
    parser.add_argument("--timeout", type=float, default=7200.0, help="per-request seconds")
    parser.add_argument("--token-slack", type=int, default=8,
                        help="allowed |server - local| prompt-token difference before a rung fails")
    parser.add_argument("--no-stream", action="store_true",
                        help="non-streaming requests (server usage, but no TTFT)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build + tokenize prompts on CPU only; assert exact lengths; no HTTP")
    parser.add_argument("--label", default="serve")
    parser.add_argument("--output", type=Path,
                        help="default: benchmarks/jitllm-flash-next/long-context-<label>.json")
    args = parser.parse_args(argv)

    model_dir = resolve_model_dir(args.model_dir)
    counter = load_counter(model_dir)
    served = None
    if not args.dry_run:
        served = served_model(args.base_url, min(args.timeout, 60.0))
        if served["context_length"] and served["context_length"] != args.context:
            print(f"warning: /v1/models reports context_length={served['context_length']}, "
                  f"--context says {args.context}; targets use the smaller", file=sys.stderr)
            args.context = min(args.context, served["context_length"])
        if args.model is None:
            args.model = served["id"]

    rungs = []
    for nominal in args.lengths:
        target = target_for(nominal, args.context, args.max_tokens, args.reserve)
        for depth in args.depths:
            codename, code = rung_identity(args.seed, nominal, depth)
            prompt = build_prompt(counter, target, depth, codename, code)
            rung = {
                "nominal_length": nominal,
                "target_prompt_tokens": target,
                "context": args.context,
                "depth": depth,
                "codename": codename,
                "expected_code": code,
                **{k: v for k, v in prompt.items() if k != "text"},
            }
            estimate_rejected = (
                prompt["server_estimated_prompt_tokens"] + args.max_tokens >= args.context)
            rung["server_estimate_would_reject"] = estimate_rejected
            if args.dry_run:
                # Dry-run qualifies prompt construction only: exact token length is the pass
                # condition. server_estimate_would_reject is recorded, not judged, because it
                # describes the server's chars/4 pre-check, which only a live request tests.
                rung["passed"] = prompt["prompt_tokens"] == target
                rung["mode"] = "dry-run (tokenization only)"
                rungs.append(rung)
                print(f"dry-run rung nominal={nominal} depth={depth}: "
                      f"{prompt['prompt_tokens']} tokens (target {target}) "
                      f"chars={prompt['prompt_chars']} estimate_rejected={estimate_rejected}")
                continue
            # The estimate flag is this script's copy of the server's pre-check, and the server
            # is the truth: send anyway and record its actual answer per rung.
            try:
                with GpuSampler(args.gpu) as sampler:
                    result = run_rung_http(args.base_url, args.model, prompt, args.max_tokens,
                                           args.timeout, stream=not args.no_stream)
            except Exception as failure:
                rung.update(passed=False, mode="http", failure=f"request failed: {failure}",
                            gpu_memory=sampler.report())
                rungs.append(rung)
                print(f"rung nominal={nominal} depth={depth}: passed=False ({failure})")
                continue
            retrieved = code in result["text"]
            server_tokens = result["server_prompt_tokens"]
            token_mismatch = (
                server_tokens is not None
                and abs(server_tokens - prompt["prompt_tokens"]) > args.token_slack)
            rung.update(
                mode="http",
                passed=retrieved and not token_mismatch,
                retrieved=retrieved,
                response_text=result["text"][:2000],
                elapsed_seconds=result["elapsed_seconds"],
                ttft_seconds=result["ttft_seconds"],
                decode_seconds=result["decode_seconds"],
                completion_tokens=result["completion_tokens"],
                streamed_chunks=result["streamed_chunks"],
                finish_reason=result["finish_reason"],
                server_prompt_tokens=server_tokens,
                prompt_tokens_source="server usage" if server_tokens is not None else "local tokenizer",
                token_count_mismatch=token_mismatch,
                gpu_memory=sampler.report(),
            )
            if token_mismatch:
                rung["failure"] = (f"server counted {server_tokens} prompt tokens, local "
                                   f"tokenizer counted {prompt['prompt_tokens']} "
                                   f"(chat-template mismatch beyond --token-slack {args.token_slack})")
            elif not retrieved:
                rung["failure"] = "planted passcode not in response"
            rungs.append(rung)
            print(f"rung nominal={nominal} depth={depth}: passed={rung['passed']} "
                  f"elapsed={result['elapsed_seconds']:.1f}s ttft={result['ttft_seconds']}")

    report = {
        "passed": all(r["passed"] for r in rungs),
        "mode": "dry-run" if args.dry_run else "http",
        "checkpoint_tokenizer": model_dir,
        "base_url": None if args.dry_run else args.base_url,
        "served_model": served,
        "context": args.context,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "gpu_index": args.gpu,
        "scope": ("needle retrieval per rung; prompt lengths are chat-templated tokenizer "
                  "counts clamped to context - max_tokens - reserve; dry-run qualifies "
                  "prompt construction only, not the model"),
        "rungs": rungs,
    }
    output = args.output or (ROOT / "benchmarks/jitllm-flash-next" / f"long-context-{args.label}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}: passed={report['passed']} rungs={len(rungs)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
