"""Apply launcher-owned Flash-Next source additions to the pinned JITLLM checkout."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
JITLLM_REF = "67a7b8b28e19f3212bfbfc251de7ff9bb35bd863"
SOURCE = ROOT / ".local/jitllm"


def prepare(source: Path = SOURCE) -> None:
    if not source.exists():
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/beehive-lab/jitllm", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", JITLLM_REF], check=True)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != JITLLM_REF:
        raise RuntimeError(f"JITLLM checkout must be at {JITLLM_REF}, found {head}; local edits are preserved")
    additions = ROOT / "extensions/jitllm/src"
    manifest = source / ".launcher-source-overlay.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    installed = {}
    operations = []
    def upstream_bytes(key):
        result = subprocess.run(["git", "-C", str(source), "show", f"HEAD:{key}"], capture_output=True)
        return result.stdout if result.returncode == 0 else None

    for path in sorted(p for p in additions.rglob("*") if p.is_file()):
        destination = source / "src" / path.relative_to(additions)
        key = str(destination.relative_to(source))
        if (destination.exists() and destination.read_bytes() != path.read_bytes()
                and hashlib.sha256(destination.read_bytes()).hexdigest() != previous.get(key)
                and destination.read_bytes() != upstream_bytes(key)):
            raise RuntimeError(f"Refusing to overwrite a divergent JITLLM source file: {destination}")
        operations.append((path, destination))
        installed[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    removed = []
    for key, checksum in previous.items():
        destination = source / key
        if key not in installed and destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
                raise RuntimeError(f"Refusing to remove a divergent overlay file: {destination}")
            removed.append((destination, upstream_bytes(key)))
    # Preflight all files before applying the overlay.
    for path, destination in operations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for destination, baseline in removed:
        if baseline is None:
            destination.unlink()
        else:
            destination.write_bytes(baseline)
    manifest.write_text(json.dumps(installed, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
