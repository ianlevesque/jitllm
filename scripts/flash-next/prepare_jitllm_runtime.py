"""Build the pinned CUDA TornadoVM SDK with this launcher's native CUDA libraries."""
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from prepare_jitllm_source import ROOT, SOURCE, prepare
from jitllm_native_libraries import prepare_native_aliases

TORNADOVM_REF = "a40ccf9a692fc06fe4c499f3700a3037e91dfebe"


def prepare_runtime() -> None:
    prepare()
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", "/usr/lib/jvm/java-25-openjdk-amd64")
    jdk21 = Path(env.get("JVMCI_SOURCE_JDK", str(ROOT / ".local/jitllm-jdk21")))
    if not (jdk21 / "lib/modules").exists():
        if "JVMCI_SOURCE_JDK" in env:
            raise RuntimeError(f"JVMCI_SOURCE_JDK has no runtime image: {jdk21}")
        with tempfile.TemporaryDirectory(prefix="jitllm-jdk21-") as temp:
            archive = Path(temp) / "jdk.tar.gz"
            subprocess.run(["curl", "-fL", "--retry", "2", "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse", "-o", str(archive)], check=True)
            with tarfile.open(archive) as tar:
                tar.extractall(Path(temp) / "extracted", filter="data")
            candidates = list((Path(temp) / "extracted").iterdir())
            if len(candidates) != 1 or not (candidates[0] / "lib/modules").exists():
                raise RuntimeError("Downloaded JDK21 has an unexpected layout")
            jdk21.parent.mkdir(parents=True, exist_ok=True)
            # The extraction lives on /tmp; rename can cross filesystems.
            import shutil
            shutil.copytree(candidates[0], jdk21)
    env["JVMCI_SOURCE_JDK"] = str(jdk21)
    native = ROOT / ".venv-qwen-flash-next/lib/python3.13/site-packages/nvidia"
    libraries = prepare_native_aliases(native,ROOT / ".local/jitllm-native-libs")
    env["CPLUS_INCLUDE_PATH"] = ":".join([str(native / "cudnn/include"), str(native / "cu13/include"), env.get("CPLUS_INCLUDE_PATH", "")])
    env["LIBRARY_PATH"] = ":".join([str(libraries), env.get("LIBRARY_PATH", "")])
    subprocess.run([str(SOURCE / "scripts/tornadovm-dev.sh"), "setup", "--ref", TORNADOVM_REF,
                    "--backend", "cuda", "--jdk", "25", "--root", str(ROOT / ".local/jitllm-tornadovm")],
                   cwd=SOURCE, env=env, check=True)


if __name__ == "__main__":
    prepare_runtime()
