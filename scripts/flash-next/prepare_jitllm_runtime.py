"""Build the pinned CUDA/JDK 25 TornadoVM SDK for this checkout."""
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]

TORNADOVM_REF = "a40ccf9a692fc06fe4c499f3700a3037e91dfebe"


def prepare_runtime() -> None:
    env = os.environ.copy()
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
    # Optional pip-installed CUDA 13/cuDNN 9 dependencies. Otherwise use the system toolchain.
    if env.get("JITLLM_NVIDIA_DIR"):
        native = Path(env["JITLLM_NVIDIA_DIR"]).resolve()
        libraries = ROOT / ".local/jitllm-native-libs"
        libraries.mkdir(parents=True, exist_ok=True)
        for name, origin in {"libcudnn.so": native / "cudnn/lib/libcudnn.so.9", "libcudart.so": native / "cu13/lib/libcudart.so.13", "libnvrtc.so": native / "cu13/lib/libnvrtc.so.13"}.items():
            if not origin.exists():
                raise RuntimeError(f"Missing native CUDA dependency: {origin}")
            link = libraries / name
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise RuntimeError(f"Native library link path is occupied: {link}")
            link.symlink_to(origin)
        env["CPLUS_INCLUDE_PATH"] = ":".join([str(native / "cudnn/include"), str(native / "cu13/include"), env.get("CPLUS_INCLUDE_PATH", "")])
        env["LIBRARY_PATH"] = ":".join([str(libraries), env.get("LIBRARY_PATH", "")])
    subprocess.run([str(ROOT / "scripts/tornadovm-dev.sh"), "setup", "--ref", TORNADOVM_REF,
                    "--backend", "cuda", "--jdk", "25", "--root", str(ROOT / ".local/jitllm-tornadovm")],
                   cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    prepare_runtime()
