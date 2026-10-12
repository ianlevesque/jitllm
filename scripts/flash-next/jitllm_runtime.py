"""Build JITLLM and run the experimental Flash-Next checkpoint probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
INSTALL = Path(os.environ.get("JITLLM_TORNADOVM_INSTALL", ROOT / ".local/jitllm-tornadovm/cuda-jdk25/current"))


def runtime_environment(install: Path = INSTALL) -> tuple[Path, dict]:
    env = os.environ.copy()
    if env.get("TORNADOVM_HOME"):
        sdk = Path(env["TORNADOVM_HOME"]).resolve()
    elif (install / "provenance.json").is_file():
        sdk = Path(json.loads((install / "provenance.json").read_text())["sdk_dir"])
    else:
        raise RuntimeError("Export TORNADOVM_HOME or run scripts/flash-next/prepare_jitllm_runtime.py first")
    if not (sdk / "bin/tornado").is_file():
        raise RuntimeError(f"TornadoVM SDK runner is missing: {sdk / 'bin/tornado'}")
    env["TORNADOVM_HOME"] = str(sdk)
    paths = [str(sdk / "lib")]
    if env.get("JITLLM_NVIDIA_DIR"):
        native = Path(env["JITLLM_NVIDIA_DIR"])
        from jitllm_native_libraries import prepare_native_aliases
        paths.append(str(prepare_native_aliases(native,ROOT/".local/jitllm-native-libs")))
        paths.extend(str(native / path) for path in ("cudnn/lib", "cu13/lib"))
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(paths)
    return sdk, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, default=INSTALL)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("devices", "build", "test"):
        commands.add_parser(name)
    paging=commands.add_parser("qsa-paging-probe")
    paging.add_argument("--device",type=int,default=1)
    probe = commands.add_parser("probe")
    probe.add_argument("model", type=Path)
    probe.add_argument("--matrix", default="model.language_model.layers.0.mlp.experts.0.gate_proj")
    probe.add_argument("--device", type=int, default=1, help="TornadoVM device ordinal; inspect devices first")
    for name in ("generate", "model-probe", "cache-probe", "api-probe", "recurrent-probe", "hc-ple-probe", "bf16-batch-probe", "qsa-probe"):
        runner = commands.add_parser(name)
        runner.add_argument("model", type=Path)
        if name != "api-probe":
            runner.add_argument("request", type=Path)
        if name == "cache-probe":
            runner.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    sdk, env = runtime_environment(args.install)
    if args.command in {"build", "test"}:
        options = ["-DskipTests", "package"] if args.command == "build" else ["-Djitllm.VectorBitSize=256", "clean", "test"]
        subprocess.run([str(ROOT / "mvnw"), "-B", *options], cwd=ROOT, env=env, check=True)
    elif args.command == "devices":
        subprocess.run([str(sdk / "bin/tornado"), "--devices"], env=env, check=True)
    elif args.command == "qsa-paging-probe":
        subprocess.run([str(sdk / "bin/tornado"), "--jvm", "--enable-preview --add-modules=jdk.incubator.vector -Xmx1g -Dtornado.device.memory=1GB",
                        "--classpath", str(ROOT / "target/classes"),
                        "org.beehive.jitllm.integration.cli.FlashNextQsaPagingProbe",str(args.device)],env=env,check=True)
    else:
        if args.command == "probe":
            main_class = "FlashNextNvfp4Probe"
            inputs = [str(args.model), args.matrix, str(args.device)]
        else:
            main_class = {"generate": "FlashNextCpuGenerate", "model-probe": "FlashNextModelProbe",
                          "cache-probe": "FlashNextCacheProbe", "api-probe": "FlashNextApiProbe", "recurrent-probe": "FlashNextDeltaNetProbe", "hc-ple-probe": "FlashNextHcPleProbe", "bf16-batch-probe": "FlashNextBf16BatchProbe", "qsa-probe": "FlashNextQsaProbe"}[args.command]
            inputs = [str(args.model)]
            if args.command != "api-probe":
                inputs.append(str(args.request))
            if args.command == "cache-probe":
                inputs.append(str(args.device))
        subprocess.run([str(sdk / "bin/tornado"), "--jvm",
                        "--enable-preview --add-modules=jdk.incubator.vector -Xmx4g -Dtornado.device.memory=24GB",
                        "--classpath", str(ROOT / "target/classes"),
                        "org.beehive.jitllm.integration.cli." + main_class, *inputs], env=env, check=True)


if __name__ == "__main__":
    main()
