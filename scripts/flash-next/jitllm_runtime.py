"""Run the pinned local TornadoVM SDK and JITLLM checkpoint probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
INSTALL = ROOT / ".local/jitllm-tornadovm/cuda-jdk25/current"


def runtime_environment(install: Path = INSTALL) -> tuple[Path, dict]:
    provenance = json.loads((install / "provenance.json").read_text())
    sdk = Path(provenance["sdk_dir"])
    env = os.environ.copy()
    env["JAVA_HOME"] = env.get("JAVA_HOME", "/usr/lib/jvm/java-25-openjdk-amd64")
    env["TORNADOVM_HOME"] = str(sdk)
    native = ROOT / ".venv-qwen-flash-next/lib/python3.13/site-packages/nvidia"
    from jitllm_native_libraries import prepare_native_aliases
    aliases=prepare_native_aliases(native,ROOT/".local/jitllm-native-libs")
    paths = [str(sdk / "lib"), str(aliases), str(native / "cudnn/lib"), str(native / "cu13/lib")]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(paths)
    from jitllm_native_libraries import native_qsa_environment
    native_qsa_environment(Path(env.get("JITLLM_QSA_PROVENANCE",ROOT/".local/jitllm-native-libs/qsa-provenance.json")),env)
    from jitllm_native_libraries import native_hc_environment
    native_hc_environment(Path(env.get("JITLLM_HC_PROVENANCE",ROOT/".local/jitllm-native-libs/hc-provenance.json")),env)
    return sdk, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, default=INSTALL)
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("devices")
    command.add_parser("build")
    command.add_parser("test")
    paging=command.add_parser("qsa-paging-probe")
    paging.add_argument("--device",type=int,default=1)
    draft=command.add_parser("qsa-draft-probe")
    draft.add_argument("--device",type=int,default=1)
    xqa=command.add_parser("xqa-probe")
    xqa.add_argument("request",type=Path)
    vision=command.add_parser("vision-probe",help="CPU vision tower parity probe")
    vision.add_argument("model",type=Path)
    vision.add_argument("image",type=Path)
    vision.add_argument("extra",nargs=argparse.REMAINDER,help="probe options such as --rgb 96x64 --dump-embeddings F")
    serve=command.add_parser("serve",help="OpenAI-compatible HTTP server (Tornado device 0 is the RTX PRO 6000 on this host)")
    serve.add_argument("model",type=Path)
    serve.add_argument("--host",default="127.0.0.1")
    serve.add_argument("--port",type=int,default=30020)
    serve.add_argument("--device",type=int,default=0)
    serve.add_argument("--context",type=int,default=4096)
    serve.add_argument("--mtp",action=argparse.BooleanOptionalAction,default=True)
    serve.add_argument("--device-memory",default="88GB")
    serve.add_argument("--gpu-cache-bytes",type=int,default=64<<30,help="Flash-Next GPU weight cache budget (projections + MoE experts)")
    serve.add_argument("--page-cache-bytes",type=int,default=8<<30,help="host NVMe page cache for checkpoint/PLE reads")
    probe = command.add_parser("probe")
    probe.add_argument("model", type=Path)
    probe.add_argument("--matrix", default="model.language_model.layers.0.mlp.experts.0.gate_proj")
    probe.add_argument("--device", type=int, default=1, help="TornadoVM ordinal; run devices first (5090 is 1 on this host)")
    for name in ("generate", "model-probe", "cache-probe", "api-probe", "gpu-api-probe", "speculative-probe", "recurrent-probe", "hc-ple-probe", "bf16-batch-probe", "nvfp4-batch-probe", "fp8-batch-probe", "fp8-moe-probe", "mtp-fusion-probe", "mtp-block-probe", "hc-projection-probe", "moe-probe", "qsa-probe"):
        runner = command.add_parser(name)
        runner.add_argument("model", type=Path)
        if name != "api-probe":
            runner.add_argument("request", type=Path)
        if name == "cache-probe":
            runner.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    sdk, env = runtime_environment(args.install)
    if args.command in {"build", "test"}:
        from prepare_jitllm_source import prepare
        prepare()
        options = ["-DskipTests", "package"] if args.command == "build" else ["-Djitllm.VectorBitSize=256", "clean", "test"]
        subprocess.run([str(ROOT / ".local/jitllm/mvnw"), "-B", *options],
                       cwd=ROOT / ".local/jitllm", env=env, check=True)
    elif args.command == "devices":
        subprocess.run([str(sdk / "bin/tornado"), "--devices"], env=env, check=True)
    elif args.command in {"qsa-paging-probe", "qsa-draft-probe"}:
        subprocess.run([str(sdk / "bin/tornado"), "--jvm", "--enable-preview --add-modules=jdk.incubator.vector -Xmx1g -Dtornado.device.memory=1GB",
                        "--classpath", str(ROOT / ".local/jitllm/target/classes"),
                        "org.beehive.jitllm.integration.cli."+("FlashNextQsaPagingProbe" if args.command=="qsa-paging-probe" else "FlashNextQsaDraftProbe"),str(args.device)],env=env,check=True)
    elif args.command == "probe":
        subprocess.run([str(sdk / "bin/tornado"), "--jvm", "--enable-preview --add-modules=jdk.incubator.vector",
                        "--classpath", str(ROOT / ".local/jitllm/target/classes"),
                        "org.beehive.jitllm.integration.cli.FlashNextNvfp4Probe", str(args.model), args.matrix, str(args.device)],
                       env=env, check=True)
    elif args.command == "serve":
        subprocess.run([str(sdk / "bin/tornado"), "--jvm",
                        f"--enable-preview --add-modules=jdk.incubator.vector -Xmx4g -Dtornado.device.memory={args.device_memory} -Djitllm.qwen4.device={args.device} -Djitllm.qwen4.mtp={str(args.mtp).lower()} -Djitllm.qwen4.gpuCacheBytes={args.gpu_cache_bytes} -Djitllm.qwen4.pageCacheBytes={args.page_cache_bytes}",
                        "--classpath", str(ROOT / ".local/jitllm/target/classes"),
                        "org.beehive.jitllm.server.OpenAIServer",
                        "--params", f"serve --model {args.model} --host {args.host} --port {args.port} --gpu --ctx-size {args.context}"], env=env, check=True)
    elif args.command == "vision-probe":
        extra=args.extra[1:] if args.extra[:1]==["--"] else args.extra
        subprocess.run([str(sdk / "bin/tornado"), "--jvm", "--enable-preview --add-modules=jdk.incubator.vector -Xmx4g",
                        "--classpath", str(ROOT / ".local/jitllm/target/classes"),
                        "org.beehive.jitllm.integration.cli.FlashNextVisionProbe",
                        "--params", " ".join([str(args.model), str(args.image), *extra])],env=env,check=True)
    elif args.command == "xqa-probe":
        subprocess.run([str(sdk / "bin/tornado"), "--jvm", "--enable-preview --enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector -Xmx1g",
                        "--classpath", str(ROOT / ".local/jitllm/target/classes"),
                        "org.beehive.jitllm.integration.cli.FlashNextXqaProbe",str(args.request)],env=env,check=True)
    else:
        main = {"moe-probe":"FlashNextMoeProbe", "nvfp4-batch-probe":"FlashNextNvfp4BatchProbe", "fp8-batch-probe":"FlashNextFp8BatchProbe", "fp8-moe-probe":"FlashNextFp8MoeProbe", "mtp-fusion-probe":"FlashNextMtpFusionProbe", "mtp-block-probe":"FlashNextMtpBlockProbe", "hc-projection-probe":"FlashNextHcProjectionProbe", "generate": "FlashNextCpuGenerate", "model-probe": "FlashNextModelProbe",
                "cache-probe": "FlashNextCacheProbe", "api-probe": "FlashNextApiProbe", "gpu-api-probe": "FlashNextGpuApiProbe", "speculative-probe": "FlashNextSpeculativeProbe", "recurrent-probe": "FlashNextDeltaNetProbe", "hc-ple-probe": "FlashNextHcPleProbe", "bf16-batch-probe": "FlashNextBf16BatchProbe", "qsa-probe": "FlashNextQsaProbe"}[args.command]
        inputs = [str(args.model)]
        if args.command != "api-probe":
            inputs.append(str(args.request))
        if args.command == "cache-probe":
            inputs.append(str(args.device))
        subprocess.run([str(sdk / "bin/tornado"), "--jvm",
                        "--enable-preview --add-modules=jdk.incubator.vector -Xmx4g -Dtornado.device.memory=24GB",
                        "--classpath", str(ROOT / ".local/jitllm/target/classes"),
                        "org.beehive.jitllm.integration.cli." + main, *inputs], env=env, check=True)
