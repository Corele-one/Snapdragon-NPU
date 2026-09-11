#!/usr/bin/env python3
"""Qwen2.5-1.5B hard-gate, PPL, throughput, and logit-distribution audit."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from model_lineage import sha256 as file_sha256, stable_case_id


PROJECT = Path(__file__).resolve().parents[1]
SPEC = json.loads((PROJECT / "experiment_spec.json").read_text())
MODEL = SPEC["model"]
MODEL_MANIFEST: dict[str, object] | None = None
REMOTE = "/data/local/tmp/stage2_9_model_audit"
CPU_REMOTE = "/data/local/tmp/llama-cpu-reference-qwen"
RELAY_HOST = "wzliao@35.209.202.207"
RELAY_ADB_SOCKET = "tcp:127.0.0.1:15037"
FINAL_PPL = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)")
CHUNK_PPL = re.compile(r"\[(\d+)\]([0-9.eE+-]+)")
LOADED_MODEL = re.compile(r"loaded meta data with (\d+) key-value pairs and (\d+) tensors .+ \(version (.+)\)")
KV_LINE = re.compile(r"llama_model_loader: - kv\s+\d+:\s+([^ ]+)\s+\S+\s+=\s+(.*)")
TYPE_LINE = re.compile(r"llama_model_loader: - type\s+(\S+):\s+(\d+) tensors")


def run(argv: list[str], log: Path, timeout: int = 1800) -> str:
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log}")
    return proc.stdout


def adb_shell(command: str, log: Path, timeout: int = 1800) -> str:
    return run(["adb", "shell", command], log, timeout)


def push_verified(local: Path, remote: str, log: Path, timeout: int = 1800) -> None:
    """Avoid the connected device's unreliable multi-source ``push --sync``.

    A matching remote SHA is reused; otherwise one file is pushed and its
    content is verified before the audit can continue.
    """
    expected = file_sha256(local)
    quoted = shlex.quote(remote)
    probe = subprocess.run(["adb", "shell", f"if [ -f {quoted} ]; then sha256sum {quoted}; fi"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    if probe.returncode == 0 and probe.stdout.split(maxsplit=1)[:1] == [expected]:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"reused verified remote file {remote} sha256={expected}\n")
        return
    if local.stat().st_size >= 64 << 20:
        push_large_file(local, remote, expected, log, timeout)
    else:
        run(["adb", "push", str(local), remote], log, timeout)
    verified = subprocess.run(["adb", "shell", f"sha256sum {quoted}"], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    if verified.returncode or verified.stdout.split(maxsplit=1)[:1] != [expected]:
        raise RuntimeError(f"device SHA256 mismatch after pushing {local} to {remote}")
    with log.open("a") as stream:
        stream.write(f"verified sha256={expected}\n")
    cleanup = getattr(push_large_file, "_cleanup", None)
    if cleanup is not None:
        chunk_dir, remote_chunks = cleanup
        if not str(remote_chunks).startswith("/data/local/tmp/stage291_upload_"):
            raise RuntimeError(f"refusing to clean unexpected remote chunk path: {remote_chunks}")
        subprocess.run(["adb", "shell", f"rm -rf {shlex.quote(remote_chunks)}"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        if chunk_dir.parent == PROJECT / "generated/upload_chunks":
            shutil.rmtree(chunk_dir)
        delattr(push_large_file, "_cleanup")
    relay_cleanup = getattr(push_large_file, "_relay_cleanup", None)
    if relay_cleanup is not None:
        if not str(relay_cleanup).startswith("/home/wzliao/stage291-model-relay/"):
            raise RuntimeError(f"refusing to clean unexpected relay path: {relay_cleanup}")
        subprocess.run(["ssh", RELAY_HOST, f"unlink {shlex.quote(relay_cleanup)}"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        delattr(push_large_file, "_relay_cleanup")


def ensure_relay() -> None:
    probe = subprocess.run(
        ["ssh", RELAY_HOST, f"ADB_SERVER_SOCKET={RELAY_ADB_SOCKET} adb devices"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    if probe.returncode == 0 and "127.0.0.1:15555\tdevice" in probe.stdout:
        return
    subprocess.run(["ssh", "-fN", "-R", "127.0.0.1:15037:127.0.0.1:5037",
                    "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", RELAY_HOST],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    probe = subprocess.run(
        ["ssh", RELAY_HOST, f"ADB_SERVER_SOCKET={RELAY_ADB_SOCKET} adb devices"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    if probe.returncode or "127.0.0.1:15555\tdevice" not in probe.stdout:
        raise RuntimeError(f"relay cannot see authenticated device: {probe.stdout}")


def push_large_file(local: Path, remote: str, expected: str, log: Path, timeout: int) -> None:
    """Relay large files through SSH without exporting the local ADB key."""
    ensure_relay()
    relay_dir = "/home/wzliao/stage291-model-relay"
    relay_path = f"{relay_dir}/{expected}-{local.name}"
    subprocess.run(["ssh", RELAY_HOST, f"mkdir -p {shlex.quote(relay_dir)}"], check=True,
                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    probe = subprocess.run(["ssh", RELAY_HOST,
                            f"if [ -f {shlex.quote(relay_path)} ]; then sha256sum {shlex.quote(relay_path)}; fi"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    outputs = [f"relay_host={RELAY_HOST}", f"relay_adb_socket={RELAY_ADB_SOCKET}"]
    if probe.stdout.split(maxsplit=1)[:1] != [expected]:
        copied = subprocess.run(["scp", str(local), f"{RELAY_HOST}:{relay_path}"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        outputs.append(copied.stdout)
        if copied.returncode:
            raise RuntimeError(f"SCP relay upload failed for {local}: {copied.stdout}")
    relay_hash = subprocess.run(["ssh", RELAY_HOST, f"sha256sum {shlex.quote(relay_path)}"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if relay_hash.returncode or relay_hash.stdout.split(maxsplit=1)[:1] != [expected]:
        raise RuntimeError(f"relay SHA256 mismatch for {local}")
    # The device ships an old ADB sync service which can finish writing a large
    # file but never acknowledge the final close packet.  Bound that wait and
    # let push_verified's device-side SHA256 be the sole success criterion.
    relay_timeout = min(timeout, max(600, math.ceil(local.stat().st_size / (4 << 20))))
    try:
        pushed = subprocess.run(
            ["ssh", RELAY_HOST, f"ADB_SERVER_SOCKET={RELAY_ADB_SOCKET} adb push "
             f"{shlex.quote(relay_path)} {shlex.quote(remote)}"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=relay_timeout)
        outputs.append(f"relay_push_returncode={pushed.returncode}\n{pushed.stdout}")
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        outputs.append(f"relay_push_timed_out_after_s={relay_timeout}\n{partial}")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(outputs) + "\n")
    setattr(push_large_file, "_relay_cleanup", relay_path)


def push_large_file_chunked(local: Path, remote: str, expected: str, log: Path, timeout: int) -> None:
    chunk_bytes = 4 << 20
    chunk_dir = PROJECT / "generated/upload_chunks" / expected[:20]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    expected_chunks = (local.stat().st_size + chunk_bytes - 1) // chunk_bytes
    complete = (chunk_dir / "COMPLETE")
    if not complete.is_file() or complete.read_text().strip() != expected:
        for existing in chunk_dir.glob("chunk_*"):
            existing.unlink()
        with local.open("rb") as source:
            for index in range(expected_chunks):
                payload = source.read(chunk_bytes)
                (chunk_dir / f"chunk_{index:06d}").write_bytes(payload)
        complete.write_text(expected + "\n")
    chunks = sorted(chunk_dir.glob("chunk_*"))
    if len(chunks) != expected_chunks or sum(path.stat().st_size for path in chunks) != local.stat().st_size:
        raise RuntimeError(f"local transfer chunk inventory is incomplete for {local}")

    remote_chunks = f"/data/local/tmp/stage291_upload_{expected[:20]}"
    subprocess.run(["adb", "shell", f"mkdir -p {shlex.quote(remote_chunks)}"], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    listing = subprocess.run(
        ["adb", "shell", f"for f in {shlex.quote(remote_chunks)}/chunk_*; do "
         "[ -f \"$f\" ] && stat -c '%n %s' \"$f\"; done"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    remote_sizes = {line.split()[0].rsplit("/", 1)[-1]: int(line.split()[1])
                    for line in listing.stdout.splitlines() if len(line.split()) == 2}
    pending = [path for path in chunks if remote_sizes.get(path.name) != path.stat().st_size]

    def upload(path: Path) -> str:
        try:
            proc = subprocess.run(["adb", "push", str(path), f"{remote_chunks}/{path.name}"],
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            return f"{path.name} returncode={proc.returncode} {proc.stdout.strip()}"
        except subprocess.TimeoutExpired:
            # This old ADB often writes the whole chunk but never completes the
            # sync close handshake. The post-batch size check is authoritative.
            return f"{path.name} timed_out_after_300s"

    def restart_adb() -> None:
        subprocess.run(["adb", "kill-server"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=30)
        subprocess.run(["adb", "start-server"], check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=30)
        connect = subprocess.run(["adb", "connect", "127.0.0.1:15555"], text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        if connect.returncode:
            raise RuntimeError(f"ADB reconnect failed: {connect.stdout}")

    outputs: list[str] = []
    original_pending = len(pending)
    for offset in range(0, len(pending), 32):
        missing = pending[offset:offset + 32]
        for attempt in (1, 2):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(missing)) as pool:
                outputs.extend(f"batch={offset // 32} attempt={attempt} {value}"
                               for value in pool.map(upload, missing))
            restart_adb()
            names = " ".join(shlex.quote(f"{remote_chunks}/{path.name}") for path in missing)
            check = subprocess.run(["adb", "shell", f"for f in {names}; do stat -c '%n %s' \"$f\" 2>/dev/null; done"],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            sizes = {line.split()[0].rsplit("/", 1)[-1]: int(line.split()[1])
                     for line in check.stdout.splitlines() if len(line.split()) == 2}
            missing = [path for path in missing if sizes.get(path.name) != path.stat().st_size]
            if not missing:
                break
        if missing:
            raise RuntimeError(f"chunk batch remained incomplete after one retry: {[p.name for p in missing]}")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"parallel_chunks={len(chunks)} uploaded={original_pending} reused={len(chunks)-original_pending}\n" +
                   "\n".join(outputs) + "\n")
    quoted_remote = shlex.quote(remote)
    assemble = (f"cat {shlex.quote(remote_chunks)}/chunk_* > {quoted_remote}.partial && "
                f"mv {quoted_remote}.partial {quoted_remote}")
    subprocess.run(["adb", "shell", assemble], check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, text=True, timeout=timeout)
    # Keep no duplicate multi-GiB chunk tree after the full-file verification
    # performed by push_verified. Local chunks are also purely resumable scratch.
    log.write_text(log.read_text() + f"assembled={remote}\n")
    setattr(push_large_file, "_cleanup", (chunk_dir, remote_chunks))


def append(path: Path, record: dict[str, object]) -> None:
    identity = {key: record.get(key) for key in
                ("record_type", "mode", "kind", "tokens", "context", "chunk", "session") if key in record}
    record.setdefault("case_id", stable_case_id(identity))
    record.setdefault("attempt", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def extract_ppl(text: str) -> tuple[float, list[float]]:
    final = FINAL_PPL.search(text)
    if not final:
        raise RuntimeError("perplexity output has no final estimate")
    cumulative = [(int(index), float(value)) for index, value in CHUNK_PPL.findall(text)]
    cumulative.sort()
    chunk_nll: list[float] = []
    previous_total = 0.0
    for index, ppl in cumulative:
        if index != len(chunk_nll) + 1 or not math.isfinite(ppl) or ppl <= 0:
            raise RuntimeError("malformed cumulative chunk PPL sequence")
        total = index * math.log(ppl)
        chunk_nll.append(total - previous_total)
        previous_total = total
    return float(final.group(1)), chunk_nll


def extract_model_metadata(text: str) -> dict[str, object]:
    loaded = LOADED_MODEL.search(text)
    if not loaded:
        raise RuntimeError("model loader output has no metadata/tensor summary")
    metadata = {key: value for key, value in KV_LINE.findall(text)}
    tensor_types = {key: int(value) for key, value in TYPE_LINE.findall(text)}
    return {"kv_count": int(loaded.group(1)), "tensor_count": int(loaded.group(2)),
            "gguf_version": loaded.group(3), "metadata": metadata, "tensor_types": tensor_types}


def compare_model_metadata(cpu: dict[str, object], hmx: dict[str, object]) -> dict[str, object]:
    # Layout-specific payloads may differ, but the quantized CPU/HMX pair must
    # describe the same logical graph, tokenizer, dimensions and tensor types.
    ignored = {"general.sampling.top_k", "general.sampling.top_p", "general.sampling.temp"}
    cpu_meta = {k: v for k, v in dict(cpu["metadata"]).items() if k not in ignored}
    hmx_meta = {k: v for k, v in dict(hmx["metadata"]).items() if k not in ignored}
    missing_cpu = sorted(set(hmx_meta) - set(cpu_meta))
    missing_hmx = sorted(set(cpu_meta) - set(hmx_meta))
    differing = sorted(k for k in set(cpu_meta) & set(hmx_meta) if cpu_meta[k] != hmx_meta[k])
    structural_match = (cpu["tensor_count"] == hmx["tensor_count"] and
                        cpu["tensor_types"] == hmx["tensor_types"] and
                        not missing_cpu and not missing_hmx and not differing)
    return {"structural_match": structural_match, "cpu_tensor_count": cpu["tensor_count"],
            "hmx_tensor_count": hmx["tensor_count"], "cpu_tensor_types": cpu["tensor_types"],
            "hmx_tensor_types": hmx["tensor_types"], "missing_from_cpu": missing_cpu,
            "missing_from_hmx": missing_hmx, "differing_metadata": differing}


def deploy(log_dir: Path) -> dict[str, str]:
    model_host = PROJECT / "artifacts/model_host"
    required = ["llama-bench", "llama-perplexity", "libllama.so", "libggml.so", "libggml-base.so",
                "libggml-cpu.so", "libggml-htp.so"]
    for name in required:
        if not (model_host / name).is_file():
            raise RuntimeError(f"missing {model_host / name}; run scripts/build_llama_backend.sh")
    skel = PROJECT / "artifacts/fair_combined/libhtp_ops_skel.so"
    ops_host = PROJECT / "artifacts/host/libhtp_ops.so"
    if not skel.is_file():
        raise RuntimeError(f"missing {skel}")
    if not ops_host.is_file():
        raise RuntimeError(f"missing {ops_host}")
    adb_shell(f"mkdir -p {REMOTE}/cdsp", log_dir / "mkdir.log")
    for name in required:
        push_verified(model_host / name, f"{REMOTE}/{name}", log_dir / f"push_host_{name}.log")
    push_verified(ops_host, f"{REMOTE}/{ops_host.name}", log_dir / "push_ops_host.log")
    push_verified(skel, f"{REMOTE}/cdsp/{skel.name}", log_dir / "push_skel.log")
    adb_shell(f"chmod 755 {REMOTE}/llama-bench {REMOTE}/llama-perplexity", log_dir / "chmod.log")
    manifest_device_paths: list[str] = []
    if MODEL_MANIFEST is not None:
        device = dict(MODEL_MANIFEST["device"])
        device_dir = str(device["directory"])
        adb_shell(f"mkdir -p {shlex.quote(device_dir)}", log_dir / "model_mkdir.log")
        local_models = dict(MODEL_MANIFEST["models"])
        model_order = ("cpu_f16", "hmx_f16", "cpu_quant", "hmx_quant")
        for name in model_order:
            record = dict(local_models[name])
            target = str(device["models"][name])
            local_value = record.get("path")
            local = Path(str(local_value)) if local_value else None
            if local is not None and local.is_file():
                push_verified(local, target, log_dir / f"push_model_{name}.log", timeout=3600)
            elif record.get("device_derived") is True:
                expected = str(record["sha256"])
                quoted = shlex.quote(target)
                probe = subprocess.run(
                    ["adb", "shell", f"sha256sum {quoted}; stat -c %s {quoted}"],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                fields = probe.stdout.split()
                if probe.returncode or len(fields) < 3 or fields[0] != expected or int(fields[-1]) != int(record["size"]):
                    raise RuntimeError(f"device-derived model verification failed for {name}: {probe.stdout}")
                evidence = log_dir / f"push_model_{name}.log"
                evidence.write_text(f"reused device-derived model {target} sha256={expected} size={record['size']}\n")
            else:
                raise RuntimeError(f"model {name} has no local file and is not validated as device-derived")
        manifest_device_paths = [str(device["models"][name]) for name in model_order]
    paths = [f"{REMOTE}/{name}" for name in required] + [f"{REMOTE}/libhtp_ops.so",
             f"{REMOTE}/cdsp/libhtp_ops_skel.so",
             *manifest_device_paths, MODEL["dataset"],
             f"{CPU_REMOTE}/llama-perplexity"]
    quoted = " ".join(shlex.quote(p) for p in paths)
    hashes = adb_shell(f"sha256sum {quoted}", log_dir / "device_sha256.log")
    return dict(line.split(maxsplit=1)[::-1] for line in hashes.splitlines() if len(line.split(maxsplit=1)) == 2)


def htp_command(binary: str, mode: str, args: list[str]) -> str:
    rendered = " ".join(shlex.quote(arg) for arg in args)
    return (f"cd {REMOTE} && LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 "
            f"DSP_LIBRARY_PATH='./cdsp;.' LLAMA_NPU_MODE={shlex.quote(mode)} ./{binary} {rendered}")


def require_htp_execution(text: str, mode: str, label: str) -> None:
    if f"STAGE29_BACKEND_CONFIRM backend=my-htp mode={mode}" not in text:
        raise RuntimeError(f"{label} lacks HTP backend confirmation")
    if re.search(r"fallback to CPU|all OPs will fallback", text, re.IGNORECASE):
        raise RuntimeError(f"{label} reported CPU fallback")


def cpu_command(args: list[str]) -> str:
    rendered = " ".join(shlex.quote(arg) for arg in args)
    return (f"cd {CPU_REMOTE} && LD_LIBRARY_PATH=. ./llama-perplexity {rendered}")


def run_ppl(binary_kind: str, mode: str, chunks: int, log: Path) -> tuple[float, list[float], str]:
    model = MODEL["cpu_model"] if binary_kind == "cpu" else MODEL["hmx_model"]
    args = ["-m", model, "-f", MODEL["dataset"], "-fa", "-c", "512", "-b", "512", "-ub", "64",
            "-t", "4", "--chunks", str(chunks)]
    command = cpu_command(args) if binary_kind == "cpu" else htp_command("llama-perplexity", mode, args)
    text = adb_shell(command, log, timeout=3600)
    ppl, chunk_nll = extract_ppl(text)
    if binary_kind == "htp":
        require_htp_execution(text, mode, str(log))
    return ppl, chunk_nll, text


def parse_bench(text: str, mode: str, kind: str, tokens: int, warmup: int = 0,
                measure: int | None = None) -> list[dict[str, object]]:
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "samples_ts" in value:
            samples = [float(sample) for sample in value["samples_ts"]]
            if measure is not None and len(samples) < warmup + measure:
                raise RuntimeError(f"llama-bench returned {len(samples)} samples; expected {warmup + measure}")
            if warmup:
                value["warmup_samples_ts"] = samples[:warmup]
                value["samples_ts"] = samples[warmup:warmup + measure if measure is not None else None]
            value.update({"record_type": "model_throughput", "mode": mode, "kind": kind, "tokens": tokens})
            records.append(value)
    if not records:
        raise RuntimeError("llama-bench returned no JSON sample record")
    return records


def run_throughput(out: Path, raw: Path) -> None:
    modes = ["baseline", "lut-exp", "scna-fp16"]
    for point, prompt in enumerate(MODEL["prompt_tokens"]):
        for session in range(MODEL["sessions"]):
            offset = (point + session) % 3
            order = modes[offset:] + modes[:offset]
            for order_index, mode in enumerate(order):
                common = ["-m", MODEL["hmx_model"], "-fa", "1", "-p", str(prompt), "-n", "0", "-b", "2048",
                          "-ub", "64", "-t", "4", "-o", "jsonl"]
                total = MODEL["warmup"] + MODEL["measure"]
                text = adb_shell(htp_command("llama-bench", mode, common + ["-r", str(total)]),
                                 raw / f"bench_s{session}_{mode}_pp{prompt}.log")
                require_htp_execution(text, mode, f"prompt throughput pp={prompt} session={session}")
                for record in parse_bench(text, mode, "pp", prompt, MODEL["warmup"], MODEL["measure"]):
                    record["session"] = session
                    record["order"] = order_index
                    append(out, record)
    for session in range(MODEL["sessions"]):
        order = modes[session % 3:] + modes[:session % 3]
        for order_index, mode in enumerate(order):
            common = ["-m", MODEL["hmx_model"], "-fa", "1", "-p", "0", "-n",
                      str(MODEL["generation_tokens"]), "-b", "2048", "-ub", "64", "-t", "4", "-o", "jsonl"]
            total = MODEL["warmup"] + MODEL["measure"]
            text = adb_shell(htp_command("llama-bench", mode, common + ["-r", str(total)]),
                             raw / f"bench_s{session}_{mode}_tg.log")
            require_htp_execution(text, mode, f"generation throughput session={session}")
            for record in parse_bench(text, mode, "tg", MODEL["generation_tokens"],
                                      MODEL["warmup"], MODEL["measure"]):
                record["session"] = session
                record["order"] = order_index
                append(out, record)


def parse_kl(text: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    patterns = {
        "mean_kl": r"Mean\s+KLD\s*:\s*([0-9.eE+-]+)",
        "maximum_kl": r"Maximum\s+KLD\s*:\s*([0-9.eE+-]+)",
        "same_top_percent": r"Same top p:\s*([0-9.eE+-]+)",
        "rms_probability_delta_percent": r"RMS Δp\s*:\s*([0-9.eE+-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    if "same_top_percent" not in metrics:
        raise RuntimeError("KL/logit audit output lacks same-top metric")
    return metrics


def run_logit_audit(out: Path, raw: Path) -> None:
    for context in MODEL["logit_contexts"]:
        base = f"{REMOTE}/cpu_logits_ctx{context}.bin"
        args = ["-m", MODEL["cpu_model"], "-f", MODEL["dataset"], "-fa", "-c", str(context),
                "-b", str(context), "-ub", "64", "-t", "4", "--chunks",
                str(MODEL["logit_fixtures_per_context"]), "--save-all-logits", base]
        adb_shell(cpu_command(args), raw / f"logits_cpu_ctx{context}.log", timeout=3600)
        for mode in ("baseline", "lut-exp", "scna-fp16"):
            htp_args = ["-m", MODEL["hmx_model"], "-f", MODEL["dataset"], "-fa", "-c", str(context),
                        "-b", str(context), "-ub", "64", "-t", "4", "--chunks",
                        str(MODEL["logit_fixtures_per_context"]), "--kl-divergence",
                        "--kl-divergence-base", base]
            text = adb_shell(htp_command("llama-perplexity", mode, htp_args),
                             raw / f"logits_{mode}_ctx{context}.log", timeout=3600)
            require_htp_execution(text, mode, f"logit audit context={context}")
            record = parse_kl(text)
            record.update({"record_type": "model_logit", "mode": mode, "context": context,
                           "fixtures": MODEL["logit_fixtures_per_context"]})
            append(out, record)


def main() -> int:
    global SPEC, MODEL, MODEL_MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=PROJECT / "experiment_spec.json")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--out-dir", "--results-dir", dest="out_dir", type=Path,
                        default=PROJECT / "results/formal")
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    SPEC = json.loads(args.spec.resolve().read_text())
    MODEL = dict(SPEC["model"])
    if args.model_manifest:
        MODEL_MANIFEST = json.loads(args.model_manifest.resolve().read_text())
        validation = dict(MODEL_MANIFEST.get("tensor_validation", {}))
        if validation.get("pass") is not True:
            raise RuntimeError("model manifest does not contain a passing tensor-level validation")
        device_models = dict(MODEL_MANIFEST["device"]["models"])
        MODEL["cpu_model"] = device_models["cpu_quant"]
        MODEL["hmx_model"] = device_models["hmx_quant"]
    raw = args.out_dir / "logs/model"
    out = args.out_dir / "raw/model.jsonl"
    try:
        hashes = deploy(raw)
        append(out, {"record_type": "model_provenance", "device_sha256": hashes})
        cpu_ppl, _, cpu_text = run_ppl("cpu", "cpu", 1, raw / "ppl_cpu_sanity.log")
        htp_ppl, _, htp_text = run_ppl("htp", "baseline", 1, raw / "ppl_baseline_sanity.log")
        pair_validation = compare_model_metadata(extract_model_metadata(cpu_text),
                                                 extract_model_metadata(htp_text))
        append(out, {"record_type": "model_pair_validation", **pair_validation})
        relative_delta = abs(htp_ppl / cpu_ppl - 1.0) if cpu_ppl > 0 else math.inf
        gate = (pair_validation["structural_match"] and math.isfinite(cpu_ppl) and math.isfinite(htp_ppl)
                and relative_delta <= MODEL["baseline_cpu_relative_ppl_delta_max"])
        append(out, {"record_type": "model_sanity", "cpu_ppl": cpu_ppl, "htp_baseline_ppl": htp_ppl,
                     "relative_delta": relative_delta, "limit": MODEL["baseline_cpu_relative_ppl_delta_max"],
                     "backend_confirmed": True, "no_cpu_fallback": True,
                     "metadata_structural_match": pair_validation["structural_match"], "pass": gate})
        if not gate:
            print(f"Model hard gate failed: CPU={cpu_ppl:.6g}, HTP={htp_ppl:.6g}, delta={relative_delta:.3%}")
            return 3
        if args.sanity_only:
            return 0
        for binary_kind, mode in [("cpu", "cpu"), ("htp", "baseline"), ("htp", "lut-exp"),
                                  ("htp", "scna-fp16")]:
            ppl, chunk_nll, _ = run_ppl(binary_kind, mode, MODEL["ppl_chunks"], raw / f"ppl_{mode}.log")
            append(out, {"record_type": "model_ppl_summary", "mode": mode, "ppl": ppl,
                         "chunks": len(chunk_nll), "backend_confirmed": binary_kind == "cpu" or True})
            for index, nll in enumerate(chunk_nll, 1):
                append(out, {"record_type": "model_ppl_chunk", "mode": mode, "chunk": index,
                             "mean_nll": nll})
        run_throughput(out, raw)
        run_logit_audit(out, raw)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        append(out, {"record_type": "model_infrastructure_error", "error": str(exc)})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
