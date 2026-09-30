"""One phone-local batch experiment per invocation; no SDK call occurs on import."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from qcphoneperf import Profiler

DEFAULT_METRIC_IDS = (4097, 4182, 4352, 4377, 4480, 4481, 4521)
PROMPT = "The capital of France is"


@dataclass
class CaptureConfig:
    directory: Path
    model: Path
    output_dir: Path
    profiler_payload: Path
    adapter_library: Path
    width: int = 1
    baseline_directory: Path | None = None
    sdk_library: Path | None = None
    executable: str = "hmx-batch-decode"
    reuse_threadpool: bool = False
    reuse_cpu_threadpool: bool = False
    offload_output: bool = False
    split_cpu_ops: bool = False
    fast_concat: bool = False
    hybrid_threads: int | None = None
    shape_aware_vtcm: bool = False
    tokens: int = 12
    sample_ms: int = 60000
    timeout_seconds: int = 90
    metric_ids: tuple[int, ...] = DEFAULT_METRIC_IDS


def normalize_config(options):
    for field in ("directory", "model", "output_dir", "profiler_payload", "adapter_library"):
        setattr(options, field, Path(getattr(options, field)).resolve())
    if options.baseline_directory is not None:
        options.baseline_directory = Path(options.baseline_directory).resolve()
    options.sdk_library = Path(options.sdk_library).resolve() if options.sdk_library is not None else options.profiler_payload / "libs/libQualcommProfilerApi.so"
    for field, low, high in (("width", 1, 32), ("tokens", 1, 120), ("sample_ms", 1, 60000), ("timeout_seconds", 1, 300)):
        value = getattr(options, field)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{field} must be an integer in {low}..{high}")
    if options.hybrid_threads is not None and (type(options.hybrid_threads) is not int or not 1 <= options.hybrid_threads <= 8):
        raise ValueError("hybrid_threads must be None or an integer in 1..8")
    for field in ("reuse_threadpool", "reuse_cpu_threadpool", "offload_output", "split_cpu_ops", "fast_concat", "shape_aware_vtcm"):
        if type(getattr(options, field)) is not bool:
            raise ValueError(f"{field} must be boolean")
    ids = tuple(options.metric_ids)
    if not 1 <= len(ids) <= 8 or len(set(ids)) != len(ids) or any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in ids):
        raise ValueError("metric_ids must contain 1..8 unique uint32 IDs")
    options.metric_ids = ids
    return options


def bounded_integer(minimum, maximum):
    def convert(text):
        try:
            value = int(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError("expected an integer") from error
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(f"expected {minimum}..{maximum}")
        return value
    return convert


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=bounded_integer(1, 32), required=True)
    parser.add_argument("--directory", type=Path, required=True,
                        help="directory containing the test executable and selected backend")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-directory", type=Path, help="optional fallback inference library directory")
    parser.add_argument("--profiler-payload", type=Path, required=True)
    parser.add_argument("--adapter-library", type=Path, required=True)
    parser.add_argument("--sdk-library", type=Path)
    parser.add_argument("--metric-id", dest="metric_ids", type=bounded_integer(0, 0xFFFFFFFF), action="append", help="advertised NPU metric ID; repeat for up to eight unique IDs")
    parser.add_argument("--executable", default="hmx-batch-decode")
    parser.add_argument("--reuse-threadpool", action="store_true")
    parser.add_argument("--reuse-cpu-threadpool", action="store_true",
                        help="attach an ordinary ggml CPU pool in the batch caller; separate from the HTP hybrid pool")
    parser.add_argument("--offload-output", action="store_true")
    parser.add_argument("--split-cpu-ops", action="store_true")
    parser.add_argument("--fast-concat", action="store_true",
                        help="request the optional contiguous dim0 concat fast path; requires a matching backend build")
    parser.add_argument("--hybrid-threads", type=bounded_integer(1, 8))
    parser.add_argument("--shape-aware-vtcm", action="store_true", help="record the compiled DSP variant; this does not change a build")
    parser.add_argument("--tokens", type=bounded_integer(1, 120), default=12,
                        help="output tokens per sequence; conservative bound for fixed context128")
    parser.add_argument("--sample-ms", type=bounded_integer(1, 60000), default=60000)
    parser.add_argument("--timeout-seconds", type=bounded_integer(1, 300), default=90)
    parsed = parser.parse_args(argv)
    if parsed.metric_ids is None:
        parsed.metric_ids = DEFAULT_METRIC_IDS
    return CaptureConfig(**vars(parsed))


def clocks():
    return {"wall_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def save(directory, name, data):
    path = directory / name
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def thermal():
    records = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            name = (zone / "type").read_text().strip()
            if any(part in name.lower() for part in ("cpu", "gpu", "npu", "soc", "skin", "battery", "quiet")):
                records.append({"zone": zone.name, "type": name,
                                "raw_temp": (zone / "temp").read_text().strip()})
        except OSError:
            pass
    return records


def workload_environment(options):
    environment = os.environ.copy()
    inference_directories = [options.directory] + ([options.baseline_directory] if options.baseline_directory is not None else [])
    directories = list(dict.fromkeys(str(path) for path in inference_directories))
    environment["LD_LIBRARY_PATH"] = ":".join(directories + ["/vendor/lib64"])
    dsp_directories = [str(path / "dsp") for path in inference_directories] + [
                      "/vendor/lib/rfsa/adsp", "/vendor/dsp/cdsp", "/vendor/dsp", "/system/lib/rfsa/adsp"]
    dsp = ";".join(dict.fromkeys(dsp_directories))
    environment["ADSP_LIBRARY_PATH"] = dsp
    environment["DSP_LIBRARY_PATH"] = dsp
    environment.pop("HTP_TRACE", None)
    environment.pop("HTP_CPU_TRACE", None)
    environment.pop("SKIP_HTP_OPS", None)
    environment.pop("HTP_REUSE_THREADPOOL", None)
    environment.pop("HTP_OFFLOAD_OUTPUT", None)
    environment.pop("HTP_SPLIT_CPU_OPS", None)
    environment.pop("HTP_FAST_CONCAT", None)
    environment.pop("HTP_HYBRID_THREADS", None)
    if options.reuse_threadpool:
        environment["HTP_REUSE_THREADPOOL"] = "1"
    if options.offload_output:
        environment["HTP_OFFLOAD_OUTPUT"] = "1"
    if options.split_cpu_ops:
        environment["HTP_SPLIT_CPU_OPS"] = "1"
    if options.fast_concat:
        environment["HTP_FAST_CONCAT"] = "1"
    if options.hybrid_threads is not None:
        environment["HTP_HYBRID_THREADS"] = str(options.hybrid_threads)
    return environment, directories


def run_workload(options, environment):
    time.sleep(1.0)
    command = ["/system/bin/timeout", str(options.timeout_seconds),
               str(options.directory / options.executable), "--model", str(options.model),
               "--parallel", str(options.width), "--tokens", str(options.tokens),
               "--context-per-sequence", "128", "--threads", "4", "--ignore-eos",
               "--prompt", PROMPT]
    if options.reuse_cpu_threadpool:
        command.append("--reuse-cpu-threadpool")
    status = {"arguments": command, "cwd": str(options.directory), "started": clocks(),
              "thermal_before": thermal(), "state": "starting", "exit_code": None,
              "exception": None, "stdout_file": "benchmark-stdout.json", "stderr_file": "workload-stderr.log"}
    save(options.output_dir, "workload-status.json", status)
    try:
        with (options.output_dir / "benchmark-stdout.json").open("wb") as stdout, \
             (options.output_dir / "workload-stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=options.directory, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            status.update({"pid": process.pid, "state": "running"})
            save(options.output_dir, "workload-status.json", status)
            status["exit_code"] = process.wait()
    except Exception as error:
        status["exception"] = repr(error)
    status["state"] = "finished"
    status["finished"] = clocks()
    status["thermal_after"] = thermal()
    status["elapsed_seconds"] = (status["finished"]["monotonic_ns"] - status["started"]["monotonic_ns"]) / 1e9
    status["timed_out"] = status["exit_code"] == 124
    stderr_path = options.output_dir / "workload-stderr.log"
    stderr_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
    status["output_offload_requested"] = options.offload_output
    status["output_offload_confirmed_in_stderr"] = "HTP: packed output projection enabled" in stderr_text
    status["fast_concat_requested"] = options.fast_concat
    status["fast_concat_confirmed_in_stderr"] = "HTP: fast dim0 concat enabled" in stderr_text
    status["reuse_threadpool_requested"] = options.reuse_threadpool
    status["reuse_threadpool_confirmed_in_stderr"] = "HTP: persistent CPU threadpool enabled" in stderr_text
    status["reuse_cpu_threadpool_requested"] = options.reuse_cpu_threadpool
    status["reuse_cpu_threadpool_confirmed_in_stderr"] = "batch_decode: ordinary CPU threadpool attached" in stderr_text
    try:
        benchmark = json.loads((options.output_dir / "benchmark-stdout.json").read_text())
        if not isinstance(benchmark, dict) or benchmark.get("mode") != "independent_sequence_decode":
            raise ValueError("unexpected benchmark JSON object")
        save(options.output_dir, "benchmark.json", benchmark)
        status["benchmark_json_valid"] = True
    except Exception as error:
        status["benchmark_json_valid"] = False
        status["benchmark_json_error"] = repr(error)
    save(options.output_dir, "workload-status.json", status)


def run_case(options):
    normalize_config(options)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    environment, directories = workload_environment(options)
    libraries = {}
    for name in ["libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-htp.so", "libhtp_ops.so"]:
        selected = next((Path(directory) / name for directory in directories
                         if (Path(directory) / name).is_file()), None)
        libraries[name] = {"path": str(selected) if selected else None,
                           "sha256": sha256(selected) if selected else None}
    dsp_paths = [Path(directory) / "dsp/libhtp_ops_skel.so" for directory in directories]
    dsp_selected = next((path for path in dsp_paths if path.is_file()), None)
    libraries["libhtp_ops_skel.so"] = {"path": str(dsp_selected) if dsp_selected else None, "sha256": sha256(dsp_selected) if dsp_selected else None}
    save(options.output_dir, "process.json", {
        "pid": os.getpid(), "uid": os.getuid(), "platform": sys.platform, "python": sys.version,
        "runtime_computer": False, "endpoint": None, "adapter_sha256": sha256(options.adapter_library),
        "executable_sha256": sha256(options.directory / options.executable), "libraries": libraries,
        "library_search_directories": directories, "width": options.width, "tokens_per_sequence": options.tokens,
        "ignore_eos": True, "context_per_sequence": 128, "threads": 4, "prompt": PROMPT,
        "model": str(options.model), "selected_metric_ids": list(options.metric_ids),
        "sample_ms_requested": options.sample_ms, "workload_timeout_seconds": options.timeout_seconds,
        "workload_start_delay_seconds": 1.0, "reuse_threadpool_requested": options.reuse_threadpool,
        "reuse_cpu_threadpool_requested": options.reuse_cpu_threadpool,
        "hybrid_threads_requested": options.hybrid_threads,
        "shape_aware_vtcm_requested": options.shape_aware_vtcm,
        "split_cpu_ops_requested": options.split_cpu_ops,
        "output_offload_requested": options.offload_output,
        "fast_concat_requested": options.fast_concat,
        "htp_verbose_trace": False, "started": clocks(),
    })
    if sys.platform != "android":
        raise RuntimeError("Capture must execute in the Android phone Python process")
    if not (options.directory / options.executable).is_file() or not options.model.is_file():
        raise RuntimeError("Selected phone executable or model is missing")
    capture = None
    capture_error = None
    with Profiler(options.profiler_payload, library=options.adapter_library, sdk_library=options.sdk_library) as phone:
        capabilities = phone.capabilities()
        save(options.output_dir, "capabilities.json", capabilities)
        selected = next((item for item in capabilities["capabilities"]
                         if item["name"].split(":")[-1] == "nsp-dsp-metrics"), None)
        if selected is None or not set(options.metric_ids).issubset(selected["metric_ids"]):
            raise RuntimeError("Required NPU metric IDs are not advertised; workload was not started")
        worker = threading.Thread(target=run_workload, args=(options, environment))
        worker.start()
        started = clocks()
        try:
            capture = phone.snapshot_metrics("npu", metric_ids=options.metric_ids, sample_ms=options.sample_ms)
            save(options.output_dir, "capture.json", capture)
        except Exception as error:
            capture_error = {"exception": repr(error), "code": getattr(error, "code", None),
                             "details": getattr(error, "details", None)}
            save(options.output_dir, "capture-error.json", capture_error)
        finally:
            save(options.output_dir, "capture-call.json", {"started": started, "finished": clocks()})
            worker.join()
    save(options.output_dir, "close.json", {"session_close": "completed", "finished": clocks()})
    status = json.loads((options.output_dir / "workload-status.json").read_text())
    case = {"workload_exit_code": status["exit_code"], "workload_timed_out": status["timed_out"],
            "benchmark_json_valid": status["benchmark_json_valid"], "capture_error": capture_error,
            "workload_finished": status["finished"], "reuse_threadpool_requested": options.reuse_threadpool,
            "reuse_threadpool_confirmed_in_stderr": status["reuse_threadpool_confirmed_in_stderr"],
            "reuse_cpu_threadpool_requested": options.reuse_cpu_threadpool,
            "reuse_cpu_threadpool_confirmed_in_stderr": status["reuse_cpu_threadpool_confirmed_in_stderr"],
            "fast_concat_requested": options.fast_concat,
            "fast_concat_confirmed_in_stderr": status["fast_concat_confirmed_in_stderr"]}
    if capture is not None:
        domain = capture["domains"]["npu"]
        case["capture"] = {key: capture.get(key) for key in ["client_mode", "endpoint", "effective_observation_ms",
                           "cleanup_grace_ms", "callback_release_error_count", "cleanup_failed"]}
        case["domain"] = {key: domain.get(key) for key in ["status", "api_status", "stop_api_status", "error",
                          "returned_metric_ids", "missing_metric_ids", "unexpected_metric_ids", "valid_sample_count",
                          "invalid_sample_count", "dropped_sample_count", "truncated"]}
        start, finish = capture["observation_started_wall_ns"], capture["observation_finished_wall_ns"]
        workload_start, workload_finish = status["started"]["wall_ns"], status["finished"]["wall_ns"]
        case["coverage"] = {
            "full_workload_lifetime_inside_observation": workload_start >= start and workload_finish <= finish,
            "unobserved_prefix_seconds": max(0, start - workload_start) / 1e9,
            "unobserved_tail_seconds": max(0, workload_finish - finish) / 1e9,
            "start_relative_to_observation_seconds": (workload_start - start) / 1e9,
            "finish_relative_to_observation_seconds": (workload_finish - start) / 1e9,
        }
    save(options.output_dir, "case-summary.json", case)
    if options.offload_output and not status["output_offload_confirmed_in_stderr"]:
        return 1
    if options.reuse_cpu_threadpool and not status["reuse_cpu_threadpool_confirmed_in_stderr"]:
        return 1
    if options.fast_concat and not status["fast_concat_confirmed_in_stderr"]:
        return 1
    if capture_error or status["exit_code"] != 0 or not status["benchmark_json_valid"]:
        return 1
    domain = capture["domains"]["npu"]
    if domain["status"] != "ok" or domain["api_status"] != 0 or domain["stop_api_status"] != 0 \
            or capture["cleanup_failed"] or capture["callback_release_error_count"] != 0:
        return 1
    return 0


def capture_case(config: CaptureConfig) -> dict:
    """Invoke the phone-local profiling API and return the retained case summary.

    SDK/initialization exceptions propagate. A returned exit_status != 0 records
    a collection/workload failure; callers must also inspect the coverage flag.
    """
    exit_status = run_case(config)
    result = json.loads((config.output_dir / "case-summary.json").read_text())
    result["exit_status"] = exit_status
    return result


if __name__ == "__main__":
    result = capture_case(arguments())
    print(json.dumps(result, allow_nan=False), flush=True)
    raise SystemExit(result["exit_status"])
