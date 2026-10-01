"""Export larger-model measurements without distributing Qualcomm SDK data.

Only measured packet values, timing/coverage, inference output, public hashes,
process settings and collector outcomes are exported. SDK binaries, database
contents/schema, callback payloads and proprietary implementation code are not.
The eight successful inference captures are required; an incomplete or failed
case raises rather than publishing a partial inventory as a completed result.

This module is importable; export(root, destination) takes the directory that
contains qwen-{0.5b,1.5b}-q4f16-m{1,32} and their -repeat counterparts. Optional
paths select the already exported continuous-control evidence and CPU output.
Optional numerical trials and failed setup/mapping outcomes are retained
separately from the eight completed activity captures.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ANALYZER_PATH = REPO / "experiments/load-calibration/summarize_intervals.py"
CASE_NAMES = tuple(
    f"qwen-{size}-q4f16-m{width}{suffix}"
    for suffix in ("", "-repeat")
    for size, width in (("0.5b", 1), ("1.5b", 1), ("0.5b", 32), ("1.5b", 32))
)
MODELS = {
    "qwen2.5-0.5b-q4_0-f16-hmx.gguf": {
        "source_model": "Qwen2.5-0.5B", "variant": "base", "size_label": "0.5B",
        "sha256": "6279420c1381e9ff40626c827411eef241df7411edcd56c89a3179ae34945227",
    },
    "qwen2.5-1.5b-instruct-q4_0-f16-hmx.gguf": {
        "source_model": "Qwen2.5-1.5B-Instruct", "variant": "instruct", "size_label": "1.5B",
        "sha256": "eb4e9683beac9d347fed3444ceebe24994e3005004caa09fe2a7557698c8a72f",
        "bytes": 1504724992,
        "source_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    },
}
CPU_MODEL = {
    "basename": "qwen2.5-1.5b-instruct-effective-f16-cpu.gguf",
    "sha256": "41cedabf5361e664a81a0eb03ca7768701758b6ac17377acb91bb2b9b3d9432f",
    "bytes": 2818361344,
    "construction": "Invert HTP Q4_0 HVX superpack, dequantize to the same rounded FP16 HMX weights, invert tile permutation; preserve ordinary Q6_K embedding bytes.",
}
METRIC_IDS = {
    "4097": "QDSP6 Load MCPS", "4182": "QDSP Clock MHz",
    "4352": "SDK HVX Utilization percent", "4377": "SDK HVX active MCPS",
    "4480": "SDK HMX Utilization percent", "4481": "SDK HMX active MCPS",
    "4521": "SDK HMX Clock MHz",
}
PHONE_STAGE = "/data/local/tmp/llama-v68-larger-20260930"


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyzer():
    spec = importlib.util.spec_from_file_location("larger_model_interval_analysis", ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_paths(value):
    """Retain reproducible relative phone layout, omit private host paths."""
    if isinstance(value, dict):
        return {key: normalize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_paths(item) for item in value]
    if isinstance(value, str):
        return value.replace(PHONE_STAGE, "$PHONE_STAGE").replace(
            "/home/qwu26/llama-v68-port-20260929", "$HOST_WORKSPACE")
    return value


def normalize_workload(value):
    value = normalize_paths(value)
    if "model" in value:
        value["model"] = Path(value["model"]).name
    return value


def selected_window(summary, name):
    windows = [window for window in summary["windows"] if window["name"] == name]
    if len(windows) != 1:
        raise ValueError(f"Expected one {name} measurement window")
    return windows[0]


def controls(calibration, calibration_path):
    cases = {case["name"]: case for case in calibration["cases"]}
    result = {}
    for engine, metric in (("hvx", "4377"), ("hmx", "4481")):
        case = cases[f"{engine}-d100"]
        window = selected_window(case["summary"], "dsp_measurement")
        rate = window["metrics"][metric]["time_weighted_mean"]
        clock = window["metrics"]["4182"]["time_weighted_mean"]
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"Invalid {engine} continuous-control activity rate")
        result[engine] = {
            "case": case["name"], "activity_metric_id": int(metric),
            "time_weighted_activity_mcps": rate, "time_weighted_qdsp_clock_mhz": clock,
            "reference_clock_equivalent_percent": window["metrics"][metric]["reference_clock_equivalent_percent"],
            "complete_intervals": window["complete_intervals"],
            "coverage_fraction": window["coverage_fraction"],
            "capture_sha256": case["capture_sha256"],
            "measurement_window": window,
        }
    return {
        "evidence_file": calibration_path.name, "evidence_sha256": sha256(calibration_path),
        "interpretation": "Activity relative to this device's continuous single-context control at the reported clock; neither engine capacity nor active rows/lanes.",
        "engines": result,
    }


def control_activity(summary, calibration_controls):
    result = []
    for window in summary["windows"]:
        engines = {}
        for engine, metric in (("hvx", "4377"), ("hmx", "4481")):
            measured = window["metrics"].get(metric)
            clock = window["metrics"].get("4182")
            control = calibration_controls["engines"][engine]
            engines[engine] = {
                "activity_metric_id": int(metric),
                "time_weighted_activity_mcps": measured["time_weighted_mean"] if measured else None,
                "reference_clock_equivalent_percent": measured["reference_clock_equivalent_percent"] if measured else None,
                "relative_continuous_control_activity_percent": (
                    100 * measured["time_weighted_mean"] / control["time_weighted_activity_mcps"]
                    if measured else None
                ),
                "time_weighted_qdsp_clock_mhz": clock["time_weighted_mean"] if clock else None,
                "same_mean_reference_clock_within_0_1_mhz": (
                    abs(clock["time_weighted_mean"] - control["time_weighted_qdsp_clock_mhz"]) <= 0.1
                    if clock else None
                ),
                "continuous_control_case": control["case"],
            }
        result.append({
            "window": window["name"], "complete_intervals": window["complete_intervals"],
            "covered_seconds": window["covered_seconds"],
            "coverage_fraction": window["coverage_fraction"], "engines": engines,
        })
    return result


def compare_sequence(sequence, reference_sequence):
    actual = sequence["token_ids"]
    reference = reference_sequence["token_ids"]
    overlap = min(len(actual), len(reference))
    first_difference = next((i for i in range(overlap) if actual[i] != reference[i]), None)
    prefix = overlap if first_difference is None else first_difference
    return {
        "seq_id": sequence["seq_id"], "generated_tokens": len(actual),
        "compared_tokens": overlap, "matching_prefix_tokens": prefix,
        "first_differing_token_index_zero_based": first_difference,
        "reference_token_at_first_difference": reference[first_difference] if first_difference is not None else None,
        "actual_token_at_first_difference": actual[first_difference] if first_difference is not None else None,
        "all_generated_tokens_match_cpu_prefix": first_difference is None and len(actual) <= len(reference),
    }


def collector_outcome(capture):
    # No metric definitions, database tables, JSON schema, or callback payloads.
    top_keys = (
        "schema_version", "source", "client_mode", "endpoint",
        "snapshot_started_wall_ns", "snapshot_finished_wall_ns",
        "observation_started_wall_ns", "observation_finished_wall_ns",
        "requested_observation_ms", "effective_observation_ms", "cleanup_grace_ms",
        "callback_count", "unmatched_callback_count", "callback_release_error_count", "cleanup_failed",
        "metadata_database_sha256", "metadata_database_version",
    )
    domain_keys = (
        "status", "api_status", "stop_api_status", "error",
        "requested_sampling_ms", "requested_streaming_ms", "valid_sample_count",
        "invalid_sample_count", "dropped_sample_count", "truncated",
        "callback_batch_count", "empty_callback_batch_count", "buffer_record_count",
        "buffer_byte_count", "requested_metric_ids", "returned_metric_ids",
        "missing_metric_ids", "unexpected_metric_ids", "response_types",
    )
    result = {key: capture[key] for key in top_keys if key in capture}
    result["domains"] = {
        name: {key: domain[key] for key in domain_keys if key in domain}
        for name, domain in capture["domains"].items()
    }
    return result


def measured_packets(capture, analysis):
    records = analysis.packets(capture)
    reported_durations = {}
    for record in capture["domains"]["npu"]["metrics"]:
        if record["id"] != 4481:
            continue
        values = [analysis.raw(value) for value in record["params"]]
        key = (record["source_timestamp"]["value"], values[4])
        reported_durations[key] = values[7]
    for record in records:
        # Retain the SDK's reported duration as well as its raw QTimer ticks
        # and the exact 19.2 MHz duration used by the analysis.
        record["sdk_reported_duration_ms"] = reported_durations[(record["sdk_end_ns"], record["index"])]
    return records


def numerical_evidence(directory):
    """Preserve optional numerical trials and parse only printed tool metrics."""
    directory = Path(directory)
    commands = read(directory / "commands.json")
    if not isinstance(commands, list) or not commands:
        raise ValueError(f"Expected numerical command inventory: {directory}")
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    row = re.compile(r"^\s*(\d+)\s+" + r"\s*±\s*".join([number] * 2) +
                     (r"\s+" + r"\s*±\s*".join([number] * 2)) * 2 +
                     (r"\s+" + r"\s*±\s*".join([number] * 2) + r"\s*%") * 2 + r"\s*$")
    result = {
        "directory": directory.name, "commands": normalize_paths(commands),
        "files": {}, "teacher_forced_metrics": [], "cpu_runtime_features": [],
        "baseline_format": {
            "encoding": "Per token, two FP32 scale/offset values followed by uint16 scaled log-probabilities, with vocabulary padding; not raw FP32 logits.",
            "quantization": "The minimum logit is clamped to max_logit - 16; log-probability codes use (max_logit - min_logit)/65535 scale and nearest-integer rounding.",
            "kl_policy": "Reported approximate CPU-baseline-to-HMX KL uses decoded saved log-probabilities and excludes baseline log-probabilities <= -16 without renormalization; units are nats per scored position.",
            "probability_rms_policy": "Reported delta-p RMS is the ground-truth next-token probability difference across scored positions, expressed in percentage points, not a whole-vocabulary RMS.",
            "uncertainty_policy": "Printed plus/minus values are the tool's estimated standard errors, with delta-method propagation for perplexity/RMS; they are not 95-percent confidence intervals.",
            "source": "llama.cpp-npu/examples/perplexity/perplexity.cpp",
            "source_sha256": sha256(REPO / "llama.cpp-npu/examples/perplexity/perplexity.cpp"),
        },
    }
    for command in commands:
        label = command.get("label")
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError(f"Invalid numerical command label: {label!r}")
        if command.get("returncode") != 0:
            raise ValueError(f"Numerical command failed: {label}")
        for suffix in ("stdout", "stderr"):
            path = directory / f"{label}.{suffix}"
            content = path.read_text()
            result["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            if suffix == "stderr":
                dispatches = re.findall(
                    r"HTP DSP completed: opcode=(\d+) count=\d+ tensor=\S+ op=(\S+) elapsed_us=\d+ status=(-?\d+)",
                    content)
                shapes = re.findall(r"HTP matrix shape: tensor=\S+ M=(\d+) K=(\d+) N=(\d+)", content)
                if dispatches:
                    result.setdefault("dsp_trace_summaries", {})[label] = {
                        "scope": "Whole-process trace; not phase-aligned. Perplexity includes its default warmup.",
                        "completed_dispatches": len(dispatches),
                        "op_counts": dict(Counter(op for _, op, _ in dispatches)),
                        "opcode_counts": dict(Counter(opcode for opcode, _, _ in dispatches)),
                        "status_counts": dict(Counter(status for _, _, status in dispatches)),
                        "matrix_shapes": [
                            {"m": int(m), "k": int(k), "n": int(n), "occurrences": count}
                            for (m, k, n), count in sorted(Counter(shapes).items())
                        ],
                    }
            if suffix == "stdout" and content.lstrip().startswith("{"):
                result.setdefault("inference_outputs", {})[label] = normalize_workload(json.loads(content))
            for line in content.splitlines():
                match = row.fullmatch(line)
                if match:
                    values = [float(value) for value in match.groups()[1:]]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"Nonfinite numerical metric: {path.name}")
                    names = ("perplexity", "ln_perplexity_ratio", "approximate_kl_cpu_to_hmx",
                             "ground_truth_probability_delta_rms_percent", "same_top_probability_percent")
                    metric = {
                        "label": label, "source_file": path.name, "printed_line": line,
                        "chunk": int(match.group(1)),
                        **{name: {"value": values[2 * i], "reported_uncertainty": values[2 * i + 1]}
                           for i, name in enumerate(names)},
                    }
                    arguments = command["command"]
                    if "-c" in arguments:
                        context = int(arguments[arguments.index("-c") + 1])
                        metric["scored_positions_from_cli_and_source"] = int(match.group(1)) * (context - 1 - context // 2)
                        metric["requested_context"] = context
                    result["teacher_forced_metrics"].append(metric)
                baseline = re.search(r"Final estimate: PPL = " + number + r"\s*\+/-\s*" + number, line)
                if baseline:
                    result.setdefault("baseline_perplexity", []).append({
                        "label": label, "source_file": path.name, "printed_line": line,
                        "value": float(baseline.group(1)), "reported_uncertainty": float(baseline.group(2)),
                    })
                if "system_info:" in line and "CPU :" in line:
                    result["cpu_runtime_features"].append({
                        "label": label, "source_file": path.name, "printed_line": line,
                        "fp16_va_reported": bool(re.search(r"\bFP16_VA\s*=\s*1\b", line)),
                    })
    result["files"]["commands.json"] = {"sha256": sha256(directory / "commands.json")}
    summary_path = directory / "summary.json"
    if summary_path.exists():
        result["summary_diagnostics"] = normalize_paths(read(summary_path))
        fixture = REPO / "experiments/larger-model/teacher-forced-input.txt"
        if result["summary_diagnostics"].get("fixture_sha256") != sha256(fixture):
            raise ValueError("Committed numerical fixture differs from the executed fixture")
        result["files"]["summary.json"] = {"sha256": sha256(summary_path)}
    result["cpu_precision_inference"] = {
        "interpretation": "An absent FP16_VA entry in the CPU runtime feature line means the native FP16 vector-arithmetic macro is disabled in that CPU backend; its generic F16 dot-product fallback converts half operands to FP32 arithmetic. This does not imply every CPU operator uses identical arithmetic to HMX.",
        "source_files": {name: sha256(REPO / name) for name in (
            "llama.cpp-npu/ggml/src/ggml-cpu/ggml-cpu.cpp",
            "llama.cpp-npu/ggml/src/ggml-cpu/ggml-cpu.c",
        )},
    }
    if not result["teacher_forced_metrics"]:
        raise ValueError(f"No numerical comparison metric row found: {directory}")
    return result


def failed_outcomes(retrieved):
    """Keep unsuccessful setup attempts separate from completed model runs."""
    retrieved = Path(retrieved)
    result = {"numerical_setup": [], "full_f16_mapping": []}
    for basename in ("numerical-v1", "numerical-v2", "numerical-v3"):
        directory = retrieved / basename
        if not directory.exists():
            continue
        commands_path = directory / "commands.json"
        commands = read(commands_path)
        item = {"directory": basename, "commands": normalize_paths(commands),
                "files": {"commands.json": {"sha256": sha256(commands_path)}}, "error_excerpts": []}
        for command in commands:
            label = command["label"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or command.get("returncode") == 0:
                raise ValueError(f"Unexpected setup-failure command: {basename}/{label}")
            for suffix in ("stdout", "stderr"):
                path = directory / f"{label}.{suffix}"
                content = path.read_text()
                item["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
                item["error_excerpts"].extend(normalize_paths(line) for line in content.splitlines()
                                              if "Permission denied" in line or "invalid argument" in line)
        if any("Permission denied" in line and "cache" in line for line in item["error_excerpts"]):
            item["classification"] = "Setup failure: cache-directory access permission; no numerical model result."
        elif any("invalid argument: --no-warmup" in line for line in item["error_excerpts"]):
            item["classification"] = "Setup failure: this tool version does not support --no-warmup; no numerical model result."
        else:
            item["classification"] = "Unclassified unsuccessful setup attempt; inspect hashed logs."
        result["numerical_setup"].append(item)
    for basename in ("smoke", "smoke-map512", "smoke-map256", "smoke-map256-buf64"):
        commands_path = retrieved / "results" / f"{basename}-command.json"
        if not commands_path.exists():
            continue
        log_path = commands_path.with_name(basename + ".log")
        lines = log_path.read_text().splitlines()
        if not any("fastrpc_mmap failed" in line for line in lines):
            raise ValueError(f"Missing observed mapping failure in {log_path}")
        command = read(commands_path)
        item = {
            "name": basename, "execution": normalize_paths(command),
            "command_sha256": sha256(commands_path), "stderr_sha256": sha256(log_path),
            "classification": "Full-F16 experiment did not complete; observed FastRPC/DMA mapping failure does not establish a hardware model-size limit.",
            "error_excerpts": normalize_paths(lines[-20:]),
        }
        output_path = commands_path.with_name(basename + ".json")
        if output_path.exists():
            item["stdout_sha256"] = sha256(output_path)
            content = output_path.read_text().strip()
            if content:
                item["partial_output"] = normalize_paths(json.loads(content))
        result["full_f16_mapping"].append(item)
    return result


def export(root, destination, *, calibration_path=None, cpu_reference_path=None,
           numerical_path=None, numerical_attention_path=None):
    root, destination = Path(root), Path(destination)
    calibration_path = Path(calibration_path or REPO / "results/LOAD-CALIBRATION-20260930.json")
    cpu_reference_path = Path(cpu_reference_path or root.parent / "retrieved/results/cpu-effective-reference12.json")
    missing = [name for name in CASE_NAMES if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Eight completed captures required; missing: {', '.join(missing)}")
    analysis = analyzer()
    reference = normalize_workload(read(cpu_reference_path))
    if reference["model"] != CPU_MODEL["basename"] or len(reference["sequences"]) != 1:
        raise ValueError("Unexpected effective CPU reference model or sequence count")
    reference_sequence = reference["sequences"][0]
    if len(reference_sequence["token_ids"]) != 12:
        raise ValueError("Expected twelve-token effective CPU reference")
    calibrated_controls = controls(read(calibration_path), calibration_path)
    evidence = {
        "schema_version": 1,
        "experiment": "Snapdragon 888 Qwen2.5 0.5B versus 1.5B mixed Q4_0+F16 phone-local HMX/HVX activity",
        "host_recorded_date": "2026-09-30",
        "device": {"model": "OnePlus 9 LE2115", "soc": "SM8350", "hexagon": "v68", "android": "14"},
        "runtime_computer": False,
        "scope": "One phone, SDK binary and prompt; two runs per configuration. Activity counters and continuous-control ratios do not measure peak capacity, useful rows or lane occupancy.",
        "comparison_constraints": [
            "Both model sizes use mixed Q4_0+F16 layer weights and ordinary Q6_K tied embeddings; HMX computation is FP16.",
            "0.5B is the base model; 1.5B is the instruct model. This is not a model-quality comparison or a controlled architecture-only scaling study.",
            "The CPU output projection is enabled for both sizes; HTP_OFFLOAD_OUTPUT=0.",
            "M1 runs generate twelve tokens per sequence; M32 runs generate two. Compare sizes within the same width.",
            "Repeat order, timestamps, thermal readings and reported DSP clocks are retained; thermal/order effects are not assumed absent.",
            "Only complete intervals inside each phase union contribute; coverage and uncovered time accompany every ratio.",
            "The effective CPU reference preserves the same dequantized FP16 weights, but CPU activation/accumulation differs from HMX; greedy prefix agreement alone is a bounded check.",
        ],
        "metric_ids": METRIC_IDS,
        "models": MODELS,
        "calibrated_continuous_controls": calibrated_controls,
        "analysis_code_sha256": sha256(ANALYZER_PATH),
        "exporter_code_sha256": sha256(Path(__file__)),
        "public_source_hashes": {
            name: sha256(REPO / name) for name in (
                "experiments/load-calibration/capture_workload.py",
                "experiments/load-calibration/inference/batch_decode.cpp",
                "experiments/larger-model/reconstruct_reference.py",
            )
        },
        "effective_cpu_reference": {
            "model": CPU_MODEL, "stdout_sha256": sha256(cpu_reference_path),
            "workload": reference,
            "performance_policy": "Reference timings are preserved as diagnostics; no controlled CPU-versus-HMX speedup claim is made.",
        },
        "cases": [],
    }
    status_path = root.parent / "status.json"
    if status_path.exists():
        status = read(status_path)
        evidence["artifact_provenance"] = {
            key: status[key] for key in ("isolated_dsp_skel_sha256", "isolated_android_backend_sha256")
            if key in status
        }
    for key, supplied, basename in (
        ("default_hmx_attention", numerical_path, "numerical-v4"),
        ("cpu_attention_isolation", numerical_attention_path, "numerical-attention-cpu-v1"),
    ):
        directory = Path(supplied) if supplied is not None else root.parent / "retrieved" / basename
        if supplied is not None or directory.exists():
            evidence.setdefault("numerical_validation", {})[key] = numerical_evidence(directory)
    outcomes = failed_outcomes(root.parent / "retrieved")
    if any(outcomes.values()):
        evidence["unsuccessful_experiment_outcomes"] = outcomes
    final_check_path = root.parent / "retrieved/final-device-check.json"
    if final_check_path.exists():
        observations = normalize_paths(read(final_check_path))
        observations["kernel_mapping_failure_excerpt"] = [
            line for line in observations.get("kernel_mapping_failure_excerpt", [])
            if "dma_buf_map_attachment" in line or "fastrpc_internal_mem_map failed" in line
        ]
        evidence["final_device_check"] = {
            "source_sha256": sha256(final_check_path), "observations": observations,
        }
    for name in CASE_NAMES:
        directory = root / name
        status = read(directory / "summary.json")
        execution = read(directory / "workload.json")
        if status.get("returncode") != 0 or execution.get("returncode") != 0 or status.get("errors"):
            raise ValueError(f"Inference case did not complete successfully: {name}: {status}")
        capture = read(directory / "capture.json")
        summary = analysis.analyze(directory)
        phase_output = normalize_workload(read(directory / "stdout.json"))
        process = read(directory / "process.json")
        if (process.get("platform") != "android" or process.get("runtime_computer") is not False
                or process.get("endpoint") is not None):
            raise ValueError(f"Collector did not run locally on Android: {name}")
        collector_source_sha = evidence["public_source_hashes"]["experiments/load-calibration/capture_workload.py"]
        if process.get("runner_sha256") != collector_source_sha:
            raise ValueError(f"Captured collector source differs from public source: {name}")
        if process["config"]["command"] != execution["command"]:
            raise ValueError(f"Configured and executed command disagree: {name}")
        width = 32 if "-m32" in name else 1
        expected_tokens = 2 if width == 32 else 12
        if (phase_output["parallel_requested"] != width
                or phase_output["tokens_limit_per_sequence"] != expected_tokens
                or len(phase_output["sequences"]) != width):
            raise ValueError(f"Unexpected width, token limit or sequence count: {name}")
        model = MODELS.get(phase_output["model"])
        if model is None:
            raise ValueError(f"Unrecognized model basename: {phase_output['model']}")
        case = {
            "name": name, "repeat": name.endswith("-repeat"),
            "model": {"basename": phase_output["model"], **model},
            "status": normalize_paths(status), "summary": summary,
            "workload": phase_output,
            "process": normalize_paths(process),
            "workload_execution": normalize_paths(execution),
            "collector_outcome": collector_outcome(capture),
            "collector_close": read(directory / "close.json"),
            "capture_sha256": sha256(directory / "capture.json"),
            "stdout_sha256": sha256(directory / "stdout.json"),
            "stderr_sha256": sha256(directory / "stderr.log"),
            "packets": measured_packets(capture, analysis),
            "control_activity_by_window": control_activity(summary, calibrated_controls),
        }
        if model["size_label"] == "1.5B":
            comparisons = [compare_sequence(sequence, reference_sequence) for sequence in phase_output["sequences"]]
            case["effective_cpu_greedy_comparison"] = {
                "sequences": comparisons,
                "all_sequences_match_cpu_prefix": all(c["all_generated_tokens_match_cpu_prefix"] for c in comparisons),
                "minimum_matching_prefix_tokens": min(c["matching_prefix_tokens"] for c in comparisons),
                "policy": "Compare each generated sequence against the same-weight CPU reference; preserve first differing token rather than claiming all tokens match.",
            }
        evidence["cases"].append(case)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--cpu-reference", type=Path)
    parser.add_argument("--numerical", type=Path)
    parser.add_argument("--numerical-attention", type=Path)
    args = parser.parse_args()
    result = export(args.root, args.destination, calibration_path=args.calibration,
                    cpu_reference_path=args.cpu_reference, numerical_path=args.numerical,
                    numerical_attention_path=args.numerical_attention)
    print(json.dumps({"cases": len(result["cases"]), "output": str(args.destination)}))


if __name__ == "__main__":
    main()
