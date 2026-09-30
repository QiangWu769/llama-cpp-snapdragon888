"""Research analysis for the verified v68 SysMon capture schema.

Select complete reported intervals only; never interpolate phase boundaries.
Coverage is an interval union. Counter integration retains each SDK-reported
interval duration, including the sub-microsecond timestamp rounding tolerance.
The private phone clock-control establishes CNTVCT/QTimer alignment separately.
Only the statically verified SDK binary below uses this inverse timestamp map.
Rates and SDK ratios are not calibrated physical engine capacity percentages.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

IDS = [4097, 4182, 4352, 4377, 4480, 4481, 4521]
QTIMER_HZ = 19200000
BOUNDARY_TOLERANCE_NS = 1000
SDK_TICK_NS = 52.08333
SDK_BINARY_SHA256 = "bb37729aaf6c54f0dc5207d5d14430a3224b7127c67a665ddd24ab325946fb57"


def raw(value):
    return value["value"] if isinstance(value, dict) else value


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def unsigned_integer(value, label):
    finite_number(value, label)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be an unsigned integer")
    return value


def sdk_timestamp_to_qtimer_ticks(timestamp):
    """Invert the verified binary64 constant, never assume SDK ns are exact ns."""
    unsigned_integer(timestamp, "source timestamp")
    # Integer decimal inversion avoids rounding the ~4e13-tick quotient.
    quotient, remainder = divmod(timestamp * 100000, 5208333)
    ticks = quotient + (2 * remainder >= 5208333)
    # Real SDK converts integer ticks to double, multiplies, then truncates.
    if abs(int(float(ticks) * SDK_TICK_NS) - timestamp) > 1:
        raise ValueError("source timestamp does not fit the verified SDK tick conversion")
    return ticks


def union_intervals(windows):
    """Merge overlap/adjacency without filling any actual gap."""
    intervals = []
    for start, end in windows:
        finite_number(start, "window start")
        finite_number(end, "window end")
        if end <= start:
            raise ValueError("nonpositive window")
        intervals.append((start, end))
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def packets(capture):
    domain = capture["domains"]["npu"]
    if (domain["status"] != "ok" or domain["dropped_sample_count"] or domain["truncated"]
            or domain.get("invalid_sample_count", 0) or domain.get("stop_api_status", 0)
            or capture.get("cleanup_failed", False) or capture.get("callback_release_error_count", 0)):
        raise ValueError("capture is not complete and successful")
    grouped = defaultdict(dict)
    for record in domain["metrics"]:
        if record["id"] not in IDS:
            continue
        params = list(map(raw, record["params"]))
        if len(params) != 12 or params[0] != 3 or params[8] != 7:
            raise ValueError("unexpected private experiment schema or DSP core")
        if record["status"] != "reported":
            raise ValueError("invalid metric status")
        finite_number(record["value"], "metric value")
        end = unsigned_integer(record["source_timestamp"]["value"], "source timestamp")
        index = unsigned_integer(params[4], "packet index")
        key = (end, index)
        if record["id"] in grouped[key]:
            raise ValueError("duplicate metric")
        grouped[key][record["id"]] = record
    result = []
    for (end, index), values in sorted(grouped.items()):
        if set(values) != set(IDS):
            raise ValueError("unaligned metric group")
        params = list(map(raw, values[4481]["params"]))
        for v in values.values():
            p = list(map(raw, v["params"]))
            if p[4:] != params[4:]:
                raise ValueError("unaligned metric interval metadata")
        ticks = unsigned_integer(params[6], "SysMon interval ticks")
        measured_ms = finite_number(params[7], "SysMon measuredTime_ms")
        unsigned_integer(params[9], "DSP packet type")
        dt_ns = ticks * 1e9 / QTIMER_HZ
        if dt_ns <= 0 or abs(dt_ns / 1e6 - measured_ms) > 0.001:
            raise ValueError("measuredTime disagrees with 19.2 MHz tick interval")
        end_ticks = sdk_timestamp_to_qtimer_ticks(end)
        corrected_end = end_ticks * 1e9 / QTIMER_HZ
        result.append({"start_ns": (end_ticks - ticks) * 1e9 / QTIMER_HZ,
                       "end_ns": corrected_end, "duration_ns": dt_ns,
                       "sdk_end_ns": end, "qtimer_end_ticks": end_ticks, "duration_ticks": ticks,
                       "index": index, "dsp_packet_type": params[9],
                       "values": {str(k): v["value"] for k, v in values.items()}})
    if not result:
        raise ValueError("capture contains no complete metric packets")
    for a, b in zip(result, result[1:]):
        if b["index"] != a["index"] + 1 or abs(b["start_ns"] - a["end_ns"]) > BOUNDARY_TOLERANCE_NS:
            raise ValueError("sample sequence has gaps or overlaps beyond timestamp rounding tolerance")
    return result


def summarize_union(records, windows):
    windows = union_intervals(windows)
    if not windows:
        raise ValueError("empty window union")
    # Each packet appears once, even when caller windows overlap.
    selected = [p for p in records if any(p["start_ns"] >= start and p["end_ns"] <= end
                                          for start, end in windows)]
    reported_duration = math.fsum(p["duration_ns"] for p in selected)
    selected_union = union_intervals((p["start_ns"], p["end_ns"]) for p in selected)
    covered_duration = math.fsum(end - start for start, end in selected_union)
    window_duration = math.fsum(end - start for start, end in windows)
    metrics = {}
    if selected:
        clock_integral = math.fsum(p["values"]["4182"] * p["duration_ns"] / 1e9 for p in selected)
        for identifier in IDS:
            key = str(identifier)
            values = [p["values"][key] for p in selected]
            integral = math.fsum(p["values"][key] * p["duration_ns"] / 1e9 for p in selected)
            metrics[key] = {"time_weighted_mean": integral / (reported_duration / 1e9),
                            "minimum": min(values), "maximum": max(values)}
            if identifier in (4097, 4377, 4481):
                metrics[key]["reported_mcycles"] = integral
            if identifier in (4377, 4481):
                metrics[key]["reference_clock_equivalent_percent"] = (
                    100 * integral / clock_integral if clock_integral > 0 else None)
    return {"window_start_ns": windows[0][0], "window_end_ns": windows[-1][1],
            "window_intervals_ns": windows, "window_seconds": window_duration / 1e9,
            "window_envelope_seconds": (windows[-1][1] - windows[0][0]) / 1e9,
            "complete_intervals": len(selected), "covered_seconds": covered_duration / 1e9,
            "covered_intervals_ns": selected_union,
            "reported_interval_seconds": reported_duration / 1e9,
            "interval_overlap_seconds": max(0.0, reported_duration - covered_duration) / 1e9,
            "coverage_fraction": min(1.0, covered_duration / window_duration),
            "uncovered_seconds": max(0.0, window_duration - covered_duration) / 1e9,
            "interval_ms_range": [min(p["duration_ns"] for p in selected)/1e6,
                                  max(p["duration_ns"] for p in selected)/1e6] if selected else None,
            "packet_types": sorted({p["dsp_packet_type"] for p in selected}), "metrics": metrics}


def summarize(records, start_ns, end_ns):
    return summarize_union(records, [(start_ns, end_ns)])


def intersection_duration(left, right):
    left, right = union_intervals(left), union_intervals(right)
    i = j = 0
    lengths = []
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        lengths.append(max(0, min(a[1], b[1]) - max(a[0], b[0])))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return math.fsum(lengths)


def calibration_active_regions(benchmark):
    periods = benchmark.get("periods")
    if not isinstance(periods, list) or not periods:
        raise ValueError("calibration workload has no measured period records")
    start = unsigned_integer(benchmark["qtimer_start"], "workload QTimer start")
    end = unsigned_integer(benchmark["qtimer_end"], "workload QTimer end")
    regions, total_active, previous_end = [], 0, start
    for expected_index, period in enumerate(periods):
        a = unsigned_integer(period["start_qtimer"], "period QTimer start")
        b = unsigned_integer(period["end_qtimer"], "period QTimer end")
        active = unsigned_integer(period["active_qtimer"], "period active ticks")
        if (period["index"] != expected_index or a < previous_end or b <= a
                or b > end or active > b - a):
            raise ValueError("invalid calibration period boundaries")
        if active:
            regions.append((a * 1e9 / QTIMER_HZ, (a + active) * 1e9 / QTIMER_HZ))
        total_active += active
        previous_end = b
    if total_active != benchmark["active_qtimer"]:
        raise ValueError("period active ticks disagree with workload total")
    return regions


def analyze(directory):
    capture = json.loads((directory / "capture.json").read_text())
    records = packets(capture)
    output = {"schema_version": 2, "case": directory.name,
              "interpretation": "Time-weighted SDK counter rates and SDK ratios; physical capacity uncalibrated",
              "boundary_policy": "Only complete intervals inside the window union; no boundary interpolation",
              "integration_policy": "Each selected SDK interval integrated once using its reported tick duration",
              "coverage_policy": "Union of selected interval timestamps; overlaps never inflate coverage",
              "reference_clock_policy": "Reported active integral divided by QDSP Clock integral; not physical HMX/HVX capacity",
              "timestamp_rounding_tolerance_ns": BOUNDARY_TOLERANCE_NS,
              "timestamp_mapping": {
                  "scope": "Private experiment with statically verified Android SDK binary; not a universal API clock contract",
                  "sdk_binary_sha256": SDK_BINARY_SHA256,
                  "sdk_converter": "qtimer_convertTicksToNs at 0x1ce58; binary64 52.08333 literal at 0x9b08; uint64 truncation",
                  "inverse": "Recover nearest QTimer tick from raw SDK timestamp / 52.08333, then convert ticks at exact 19.2 MHz",
                  "raw_capture_preserved": True,
                  "mapped_clock": "CNTVCT/QTimer nanoseconds at 19.2 MHz",
                  "correction_ms_range": [min(p["end_ns"] - p["sdk_end_ns"] for p in records)/1e6,
                                          max(p["end_ns"] - p["sdk_end_ns"] for p in records)/1e6],
                  "adjacent_tick_delta_errors": sum(b["qtimer_end_ticks"] - a["qtimer_end_ticks"] != b["duration_ticks"]
                                                    for a, b in zip(records, records[1:]))},
              "packet_count": len(records), "windows": []}
    stdout = directory / "stdout.json"
    benchmark = json.loads(stdout.read_text()) if stdout.exists() else {}
    phases = benchmark.get("phases", [])
    if phases:
        if benchmark.get("phase_clock") != "CNTVCT_ns":
            raise ValueError("phase timestamps are not in the verified CNTVCT nanosecond clock")
        decode_windows = []
        decode_index = 0
        for phase in phases:
            name = phase["name"]
            if name == "decode_call":
                decode_windows.append((phase["start_ns"], phase["end_ns"]))
                name = f"decode_call_{decode_index}"
                decode_index += 1
            output["windows"].append({"name": name, **summarize(records, phase["start_ns"], phase["end_ns"])})
        if decode_windows:
            output["windows"].append({"name": "decode_calls_union", "decode_calls": len(decode_windows),
                                      **summarize_union(records, decode_windows)})
    elif "qtimer_start" in benchmark:
        if (benchmark.get("passed") is not True or benchmark.get("qtimer_hz") != QTIMER_HZ
                or benchmark.get("host_cntfrq_hz") != QTIMER_HZ):
            raise ValueError("calibration workload failed")
        start_ticks = unsigned_integer(benchmark["qtimer_start"], "workload QTimer start")
        end_ticks = unsigned_integer(benchmark["qtimer_end"], "workload QTimer end")
        host_start = unsigned_integer(benchmark["host_rpc_start_cntvct"], "host CNTVCT RPC start")
        host_end = unsigned_integer(benchmark["host_rpc_end_cntvct"], "host CNTVCT RPC end")
        if not host_start <= start_ticks < end_ticks <= host_end:
            raise ValueError("DSP QTimer measurement is outside its host CNTVCT RPC brackets")
        output["clock_alignment_checks"] = {"host_cntfrq_hz": QTIMER_HZ,
                                             "dsp_qtimer_within_host_cntvct_rpc": True}
        start = start_ticks * 1e9 / QTIMER_HZ
        end = end_ticks * 1e9 / QTIMER_HZ
        window = {"name": "dsp_measurement", **summarize(records, start, end)}
        regions = calibration_active_regions(benchmark)
        active_ns = intersection_duration(window["covered_intervals_ns"], regions)
        window["covered_measured_active_region_seconds"] = active_ns / 1e9
        window["covered_measured_active_region_percent"] = (
            100 * active_ns / (window["covered_seconds"] * 1e9) if window["covered_seconds"] else None)
        window["whole_workload_measured_active_region_percent"] = benchmark["measured_active_region_percent"]
        window["active_region_policy"] = "Intersection of measured kernel-region periods and selected SysMon interval union; not instruction issue occupancy"
        output["windows"].append(window)
    output["windows"].append({"name": "all_returned_intervals", **summarize(records, records[0]["start_ns"], records[-1]["end_ns"])})
    (directory / "interval-summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        result = analyze(Path(argument))
        print(result["case"])
        for window in result["windows"]:
            print(window["name"], "coverage", round(window["coverage_fraction"], 4),
                  {key: round(value["time_weighted_mean"], 4) for key, value in window["metrics"].items()})
