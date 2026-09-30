"""Export measurement values without redistributing proprietary SDK artifacts."""
import hashlib
import json
from pathlib import Path
import sys

from summarize_intervals import analyze, packets


def read(path):
    return json.loads(path.read_text())


def export(root, destination):
    evidence = {
        "schema_version": 1,
        "experiment": "Snapdragon 888 phone-local HVX/HMX activity calibration",
        "host_recorded_date": "2026-09-30",
        "device": {"model": "OnePlus 9 LE2115", "soc": "SM8350", "hexagon": "v68", "android": "14"},
        "runtime_computer": False,
        "scope": "One device and SDK version; one DSP worker/context; no physical capacity or useful-lane claim",
        "metric_ids": {"4097": "QDSP6 Load MCPS", "4182": "QDSP Clock MHz", "4352": "SDK HVX Utilization percent",
                       "4377": "SDK HVX active MCPS", "4480": "SDK HMX Utilization percent",
                       "4481": "SDK HMX active MCPS", "4521": "SDK HMX Clock MHz"},
        "clock_control": read(root / "clock-control/clock-analysis.json"),
        "timestamp_correction": read(root / "timestamp-scale-evidence.json"),
        "cases": [],
    }
    cases = [f"{mode}-d{duty}" for mode in ("scalar", "hvx", "hmx") for duty in (0, 25, 50, 100)]
    cases += ["inference-m1", "inference-m32"]
    for name in cases:
        directory = root / name
        summary = analyze(directory)
        capture = read(directory / "capture.json")
        workload = read(directory / "stdout.json")
        workload.pop("sequences", None)
        workload.pop("model", None)
        metadata = {key: value for key, value in capture.items()
                    if key.startswith("metadata_") or key in ("client_mode", "endpoint", "cleanup_failed", "callback_release_error_count")}
        evidence["cases"].append({
            "name": name, "summary": summary, "workload": workload,
            "capture_sha256": hashlib.sha256((directory / "capture.json").read_bytes()).hexdigest(),
            "sdk_provenance": metadata,
            "packets": packets(capture),
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


if __name__ == "__main__":
    result = export(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({"cases": len(result["cases"]), "output": sys.argv[2]}))
