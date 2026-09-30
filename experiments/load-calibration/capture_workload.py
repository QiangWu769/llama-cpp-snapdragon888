"""Private phone-local research runner built on the qcphoneperf Python library."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from qcphoneperf import Profiler


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def thermal():
    result = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            kind = (zone / "type").read_text().strip()
            if any(part in kind.lower() for part in ("cpu", "gpu", "npu", "soc", "skin", "battery", "quiet")):
                result.append({"zone": zone.name, "type": kind,
                               "raw_temperature": (zone / "temp").read_text().strip()})
        except OSError:
            pass
    return result


def collect(directory, config, clock_library):
    directory.mkdir(parents=True, exist_ok=False)
    library = ctypes.CDLL(str(clock_library))
    library.calibration_read_clocks.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint]
    library.calibration_read_clocks.restype = ctypes.c_int
    anchors, errors = [], []
    stop = threading.Event()

    def clock():
        data = (ctypes.c_uint64 * 7)()
        if library.calibration_read_clocks(data, 7):
            raise RuntimeError("clock anchor failed")
        names = ["monotonic_before_ns", "boottime_ns", "monotonic_raw_ns", "realtime_ns",
                 "cntvct_ticks", "cntfrq_hz", "monotonic_after_ns"]
        value = dict(zip(names, map(int, data)))
        if value["cntfrq_hz"] != 19200000:
            raise RuntimeError("unexpected QTimer frequency")
        value["cntvct_ns_integer"] = value["cntvct_ticks"] * 1000000000 // value["cntfrq_hz"]
        return value

    def sample_clocks():
        try:
            while not stop.is_set():
                anchors.append(clock())
                stop.wait(0.05)
        except Exception as error:
            errors.append({"source": "clock", "error": repr(error)})

    def run_workload():
        time.sleep(config.get("start_delay_seconds", 1.5))
        state = {"thermal_before": thermal(), "started": clock(), "command": config["command"]}
        save(directory / "workload.json", state)
        try:
            environment = os.environ.copy()
            environment.update(config.get("environment", {}))
            with (directory / "stdout.json").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
                process = subprocess.Popen(config["command"], env=environment,
                                           stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                state["pid"] = process.pid
                try:
                    state["returncode"] = process.wait(timeout=config.get("timeout_seconds", 30))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    state["timed_out"] = True
                    raise
        except Exception as error:
            state["error"] = repr(error)
            errors.append({"source": "workload", "error": repr(error)})
        finally:
            state["finished"] = clock()
            state["thermal_after"] = thermal()
            save(directory / "workload.json", state)

    save(directory / "process.json", {"pid": os.getpid(), "uid": os.getuid(), "platform": sys.platform,
         "runtime_computer": False, "endpoint": None, "config": config,
         "clock_library_sha256": hashlib.sha256(clock_library.read_bytes()).hexdigest(),
         "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "executable_sha256": hashlib.sha256(Path(config["command"][0]).read_bytes()).hexdigest()})
    clock_thread = threading.Thread(target=sample_clocks)
    workload = threading.Thread(target=run_workload)
    capture = None
    with Profiler("/data/local/tmp/qualcomm-profiler",
                  library=config.get("adapter", "/data/local/tmp/qcphoneperf-api/lib/libqcphoneperf.so")) as phone:
        clock_thread.start()
        workload.start()
        try:
            capture = phone.snapshot_metrics("npu", metric_ids=config.get("metric_ids", [4097, 4182, 4352, 4377, 4480, 4481, 4521]),
                                             sample_ms=config.get("capture_ms", 14000))
            save(directory / "capture.json", capture)
        except Exception as error:
            errors.append({"source": "capture", "error": repr(error)})
        finally:
            workload.join()
            stop.set()
            clock_thread.join()
            save(directory / "anchors.json", {"anchors": anchors, "errors": errors})
    save(directory / "close.json", {"state": "closed", "clock": clock()})
    state = json.loads((directory / "workload.json").read_text())
    result = {"directory": str(directory), "errors": errors, "returncode": state.get("returncode"),
              "npu_status": capture["domains"]["npu"]["status"] if capture else None,
              "valid_samples": capture["domains"]["npu"]["valid_sample_count"] if capture else 0}
    save(directory / "summary.json", result)
    print(json.dumps(result))
    if errors or state.get("returncode") != 0:
        raise RuntimeError("workload or capture failed; evidence preserved")


if __name__ == "__main__":
    collect(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]))
