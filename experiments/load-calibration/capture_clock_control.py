"""Phone-local library experiment: preserve raw SDK samples and clock anchors."""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import sys
import threading
import time

from qcphoneperf import Profiler


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def collect_clock_control(directory: Path, clock_library: Path, duration_ms=6000):
    directory.mkdir(parents=True, exist_ok=False)
    library = ctypes.CDLL(str(clock_library))
    library.calibration_read_clocks.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint]
    library.calibration_read_clocks.restype = ctypes.c_int
    anchors, errors = [], []
    stop = threading.Event()

    def anchor():
        data = (ctypes.c_uint64 * 7)()
        if library.calibration_read_clocks(data, 7):
            raise RuntimeError("clock anchor failed")
        names = ["monotonic_before_ns", "boottime_ns", "monotonic_raw_ns", "realtime_ns",
                 "cntvct_ticks", "cntfrq_hz", "monotonic_after_ns"]
        value = dict(zip(names, map(int, data)))
        value["cntvct_ns_integer"] = value["cntvct_ticks"] * 1000000000 // value["cntfrq_hz"]
        anchors.append(value)

    def sample_clocks():
        try:
            while not stop.is_set():
                anchor()
                stop.wait(0.05)
        except Exception as error:
            errors.append(repr(error))

    metadata = {"pid": os.getpid(), "uid": os.getuid(), "platform": sys.platform,
                "mode": "idle-clock-control", "duration_ms": duration_ms,
                "clock_library_sha256": hashlib.sha256(clock_library.read_bytes()).hexdigest()}
    save(directory / "process.json", metadata)
    worker = threading.Thread(target=sample_clocks)
    with Profiler("/data/local/tmp/qualcomm-profiler",
                  library="/data/local/tmp/qcphoneperf-api/lib/libqcphoneperf.so") as phone:
        worker.start()
        try:
            result = phone.snapshot_metrics("npu", metric_ids=[4097, 4182, 4352, 4377, 4480, 4481, 4521],
                                            sample_ms=duration_ms)
            save(directory / "capture.json", result)
        finally:
            stop.set()
            worker.join()
            save(directory / "anchors.json", {"anchors": anchors, "errors": errors})
        closed = phone.close()
        save(directory / "close.json", closed)
    if errors:
        raise RuntimeError(errors)
    print(json.dumps({"anchors": len(anchors), "status": result["domains"]["npu"]["status"],
                      "cntfrq_hz": sorted({v["cntfrq_hz"] for v in anchors}), "closed": closed}))


if __name__ == "__main__":
    collect_clock_control(Path(sys.argv[1]), Path(sys.argv[2]))
