# Phone-local activity calibration experiments

These are research callers and analysis functions. The collector remains the
phone-local `qcphoneperf` Python/C++ library; no collector CLI or APK is installed.
The [report](../../docs/LOAD-CALIBRATION.md) explains measured activity, the SDK's
percentage denominator, and the version-specific timestamp correction.

- [probe/](probe/README.md): sustained scalar, one-context HVX and resident FP16
  HMX controls, each contained in one bounded DSP RPC.
- `phone_clocks.c`: read-only clock helper; reads CLOCK_MONOTONIC, BOOTTIME,
  MONOTONIC_RAW, REALTIME, CNTVCT and CNTFRQ. No PMU programming.
- `capture_clock_control.py`: phone-local idle capture with repeated clock
  anchors, using `qcphoneperf.Profiler`.
- `capture_workload.py`: phone-local library caller that captures a bounded
  workload, preserves raw SDK data and clock anchors, and records outcomes.
- `inference/`: the existing independent-sequence inference caller with CNTVCT
  phase markers. It links matching existing llama libraries.
- `summarize_intervals.py`: importable `packets`, `summarize_union` and `analyze`
  functions. The decoder coefficient is specific to the documented SDK hash.
- `export_evidence.py` and `make_figures.py`: export measured values without SDK
  binaries and render the report figures. Matplotlib is needed only for plots.

Build `phone_clocks.c` as an Android ARM64 shared library with the existing NDK,
then pass its phone path to the capture function. Calls to
`collect(directory, config, clock_library)` take a fresh output directory and a
Python dictionary. Relevant configuration fields are:

```python
config = {
    "command": ["/data/local/tmp/calibration/hmx_load_calibration_test",
                "--mode", "hmx", "--seconds", "8", "--period-us", "250000",
                "--duty", "25", "--workers", "1"],
    "capture_ms": 13000,
    "timeout_seconds": 20,
    "environment": {
        "LD_LIBRARY_PATH": "/data/local/tmp/calibration:/vendor/lib64",
        "ADSP_LIBRARY_PATH": "/data/local/tmp/calibration/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp;/system/lib/rfsa/adsp",
    },
}
```

The capture examples default to the documented research phone's separately
installed SDK, Python bindings and adapter paths. Supply `config["adapter"]` to
select an alternate native build. Set the process loader environment before
starting phone Python, as described by the library API. Root permissions and
the authorized Qualcomm SDK are external prerequisites for this tested device.

Stage each payload into an isolated directory. Run workload and profiler cases
serially; do not mix multiple collectors or reprogram PMU counters concurrently.
The development computer is not involved once `collect()` runs on the phone.

For the verified experiment, `params[6]` is the actual interval in 19.2 MHz ticks.
Recover hardware time from the SDK's truncated 52.08333 ns/tick conversion only
after checking the exact SDK library hash. The analyzer verifies recovered tick
deltas, complete metric groups, interval continuity and workload checks. It
keeps raw source timestamps, rejects crossing phase intervals, integrates zeros,
and reports coverage separately from the reported-duration sum.

Do not interpret an SDK percent or the control-normalized activity as physical
MAC capacity, lane occupancy or useful row count. A different SDK, target,
clock behavior or context configuration needs independent verification.

The compact [evidence JSON](../../results/LOAD-CALIBRATION-20260930.json) contains
per-packet values, raw and corrected timing, per-period workload records, phase
markers, checksums and interval summaries. Original SDK capture files are kept
outside Git; their hashes are included. The public package contains no Qualcomm
SDK headers, generated RPC files, binaries, database or disassembly.
