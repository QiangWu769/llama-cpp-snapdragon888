# Snapdragon 888 HMX feature evidence

Read [HMX-FEATURES.md](../../docs/HMX-FEATURES.md) for interpretation and
[the probe README](../../experiments/hmx-features/README.md) for reproduction.
This snapshot contains 87 real-device cases: 53 reference passes, 3 explicit
compatibility mismatches and 31 observations. The passes include four bias
transfer checks and 49 arithmetic cases checking 58,368 values with zero
maximum absolute error.

- [summary.json](summary.json): every case's exact numeric probe arguments,
  process status, comparison counts and raw half/sentinel observations.
- `phone-CASE.log`: full raw log for each named case, including guards and
  resource cleanup. Every RPC completed with stage 130, result 0, and no
  output guard changes.
- [metadata.json](metadata.json): device/toolchain configuration, totals and
  runtime source hashes.
- [binary.json](binary.json): identical local and deployed DSP skeleton SHA256.
- [elf-header.txt](elf-header.txt), [disassembly.txt](disassembly.txt) and
  [compiler-version.txt](compiler-version.txt): V68 architecture, 61 emitted
  HMX instruction lines and Tools 19.0.07.
- [compile/summary.json](compile/summary.json): 26 separate compile checks;
  the two modern instructions reject v68 and accept v73/v81. Diagnostic logs
  for every combination are included.

`passed=null` is observation-only. A candidate equation mismatch in these
entries is not a universal hardware-support conclusion. NaN outputs can
report `max_abs_error=0` because no finite difference updates that statistic;
the nonfinite/bad counts and raw bits show the actual mismatch.

Absolute build-host/toolchain paths are replaced with placeholders. Private
credentials, network/device identifiers, compiled objects and licensed SDK
files are excluded. This is a functional snapshot with no performance,
utilization, power or thermal measurements.
