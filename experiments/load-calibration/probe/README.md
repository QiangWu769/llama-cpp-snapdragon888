# Sustained Snapdragon 888 DSP controls

This independent calibration workload selects scalar DSP, HVX or HMX
arithmetic and executes a bounded measurement window inside **one compute
RPC**. Setup, packing, one sanity/warmup operation, and final verification
occur outside that window. Existing inference binaries are not modified.
The new payload/interface names are `hmx_load_calibration_test`,
`libhmx_load_calibration.so` and `libhmx_load_calibration_skel.so`.

| Mode | Resident arithmetic per iteration | Output verification |
| --- | --- | --- |
| `scalar` | Four independent unsigned scalar additions. | Exact modular values from completed iteration count. |
| `hvx` | Four independent 128-byte vector additions. | All 128 word lanes checked against a modular oracle. |
| `hmx` | F16 M32/K1024/N32 matrix multiplication. | All 1,024 finite outputs checked against exact signed binary fractions. |

The HMX operands and result reside in one **256 KiB VTCM allocation**. A and
B occupy 64 KiB each, C 2 KiB, and the legacy zero-bias region 256 bytes.
Allocation is checked against the runtime VTCM query and alignment. All 32
matrix rows contain real input values. The v68 sequence clears the accumulator,
loads resident activation/weight tiles, and uses the legacy `:after.hf` store.
An initial NaN sentinel catches missing writes.

Scalar/HMX control code disables HVX and compiler vectorization; HVX arithmetic
is in a separate 128-byte translation unit. Volatile register assembly retains
the arithmetic loops. Disassembly was inspected privately: hardware loops
contain scalar adds, HVX adds, or the intended HMX sequence. No HVX memory
access occurs inside its arithmetic loop, only at chunk boundaries. No SDK
headers, generated sources, binaries or proprietary disassembly are bundled.

Only **one DSP thread and one HVX context** are supported. Other `--workers`
values fail visibly. HVX and HMX ownership belong to their separately selected
modes. The probe does not program PMU registers or reset shared counters.
It does not establish saturation of all physical HVX units, useful-lane
occupancy, HMX row gating or the meaning of an undocumented raw event.

## Build

The tested toolchain is Hexagon SDK 6.6.0.0 / Tools 19.0.07, Android NDK r26d,
v68 DSP and arm64-v8a Android API 26. Supply externally installed dependencies:

```sh
python3 experiments/load-calibration/probe/build.py \
  --sdk "$HEXAGON_SDK_ROOT" --ndk "$ANDROID_NDK" --qaic-bin "$QAIC_BIN"
```

`--qaic-bin` is optional if QAIC is already available on `PATH`. The script
builds locally on its invocation host and makes no device calls. Outputs are:

```text
hexagon_ReleaseG_toolv19_v68/ship/libhmx_load_calibration_skel.so
android_ReleaseG_aarch64/ship/libhmx_load_calibration.so
android_ReleaseG_aarch64/ship/hmx_load_calibration_test
```

Stage them in a new isolated Android payload directory using the matching
vendor FastRPC runtime. Supply `LD_LIBRARY_PATH` for the host library and
`ADSP_LIBRARY_PATH`/`DSP_LIBRARY_PATH` for the skeleton plus the device's normal
DSP paths. Preserve stdout JSON, stderr and process exit status separately.

## Bounded duty-cycle controls

```sh
./hmx_load_calibration_test --mode scalar --seconds 8 --period-us 250000 --duty 100
./hmx_load_calibration_test --mode hvx --seconds 8 --period-us 250000 --duty 50
./hmx_load_calibration_test --mode hmx --seconds 8 --period-us 250000 --duty 25
```

Accepted duty values are **0/25/50/100%**. The DSP runs whole chunks until
each absolute active deadline, then sleeps for the remaining period. Inactive
time is not a scalar busy-wait. Qtimer deadlines bound the window to 1–60
seconds. The period must divide the duration exactly, be 250,000–5,000,000 us,
and produce at most 128 periods. Host and DSP validate these conditions.

Default chunks contain 16,384 scalar iterations, 4,096 HVX iterations or 32
HMX products. `--chunk 1..65536` permits a separate overhead check. Whole
chunks can overrun a requested active deadline; actual active and elapsed
timing is returned per period. Zero duty keeps the mode's resource reservation
and still executes its single sanity operation **before** the measured window.

## Timing interpretation

The report contains DSP start/end Qtimer and processor cycles, per-period
active/elapsed values, completed iterations, exact-result checksums and resource
return codes. Host `CLOCK_MONOTONIC` nanoseconds and ARM `CNTVCT`/`CNTFRQ`
bracket the complete RPC. These outer edges include setup/cleanup; use the
inner DSP window and complete interior counter intervals for calibration.

The installed SDK documents Qtimer as a relative 56-bit **19.2 MHz** reference
that continues across DSP power states. The 64-bit processor-cycle counter
stops in low-power states such as clock gating. Neither is an HMX/HVX busy
counter. User counter reads can return zero when RTOS permission is absent;
the report exposes processor-counter availability and rejects an unavailable
Qtimer.

`measured_active_region_percent` measures elapsed time in the active-loop
regions, including scalar control, timer checks and instruction issue overhead.
It is **not physical engine busy percentage**. Requested duty, event rate and
the SDK's derived percentage are distinct quantities. Keep scalar and
zero-duty reservation controls, verify measured intervals, and avoid assuming
that a clamped SDK 100% means all physical units or useful lanes are occupied.

Official SDK itrace metadata distinguishes VFIFO activity, HVX instruction
packets and committed packets on XE-enabled threads. Its `0x80xx` event IDs
are explicitly not raw PMU selectors. A raw decoder field requires a verified
mapping before assigning one of those meanings. A V81 HMX instruction manual
does not establish V68 counter semantics.

## Portable oracle regression

```sh
sh experiments/load-calibration/probe/run-oracle.sh
```

The test runs 1,920 modular-counter and signed-matrix cases with UBSan,
including 32-bit iteration wraparound and negative HMX columns. The host
reference explicitly converts the row factor to float before multiplying a
signed column factor; unsigned promotion must not turn negative columns into
large positive expected values. The report occupies 10,928 bytes in a 16 KiB
RPCMEM allocation. Host tests do not replace physical DSP execution and
numerical checks.
