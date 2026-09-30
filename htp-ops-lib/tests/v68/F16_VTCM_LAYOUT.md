# Experimental FP16 VTCM layout

The CMake option `HTP_F16_M_AWARE_VTCM` defaults to `OFF`. Build the same source
with `HTP_USE_HMX=ON`, `HEXAGON_ARCH=v68` (the SDK's `DSP_VERSION=v68`) and
`HTP_F16_M_AWARE_VTCM=OFF` or `ON` into separate build directories for A/B tests.
The option affects only FP16 HMX matmul; quantized matmul, cache maintenance,
DMA waits, HMX locking and K accumulation order remain unchanged.

From the repository root, with the SDK and NDK environment variables set:

```sh
python3 tools/build.py --backend hmx --m-aware-vtcm
```

Omit `--m-aware-vtcm` for the baseline; the script explicitly configures `OFF`
so an earlier optimized CMake cache cannot accidentally retain the option.
The flag is rejected for HVX and CPU backends. Use `--dry-run` to inspect the
commands without building.

The enabled planner uses the manager's actual usable VTCM, after reservations.
It keeps the established maximum row chunk, searches whole 32-row chunks, and
minimizes the number of M/N chunks; ties favor larger M chunks. Weight and
output capacities grow together with N. Allocation includes complete physical
32-row activation/output tiles even for M=1, and 256-byte scales. Matrix areas
and base addresses have 2048-byte alignment. Allocation bounds and pointer
overflow are checked before sequential allocation. Unsupported shapes or
insufficient VTCM fail visibly rather than overrunning an allocation.

Run the standalone tests without a Qualcomm SDK:

```sh
sh tests/v68/run_f16_vtcm_layout_tests.sh
```

The tests compare selected geometry against an independent exhaustive feasible
pair search. They cover M=1/odd/full/tail rows, Qwen projection shapes, small
VTCM and exact minimum capacity, invalid/overflow inputs, alignment, reserved
memory guards, full physical output tile writes and complete logical output
coverage. AddressSanitizer and UndefinedBehaviorSanitizer are enabled.
If the host cannot initialize AddressSanitizer, use
`SANITIZERS=undefined sh tests/v68/run_f16_vtcm_layout_tests.sh`, and run the
default configuration separately on a host with a working ASan runtime.
These tests validate layout arithmetic and memory geometry, not Hexagon ISA,
hardware accumulator behavior or numerical inference.

Before interpreting performance, run both device builds through the existing
`htp_v68_test --hmx --full`, `--pipeline` and `--hmx-attention` suites. Compare
FP16 outputs for M=1,31,32,33,64,65,128,129; K/N=896/896,896/4864,4864/896;
include partial final N chunks. The existing half-input/half-output reference
and an identical model/log-probability fixture should pass for both builds.
Also compare baseline and optimized outputs directly: only M/N tiling changes,
so valid output values are expected to match. Preserve quantized/attention
regression results and verify the binary build flags and hashes.

The additional `htp_v68_test --vtcm-layout` suite uses the existing independent
half-input/half-output reference and fixed tolerance, with deterministic seeds.
It runs 24 cases from M={1,31,32,33,64,65,128,129} and Qwen K/N={896/896,
896/4864,4864/896}, four partial-N cases at M={1,33}, K/N={896/5216,4864/928},
then four rejection tests (M=0, K/N not divisible by 32, misaligned activation).
Rejection tests require status -1 and an unchanged output including guards.
Each valid case prints its seed and an FNV-1a digest of canonical float output
bits for direct baseline/optimized comparison. This digest detects differences;
the independent numerical reference determines correctness. The suite has a
600-second process deadline and the existing 20-second individual DSP deadline.
