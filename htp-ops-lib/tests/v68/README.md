# v68 matrix multiplication validation

From the repository root, with Clang installed:

```sh
sh tests/v68/run_host_tests.sh
```

The script builds the actual `src/dsp/ops/mat_mul_v68.c` using a test-only scalar
replacement for its HVX intrinsics and UndefinedBehaviorSanitizer. It runs 576
cases across F16, Q4_0, Q8_0 and IQ4_NL; M = 1/2/3/4/5/9;
K = 32/64/96/256; N = 32/64/96; normal and shifted output pointers. It checks
output guards and invalid-input rejection. This verifies layout, indexing and
control flow. It does **not** emulate Hexagon qfloat precision, prove DSP execution,
or measure hardware performance. The shim is never used by the device build.

`matmul_reference.h` independently walks the converter's tensor axes, constructs
standard GGML quant blocks before the upstream superblock repacking, and computes
ordinary matrix multiplication with double-precision accumulation. Its data
includes signed quant values and positive, negative and zero scales.

The Android `htp_v68_test` target (`src/host/v68_test.c`) uses this same test-data
generator and sends every tested operator through FastRPC to the DSP. It exercises
decode-sized M=1, a partial four-row tile at M=5, and M=8 for all quant formats,
in addition to F16 and the existing attention cases. Initialize outputs to NaN
and require `V68_RPC_TEST PASS failures=0`. The numerical comparison permits
`2e-4 + 2e-4 * abs(reference)` per output. A host-test pass alone is not a device
test pass; test the compiled v68 skeleton on the target phone separately.

## HMX FastRPC suites

Use the same Android host executable against the intended **HMX** DSP skeleton:

```sh
./htp_v68_test --hmx
./htp_v68_test --hmx --full
./htp_v68_test --hmx-attention
./htp_v68_test --pipeline
```

Run these separately. No argument retains the original HVX reference and small
matrix/attention suite. A suite selector changes the test reference, not the DSP
backend. Record the deployed skeleton/build identity and DSP HMX dispatch logs;
an opcode number or a host numerical pass alone does not establish HMX execution.

- `--hmx`: F16, Q8_0 and IQ4_NL, each with a small `(M,K,N)=(5,96,96)` case and
  `(1,896,896)`, `(5,4864,896)`, `(32,896,896)`, `(33,4864,896)` — 15 cases total.
- `--hmx --full`: the same three small cases, plus every combination of
  `M={1,5,32,33}`, `K={896,4864}`, `N=896` for all three types — 27 cases total.
- `--pipeline`: Q8_0 and IQ4_NL, each with `(128,896,896)` for the four-stage
  pipeline and `(129,1568,1056)` for output-stationary accumulation — four cases.
  The latter includes M/N/K tails and uses K chunks `512,512,512,32`.
- `--hmx-attention`: `(Q,KV,H,KH,D)=(1,65,14,2,64)`, `(5,67,14,2,64)` and
  `(33,33,2,2,64)`, each with a null mask, a causal mask with finite negative
  additive values, and a mask whose first query is entirely excluded — nine cases.

Matrix inputs vary across every row and every 32-element tile; adjacent output
tiles do not repeat. Activations and quant scales include values not exactly
representable in FP16. The independent ordinary-matrix reference explicitly
rounds FP32 activations and dequantized weights to FP16, accumulates products in
double precision, then rounds the output to FP16. Output-stationary tests also
round the running output after each 512-element K chunk, matching its documented
conversion boundaries. The ordinary M<=33 cases accumulate the full K dimension
before output conversion; their internal input-load packets do not round partial
sums. A 512 KiB VTCM region size is distinct from a 512-element K chunk.

Matrix acceptance is `abs(error) <= 0.01 + 0.001 * abs(reference)`: the relative
part is approximately one FP16 ULP, and the fixed absolute part allows small
accumulation differences under cancellation for the tested K range. This is an
engineering acceptance envelope for these fixtures, not a specification of the
undocumented accumulator or a guarantee for arbitrary values. Logs include all
failed indices (up to three examples), maximum absolute error, RMSE and maximum
error divided by its bound. Decode cases also report the expected difference
between the rounded reference and the reference without HMX conversion.

Attention uses an independent full-precision dot/softmax reference. Its tested
Q/K/V values stay in `[-1,1]`; the absolute error limit is `0.003`, including QK,
probability, PV and output conversion effects. This bound does not cover arbitrary
large Q/K magnitudes. Completely masked query rows must be exactly zero. K/V and
mask padding contain NaNs to detect accidental participation of tail storage.

Every RPC output has 128-byte guards on both sides and starts as NaN. The shared
16 MiB arena is reused only after completion. CPU matrix references use ordinary
cached allocations, then copy inputs to the uncached RPC arena; reference time
and RPC wall time are reported separately. These times are diagnostic, not a
steady-state performance benchmark. The process stops on a 20-second RPC timeout
instead of reusing potentially live DSP buffers; matrix HMX suites have a
300-second overall alarm.

The host scalar-shim suite and local syntax checks do not execute these HMX paths.
Require the chosen real-phone suite to report `V68_RPC_TEST PASS failures=0`,
inspect its numerical statistics and guards, and retain its DSP execution evidence
before claiming that suite passed on hardware.
