# Portable v68 attention tests

Run from any directory with a host Clang that supports the `__fp16` storage
type (tested with Apple Clang on arm64):

```sh
sh tests/v68/run_attention_tests.sh
```

The script builds `flash_attn_v68.c` and `attention_portable_test.c` with
`-std=c11 -O2 -Wall -Wextra -Werror -fsanitize=undefined
-fno-sanitize-recover=all`, executes the test, and removes its temporary build
directory. Override the compiler with `CC=/path/to/clang`. To request both
AddressSanitizer and UndefinedBehaviorSanitizer on a supported host:

```sh
SANITIZERS=address,undefined sh tests/v68/run_attention_tests.sh
```

The implementation's non-Hexagon branch is portable; no HVX/worker-pool stub
is needed. The test includes the actual `dsp/ops.h` to check the exported ABI.
It runs 618 deterministic comparisons against independently computed double
precision attention, covering MHA/GQA, varying query counts and head dimensions,
KV lengths across 64-element mask-padding boundaries, absent/causal/fully masked
and finite additive masks, and large logits. Additional cases cover FP16
subnormals, NaNs in excluded K/V slots, positive-infinite mask entries, and
invalid inputs. Expected maximum absolute error on the tested host is below
`2e-4` (observed `1.30e-5`).

These are numerical/layout tests of the scalar portable path. They do **not**
test Hexagon instructions, worker-pool concurrency, FastRPC, cache coherency,
or DSP execution. Run the v68 target tests on hardware for those properties.
AddressSanitizer failed during runtime initialization in the original sandbox;
the reported successful local run used UBSan only.

## v68 HMX attention path

`./tests/v68/run_attention_hmx_tests.sh` builds the `HTP_HMX_V68` branch in
`flash_attn.c` with a host implementation of a 32x32 FP16 HMX tile product.
The production path uses real HMX for both QK and PV. Both builds run the same
packing, online FP32 softmax, additive mask, GQA, tail and output logic.

The test compares against an independent double-precision attention reference.
It covers 618 shapes/masks, including head dimension 64 with GQA factor 7,
queries spanning a 32-row tile, KV tails, absent/causal/additive/all-masked
masks, and positive-infinite masks. The normal-input absolute-error bound is
0.002. The input-amplitude-8 stress cases use 0.025 because QK and PV are
stored as FP16; they deliberately amplify rounding near competing logits.
The observed normal-input maximum error was 0.0005962; the amplitude-8
maximum was 0.0213391. The FP16 tile path permits one binary16 subnormal
step of output rounding.

The script runs UBSan by default; use `SANITIZERS=address,undefined` to include
ASan. This test verifies layout and algorithm code, not the Hexagon ISA,
legacy conversion configuration, cache behavior, locking or performance;
those still require execution on the phone. The v68 branch currently accepts
head dimensions from 1 through 512, while the non-HMX fallback retains its
separate support range.
