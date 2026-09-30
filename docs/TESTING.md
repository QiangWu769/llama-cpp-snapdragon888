# Testing

The GitHub Actions workflow runs numerical and memory-layout checks on the
Ubuntu host CPU. It requires Clang, Python 3, and Clang's UndefinedBehaviorSanitizer
runtime. It does not require a Qualcomm SDK, an Android device, a model download,
or credentials.

Run the same suites locally from the repository root:

```sh
cd htp-ops-lib
export CC=clang
export SANITIZERS=undefined
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
sh tests/v68/run_host_tests.sh
sh tests/v68/run_attention_tests.sh
sh tests/v68/run_attention_hmx_tests.sh
sh tests/v68/run_hvx_conversion_tests.sh
sh tests/v68/run_integer_dequant_tests.sh
```

The tests use Clang's `__fp16` storage type for IEEE binary16 data, with FP32
arithmetic or explicit integer conversions as appropriate. The workflow first
compiles and executes a small binary16 conversion program so an incompatible
host compiler or runtime fails with a clear diagnostic. No `_Float16` macro
substitution or fast-math flags are used.

| Suite | What it checks |
| --- | --- |
| `run_host_tests.sh` | Portable v68 matmul fallback against an independent reference, including layouts and tails. |
| `run_attention_tests.sh` | Portable attention against a double-precision reference, including GQA, masks, and tails. |
| `run_attention_hmx_tests.sh` | Production attention packing and online softmax, with a host FP16 tile-product model replacing hardware matrix instructions. |
| `run_hvx_conversion_tests.sh` | Binary16/binary32 bit conversions, rounding boundaries, and lane order using a small integer-operation model. |
| `run_integer_dequant_tests.sh` | Integer dequantization arithmetic for every INT8 value and binary16 scale bit pattern, using host models of the required integer operations. |

The host models exercise selected source-level operations. They are not Hexagon
ISA emulators. Passing these suites does not establish DSP instruction support,
FastRPC transport correctness, HMX ownership, VTCM visibility, device performance,
or whole-model accuracy.

Device validation is a separate step: cross-compile with the supported Hexagon
and Android toolchains, run operator checks on the target phone, then compare
whole-model outputs against a CPU reference. Preserve the source revision,
build options, device/firmware information, commands, and complete logs when
reporting such results. Performance measurements should identify the model,
quantization, context and generation lengths, and hardware configuration.

See the individual test notes in
[`htp-ops-lib/tests/v68`](../htp-ops-lib/tests/v68) for reference definitions,
tolerances, and coverage. CI uses read-only repository permissions and performs
no deployment or publishing.
