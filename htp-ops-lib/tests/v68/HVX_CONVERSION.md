# v68 FP16 / FP32 conversion tests

Run `./tests/v68/run_hvx_conversion_tests.sh` with a host Clang supporting
`__fp16`. UBSan is enabled by default; use `SANITIZERS=address,undefined` for
both sanitizers. `CONVERSION_HEADER=/absolute/path/header.h` can select a
candidate header without replacing the production file.

The first test isolates the actual scalar integer reference helpers from
`include/dsp/hvx_convert.h` and compares them with native IEEE casts: all
65,536 binary16 patterns, rounding-midpoint neighbors, and one million random
binary32 bit patterns. Finite results must match bit for bit; NaNs must remain
NaNs with their sign.

The second test includes the actual conversion wrappers with a small integer
HVX instruction model. It checks all binary16 patterns, the alternating
low/high lane contract, and 2,097,152 random binary32 patterns. This catches
packing and integer-vector formula errors. Neither host test validates the
actual Hexagon instructions or their performance; DSP tests are still needed.

The v68 implementation avoids the intermediate qf16 representation, whose
smaller significand loses precision during an apparently neutral addition of
zero. FP16 to FP32 uses integer HVX for every bit pattern. FP32 to FP16 uses
integer HVX with nearest, ties-to-even rounding for normal finite values and
scalar integer repair for underflow, overflow, and nonfinite lanes. The
original v73 conversion path is preserved.
