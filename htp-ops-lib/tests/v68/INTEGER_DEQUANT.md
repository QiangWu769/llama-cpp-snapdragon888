# v68 integer HVX dequantization

From the repository root, with Clang and Python 3 installed:

```sh
sh tests/v68/run_integer_dequant_tests.sh
```

The runner extracts the actual integer dequantization helper from `mat_mul.c`
and runs it with scalar definitions of its integer HVX instructions under
UndefinedBehaviorSanitizer. It checks all 256 INT8 values against all 65,536
binary16 scale bit patterns: **16,777,216 products**. Adjacent vector lanes use
different scales to expose even/odd lane errors. The independent reference uses
native IEEE float multiplication and half conversion; finite values, signed zero
and infinities must match exactly, while NaNs must agree in classification.

The production helper reconstructs the integer quant value from its half bits,
multiplies its magnitude by the scale's integer significand, and constructs the
exact binary32 product bits. `hvx_my_wsf_to_vhf` then rounds once to binary16.
No qfloat operation participates in this multiplication, avoiding qfloat's odd
low-bit rounding at products halfway between two half values. Run
`sh tests/v68/run_hvx_conversion_tests.sh` to check the separate conversion code.

This host test checks source arithmetic and lane semantics; it is not a Hexagon
simulator and does not establish hardware execution or performance. Build for
v68 and run the FastRPC matrix suites separately, preserving the tested binary
identity and logs. `-DHTP_V68_DEQUANT_SCALAR_REFERENCE=1` selects the retained
scalar IEEE dequantization diagnostic; it is disabled by default. That switch
affects only quant-value times scale, retaining the normal unpacking and HMX
matrix multiplication path.
