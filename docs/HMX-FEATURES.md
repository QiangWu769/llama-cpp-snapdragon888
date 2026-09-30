# HMX feature validation on Snapdragon 888

These are new instruction-level experiments on the same OnePlus 9 LE2115
(Snapdragon 888 / Hexagon v68, Android 14). They compare the features described
in Qualcomm's [V81 HMX Programmer's Reference Manual, 80-N2040-62 Rev. AA](https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf)
with the behavior of the available v68 instruction path. The V81 manual is a
test specification, not a guarantee that its newer conversion controls apply
to v68.

**87 independent phone processes completed: 53 reference passes, 3 explicit
compatibility mismatches, and 31 observation-only experiments.** The passes
comprise 49 arithmetic cases checking 58,368 output values with maximum
absolute error **0**, plus four exact bias-transfer checks. Every process
completed its RPC, preserved output guards, and released its HMX lock,
compute resource and power request successfully. A successful RPC by itself
is not a numerical pass.

The source is in [experiments/hmx-features](../experiments/hmx-features).
The complete [case summary](../results/hmx-features-2026-09-29/summary.json),
[compile gates](../results/hmx-features-2026-09-29/compile/summary.json) and
individual raw logs are retained in
[results/hmx-features-2026-09-29](../results/hmx-features-2026-09-29).

## Results by function

| Manual feature | Result on this v68 phone | Evidence and interpretation |
| --- | --- | --- |
| FP16 32x32 matrix multiply | **Validated** | Basic and one-tile `activation:deep` forms; signed nonsymmetric, all-ones and identity inputs. Activation controls `0x77c`, `0x7fe` and the known legacy `0x7ff` pass with legacy output control `0x7ff`. |
| Partial input channels | **Validated** | K intervals 0–7, 8–15, 16–31, 24–31 and 0–15. Activation tiles remain full; selected weights form a compact stream. |
| Activation `.deep` | **Validated** | 2, 3, 8 and 32 contiguous activation tiles: K=64, 96, 256 and **1024**. Partial first/last tiles also pass, including start greater than stop across multiple tiles. |
| Weight `.deep`, 64 output channels | **Validated** | Regular activation plus two distinguishable weight groups; both 32x32 output tiles match separately. Two ordinary `after.hf` stores expose the two banks. An additional swap between stores deliberately loses the second result. |
| Activation `.single`, shifted window | **Validated** | Spatial mask 28 (`YYYXX`), Y shifts 0/1/5/7, dY=+2048 and -2048, partial K, and nonadjacent dY=4096 with a poisoned gap. Signed, nonuniform inputs distinguish window selection from ordinary contiguous reads. |
| Weight negate, Rs bit 5 | **V81 encoding mismatch** | Two tests return the original product rather than its negative. All-ones input produces +32 instead of -32. This does not exclude another legacy mechanism for negation. |
| Repeated MAC and ACC retention | **Validated** | Two MACs give 2C; retain-store, MAC, store gives C then 2C. Repeated retain-store returns C twice. |
| ACC clear and swap | **Validated** | Ordinary conversion followed by another MAC gives C again; explicit swap exposes the other bank. Both banks are seeded and observed as C and 2C, then one `mxclracc.hf` makes both observed outputs zero. |
| Bias load/store | **Validated transfer only** | Legacy `mxmem` roundtrip is 128 B; `mxmem2` roundtrip is 256 B. Cross-format transfers preserve the first 128 B, expand the upper 128 B to zero, or truncate them as appropriate. All untouched tail words retain their sentinel. |
| Four independently selectable bias banks | **V81 encoding mismatch** | Load four different 256 B records with bank IDs in address bits 1:0; all four readbacks contain the last record. The documented selector aliases in this tested path. |
| Per-output-channel additive bias | **Validated legacy equivalent** | High 16 bits of each legacy word hold the tested FP16 bias; low 16 bits remain zero. Uniform +1 and varying channel biases pass across two signed input patterns. |
| Identity and ReLU shape | **Validated legacy equivalents** | Zero legacy bias gives the product. `after:pos.hf` gives `max(ACC,0)`; with the tested additive bias it gives **`max(ACC+b,0)`**. |
| All eight V81 shape fields | **Not portable through the tested legacy converter** | All eight `mxmem2` bias configurations fail the V81 equation when consumed by `after.hf`. The dedicated V81 conversion instruction is rejected for v68. Identity/ReLU remain available through the legacy equivalents above; the other six shapes are not established. |
| V81 input bias / scale / output bias equation | **Not validated on v68** | Separate and combined V81 records fail the specified equation through the legacy converter. In particular, a low-word FP16 scale of 1 is not an identity conversion control here. |
| CVT state with independent/repeated writeback | **API gate** | `cvt.hf = acc(Rs)` and `mxmem(Rs,Rt) = cvt` require `-mv73` or higher in Tools 19.0.07. Repeated legacy retain-stores are tested, but do not establish the independent CVT state semantics. |
| Feedback min/max to scale or output bias | **Not executed: API gate** | Its controls belong to the rejected independent CVT instruction. No controls were transplanted into the unrelated legacy output operand. |
| Extra four CVT mantissa bits / extra bias precision | **Not executed: API gate** | Reference fixtures are provided, but no usable v68 implementation of the documented modern conversion path was established. |
| Overflow control | **Default behavior observed; policy selection unverified** | Finite identity inputs up to 65504 roundtrip with zero legacy control. Products exceeding FP16 range write raw `0x7fff` or `0xffff` NaNs in the tested default state. Two distinct finite sentinels establish that these are writes, not untouched output. V81 policy bits were not used in a legacy operand; USR was not modified. |
| 37-bit accumulator / 20-bit CVT internal width | **Not established** | These are V81 architectural descriptions; the present v68 output tests do not measure internal register widths. |

## Important legacy conversion findings

The old bias word must not be initialized as an ordinary FP16 scale. With
the low word `0x3c00`, the amplitude test returns values clipped to
**[-1.9990234375, +1.9990234375]**. A separate 1,024-element reference test
verifies that observed clipping behavior. Low words 0, `0x2800` and `0x4000`
preserve the tested identity-input amplitude set through 65504. This does
not define all legacy fields or establish a general scaling formula.

`before.hf` leaves every output sentinel unchanged in these configurations,
including after a successful `after:retain.hf` store. Both finite sentinel
fills, `0x3555` and `0xb955`, remain unchanged. It must not be used as the
working output path based only on assembler acceptance.

The correct ordinary v68 sequence remains:

```c
// All pointers refer to appropriately aligned VTCM memory; power and lock
// ownership must already be established on this issuing thread.
bias = mxmem(zero_legacy_control);
activation.hf = mxmem(activation, activation_control):deep;
weight.hf = mxmem(weight, weight_range);
mxmem(output, 2047):after.hf = acc;
```

This is illustrative assembly syntax, not standalone C. The production
backend also clears the accumulator explicitly. Its helper cannot be used
unchanged to study retention because that additional clear changes the state.

## Method and scope

Build: Hexagon SDK 6.6.0.0, Tools 19.0.07, `-mv68 -mhmx`, Android NDK r26d.
The skeleton ELF header identifies **V68**, and its disassembly includes
the tested HMX operations. The local and deployed skeleton SHA256 agree:

```text
50451e24353d72d4bb8464da1667f90a876c052fbc86d0117600eb65a06f4605
```

Each case uses a fresh Android process and unsigned cDSP RPC, requests
256 KiB from the observed 4 MiB VTCM, and acquires the legacy exclusive HMX
lock. Inputs are deterministic binary fractions. The cached ARM host
calculates a scalar reference over logical matrices and rounds at the FP16
output boundary; it does not implement the result on behalf of the DSP.
Every arithmetic output is checked, including both tiles in the 64-channel
case. NaN or finite output sentinels, exact output counts, raw half samples,
tail guards, cache maintenance, barriers and bounded process/RPC deadlines
make missing writes visible.

Six matrix instructions and fourteen legacy epilogue/bias instructions
compile independently for v68. The two modern CVT instructions reject v68
and compile for both v73 and v81. Compilation for a newer architecture is
a control check, not a phone runtime test.

`passed: null` marks observation-only cases in the JSON. Their `reference_match`
field compares with the stated candidate equation and must not be interpreted
as a general feature verdict. For nonfinite mismatches, `bad` and raw half
bits carry the result; `max_abs_error` alone is not meaningful.

The experiment changes no production kernel and measures no performance,
power, CPU/GPU/NPU utilization, or thermal behavior. Results apply to the
listed firmware and tested operands. They do not establish all v68 ISA
capabilities or promise bit-exact agreement for arbitrary inputs.
