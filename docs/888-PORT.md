# Snapdragon 888 / Hexagon v68 Porting Notes

This document explains the changes relative to the pinned haozixu upstream revisions and the problems they address. See [UPSTREAM.json](../UPSTREAM.json) for provenance, the [repository README](../README.md) for results and performance limits, and [BUILD.md](BUILD.md) for build instructions.

The port retains the upstream llama.cpp HTP host interface, FastRPC channel, GGUF conversion, and main matrix tiling structure. It adds a v68 HMX path while preserving a separate HVX reference implementation. Operator tests and full Qwen2.5-0.5B inference have run on the Snapdragon 888's actual cDSP. HMX is the matrix unit; HVX is the vector unit. They are distinct execution paths.

## 1. Hardware and firmware adaptation

### 1.1 HMX output: replace newer conversion instructions with legacy `after.hf`

Upstream [hmx_utils.h](../htp-ops-lib/include/dsp/hmx_utils.h) uses an accumulator conversion and writeback sequence for newer architectures. The v68 branch uses legacy writeback and explicitly clears the accumulator:

```text
Original path:
  cvt.hf = acc(2)
  mxmem(out, 0) = cvt

v68 path:
  mxmem(out, 2047):after.hf = acc
  mxclracc.hf
```

The combination of `2047` and legacy writeback was validated on the phone. Tests with multiple output tiles, positive and negative values, long K dimensions, and pipelines check layout, output range, and residual accumulator state across tiles. These tests do not establish the complete ISA semantics of every bit in that control parameter. Conversion fields from newer HMX manuals cannot be assumed to apply directly to v68.

Inputs retain the upstream 32×32 FP16 tile layout. Within one tile, logical `(row, col)` maps to halfword index `(row / 2) * 64 + col * 2 + row % 2`. The port does not introduce a different model file layout for the 888.

### 1.2 Zero initialization for `bias = mxmem` is not a numerical scale

Upstream uses `bias = mxmem2(...)` to load newer control/scale data. The v68 branch uses `bias = mxmem(...)` and clears the supplied legacy control region. The helper retains its original name, `hmx_init_column_scales`, but the v68 branch explicitly ignores its `v_scale` argument.

**The low 16 bits of the legacy bias region must not be interpreted as an ordinary FP16 multiplier.** On the tested v68 device, retaining the upstream `0x3c00` initialization changes output behavior; zero initialization passed the matrix numerical tests. This documents a validated initialization protocol, not a complete decoding of the legacy fields.

Attention therefore applies `1/sqrt(D)` explicitly in FP32 rather than using the bias region for numerical scaling. Sources: [hmx_utils.h](../htp-ops-lib/include/dsp/hmx_utils.h), [flash_attn.c](../htp-ops-lib/src/dsp/ops/flash_attn.c).

### 1.3 Use the firmware's legacy exclusive HAP lock

The original path uses `HAP_compute_res_hmx_lock2(..., HAP_COMPUTE_RES_HMX_SHARED)`. The tested firmware uses the older `HAP_compute_res_hmx_lock` / `HAP_compute_res_hmx_unlock` API. The v68 path therefore acquires and releases exclusive ownership on the worker that actually issues matrix instructions, rather than only initializing ownership on the thread that creates the worker pool.

A lock failure immediately terminates DSP execution. Compiler and DSP memory barriers surround HMX ownership transitions. Resource acquisition and worker-pool initialization failures propagate to the caller and release resources already acquired. Sources: [hmx_mgr.c](../htp-ops-lib/src/dsp/hmx_mgr.c), [worker_pool.c](../htp-ops-lib/src/dsp/worker_pool.c), [power.c](../htp-ops-lib/src/dsp/power.c).

### 1.4 Fit the measured 4 MiB VTCM capacity

The upstream matrix paths organize activations, weights, outputs, and double-buffered scratch space into multiple 1 MiB regions. The tested 888 reports 4 MiB of VTCM, which cannot hold some original layouts. The v68 path reduces each basic workspace region to **512 KiB**. The largest corresponding layout uses six regions plus an identity tile and control region, approximately 3 MiB. Before execution, bounds are checked against the actual available space.

This changes M/N chunk sizes; it does not split every matrix product's K dimension into chunks of 512. The ordinary path still accumulates the full K dimension before output conversion. The quantized output-stationary branch separately retains its existing chunks of **512 K elements**. These two uses of “512” refer to different quantities.

The port also adds DMA descriptor initialization and alignment, visibility barriers after DMA completion, barriers at worker handoffs, dimension/byte-count checks, and safe pointer handling for odd rows while retaining their existing zero padding. Sources: [mat_mul.c](../htp-ops-lib/src/dsp/ops/mat_mul.c), [vtcm_mgr.cc](../htp-ops-lib/src/dsp/vtcm_mgr.cc). Workspace sizing is a hardware adaptation; bounds and synchronization checks also improve general robustness.

### 1.5 Build selection and firmware runtimes

[CMakeLists.txt](../htp-ops-lib/CMakeLists.txt) adds `HTP_USE_HMX`, which defaults to OFF for v68. Setting it to ON defines `HTP_HMX_V68`, enables HMX compiler options, and selects the v68 HMX branches. OFF selects the new [mat_mul_v68.c](../htp-ops-lib/src/dsp/ops/mat_mul_v68.c) and [flash_attn_v68.c](../htp-ops-lib/src/dsp/ops/flash_attn_v68.c) as a separate HVX reference implementation. Non-v68 builds retain their original architecture branches, but other devices were not revalidated during this port.

The tested firmware's DSP C++ runtime also affects loading: the DSP `libc++.so.1` / `libc++abi.so.1` shipped with SDK 19 reference `aligned_alloc`, which the device does not provide. Deployment uses the device firmware's DSP runtime. Android ARM64 `libc++_shared.so` is a separate runtime and must still be deployed with the host-side executables. See the [build and deployment instructions](BUILD.md). This repository does not distribute these SDK or firmware binaries.

## 2. Numerical correctness adaptation

### 2.1 Convert FP16 / FP32 without lossy qfloat intermediates

The original HVX conversion chain includes paths through qf16/qf32 intermediate formats. Even adding zero to convert FP16 into qf16 can lose significant bits from the original IEEE binary16 value; converting back to FP32 cannot restore them. Small integer-valued matrix tests alone do not detect this class of error.

The v68 [hvx_convert.h](../htp-ops-lib/include/dsp/hvx_convert.h) reconstructs IEEE bit patterns with integer HVX operations:

- FP16 → FP32 preserves signs, normal values, subnormals, and infinities. NaNs are quieted while retaining their corresponding payload.
- FP32 → FP16 uses a vector integer fast path for ordinary finite values with round-to-nearest, ties-to-even. Integer scalar code handles underflow, overflow, and nonfinite lanes.
- Packing and unpacking explicitly handle even/odd lane order, avoiding the assumption that low/high vectors always represent two consecutive halves.

Independent host models check all 65,536 FP16 bit patterns, FP32 rounding boundaries, and random values. See [HVX_CONVERSION.md](../htp-ops-lib/tests/v68/HVX_CONVERSION.md). These host tests verify source-level arithmetic and lane organization; separate DSP tests validate actual instruction execution.

### 2.2 Quantized value × scale: exact integer product, then one rounding

The quantized path retains upstream Q8_0 / IQ4_NL layouts and codebooks. It dequantizes values to FP16 before using FP16 HMX. The main v68 changes are in [mat_mul.c](../htp-ops-lib/src/dsp/ops/mat_mul.c):

1. Construct exact FP16 bit patterns for signed INT8 values.
2. Use integer widening multiplication for the quantized integer and FP16 scale significand, reconstructing the exact FP32 product bits.
3. Round to FP16 using the IEEE nearest/ties-to-even conversion described above.
4. Explicitly pair the widening multiply's even/odd lanes with their scales, fixing mismatches exposed by adjacent values with different scales.

This avoids extra low-bit perturbations from qfloat multiplication when products fall at or near FP16 rounding midpoints. The default implementation uses the vector integer path and retains `HTP_V68_DEQUANT_SCALAR_REFERENCE` as a diagnostic branch.

Independent tests enumerate **256 × 65,536 = 16,777,216** INT8 × FP16 scale bit-pattern combinations and use adjacent different scales to check lane order. See [INTEGER_DEQUANT.md](../htp-ops-lib/tests/v68/INTEGER_DEQUANT.md). Full-model validation covers IQ4_NL + Q8_0. No full Q4_0 model was validated; shared dequantization code does not imply equivalent test coverage.

### 2.3 Independent GEMM references must model FP16 boundaries

[matmul_reference.h](../htp-ops-lib/tests/v68/matmul_reference.h) does not call the DSP kernel under test. It decodes the weight layout, constructs inputs, and independently computes reference results: FP32 activations are first rounded to FP16; quantized values multiplied by their stored FP16 scales are rounded to FP16; dot products use high-precision reference accumulation, followed by FP16 output rounding.

The quantized output-stationary branch applies when `M >= 128 && K > N && N > 1024 && K < 16384`. It writes FP16 output after every 512 K elements and reloads the previous partial sum through an identity tile. Its reference therefore performs `half(previous + partial_sum)` for each chunk. A reference that rounds only once after the full K dimension would not model this path.

The device-test tolerance is `0.01 + 0.001 * abs(reference)`, allowing rounding and cancellation-related differences while detecting layout errors. All 31 final GEMM/pipeline cases actually had maximum absolute error **0**. This result is relative to the declared FP16 reference; it does not mean bitwise equality with full FP32 GEMM.

## 3. Rewritten HMX attention path for v68

The `HTP_HMX_V68` branch in [flash_attn.c](../htp-ops-lib/src/dsp/ops/flash_attn.c) implements a separate tiled path. The original v73 implementation remains in another conditional branch of the same file.

The interface keeps Q/O in FP32 and K/V in FP16, with Q/O layout `[token][head][D]` and K/V layout `[token][kv_head][D]`. The mask is an FP16 additive mask whose row stride is aligned to 64 KV elements. The public function's historical `__fp16 *` declarations do not describe Q/O's actual storage type.

Each task processes up to 32 query/grouped-head rows for one KV head:

1. Convert Q to FP16 and pack Q/K into HMX tiles, zero-padding tails.
2. Execute QK on HMX, then multiply the result by `1/sqrt(D)` and add the mask in FP32.
3. Use an FP32 running maximum, denominator, and output-combination state for stable online softmax.
4. Normalize P before converting it to FP16 and executing PV with FP16 V on HMX. This avoids requiring HMX output to hold a long unnormalized sum.
5. Return zero for fully masked rows and exclude invalid V data from matrix products for tail positions or entirely invisible keys.

Each worker's scratch capacity is checked against its actual D and available VTCM. Each HMX dot acquires ownership on its worker, loads the zero control region, and clears the accumulator. Tests cover Qwen's GQA=7. The kernel requires `D <= 512`, and the current host RPC interface does not support arbitrary strides, ALiBi, or logit softcap.

The 9 device cases had maximum absolute error **0.000559777** against a full-precision softmax reference, with a test tolerance of 0.003. This reflects FP16 boundaries in QK/PV; equality with pure FP32 attention is not required. Direct operator tests support a null mask. The llama host routing currently requires a mask that meets its RPC mapping conditions, so direct kernel capability does not imply that every GGML graph can be offloaded. See [ATTENTION.md](../htp-ops-lib/tests/v68/ATTENTION.md).

## 4. Host routing and IPC: general correctness fixes

These issues were exposed and fixed during the port. They are separate from instruction-set limitations of the Snapdragon 888.

### 4.1 Preserve the upstream model format and prevent invalid CPU fallback

The upstream [HTP converter](../llama.cpp-npu/extras/convert_hf_to_gguf_htp.py) already applies a 32×32 permutation to seven layer-weight families: attention Q/K/V/output and FFN up/down/gate. This port does not modify the converter, quantizer, model architecture, or tokenizer. It also retains the upstream quantized superblock ABI selected by `REPACK_FOR_HVX=1`.

[htp-ops.cc](../llama.cpp-npu/ggml/src/ggml-htp/htp-ops.cc) precisely recognizes these seven `blk.<id>.*.weight` families, including copy names decorated by the GGML scheduler. Because these weights are permuted, unsupported DSP execution or failed initialization cannot safely fall back to ordinary CPU matmul reading row-major data. Unmet execution requirements therefore cause an immediate error instead of silently producing incorrect results; [htp-cpu-impl.c](../llama.cpp-npu/ggml/src/ggml-htp/htp-cpu-impl.c) enforces this at the CPU-fallback boundary.

`token_embd.weight` and the final `output.weight` retain their ordinary layout. [ggml-htp.cc](../llama.cpp-npu/ggml/src/ggml-htp/ggml-htp.cc) leaves these operations on the normal CPU-buffer path; other valid ordinary operations may still use the CPU. No automatic format marker was added to GGUF, and tensor names alone do not prove that a file underwent HTP conversion: users must follow the converter's layout contract.

The host also strictly checks matmul dimensions, types, contiguity, and RPC buffers, plus attention's fixed layout and scale. The RPC ABI does not carry arbitrary strides/scales and cannot represent GGML operations beyond those constraints. RMSNorm remains on the CPU because its epsilon is not passed through the RPC ABI. `HTP_DISABLE_FLASH_ATTN=1` can diagnose the CPU attention path.

### 4.2 Propagate DSP failures to the caller

[ggml-htp.cc](../llama.cpp-npu/ggml/src/ggml-htp/ggml-htp.cc) loads the dynamic library with immediate symbol resolution and checks required symbols, the DSP session, backend initialization, and message-channel results. Initialization directly checks the generated FastRPC stub's return value, avoiding the original convenience wrapper that discarded errors.

[htp-ops.cc](../llama.cpp-npu/ggml/src/ggml-htp/htp-ops.cc) serializes access to the shared mapper, parameter buffer, and message buffer. Requests start with a pending status, and map, unmap, and compute completion statuses are checked. The default per-request timeout is 120 seconds, configurable through `HTP_OP_TIMEOUT_MS`. A timeout terminates execution rather than reusing buffers the DSP might still access.

`HTP_TRACE=1` reports successful requests' opcodes, cumulative counts, tensor names, and host-observed elapsed time. It helps confirm requests and diagnose problems; it is not an HMX instruction counter, NPU utilization measurement, or power measurement.

### 4.3 Lifecycle, synchronization, and CPU workspace

[commu.c](../htp-ops-lib/src/dsp/commu.c) publishes channel state before starting the receiver thread and uses an atomic stop flag. Empty messages continue waiting. Shutdown waits for the thread to exit before releasing mappings and its stack. Some initialization-failure paths also gain resource cleanup. [op_executor.cc](../htp-ops-lib/src/dsp/op_executor.cc) rejects unknown opcodes and maps mask data using its actual aligned stride.

[ggml-htp.cc](../llama.cpp-npu/ggml/src/ggml-htp/ggml-htp.cc) fixes CPU workspace-capacity bookkeeping and registers a thread-setting interface so `-t` / `-tb` can affect CPU work in the hybrid backend. These fixes support reproducibility and performance measurement; they do not by themselves establish an HMX hardware speedup.

## 5. Validation coverage and limits

The device-test entry point is [v68_test.c](../htp-ops-lib/src/host/v68_test.c):

| Command | Coverage |
| --- | --- |
| `htp_v68_test --hmx --full` | 27 F16/Q8_0/IQ4_NL cases, including M=1/5/32/33, K=896/4864, N=896, multiple output tiles, and asymmetric data |
| `htp_v68_test --pipeline` | Two cases each for Q8_0/IQ4_NL: M=128/K=896/N=896 and M=129/K=1568/N=1056, covering the four-stage pipeline and K-chunked output-stationary path |
| `htp_v68_test --hmx-attention` | 9 attention cases covering GQA, additive/null masks, fully masked rows, and KV tails |
| `htp_v68_test` | Operator tests for the retained HVX path; run against the corresponding HVX build |

Independent reference computation uses ordinary host memory, avoiding large CPU reference loops over uncached RPC buffers. Outputs have sentinels and boundary guards, and RPC waits are bounded. See [TESTING.md](TESTING.md) for the scope and limitations of host automation.

The final HMX vector implementation passed all 40 device operator cases, full-model runs generating 32 tokens each for F16 and mixed quantization, and the F16 fixed-input probability comparison. See the [validated results](../README.md#validated-results) and [sanitized records](../results/README.md). These demonstrate execution and matching results on the tested 888; they do not establish support for larger models, long contexts, every firmware, or arbitrary quantization configurations.

Current short-run HMX F16 pp32/tg8 performance is approximately 27.57/4.42 tokens/s, below the CPU F16 result of 42.58/8.88 with the same parameters. These timings include CPU operations, FastRPC, packing, and dequantization. Temperature was not controlled and affinity was not tuned. The current result is therefore **a functional v68 HMX port with numerical validation**, not an established end-to-end speedup over the CPU.
