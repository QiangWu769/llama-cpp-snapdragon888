# llama.cpp on Snapdragon 888 / Hexagon v68

A port of the research implementations in [haozixu/llama.cpp-npu](https://github.com/haozixu/llama.cpp-npu) and [haozixu/htp-ops-lib](https://github.com/haozixu/htp-ops-lib) to **Snapdragon 888 (SM8350 / Hexagon v68)**. Qwen2.5-0.5B runs with real HMX matrix instructions on a OnePlus 9 running Android 14, using both F16 and mixed IQ4_NL + Q8_0 models.

This is a **hybrid CPU + cDSP LLM inference backend**. HMX executes matrix multiplication and attention QK/PV products; HVX and scalar DSP code handle conversion, packing and other computation. Embedding lookup, final output projection and ordinary operations run on the CPU. The GPU is unused. This backend can support on-device Agent inference; the repository does not include a mobile GUI Agent application.

**End-to-end inference works, but the current short benchmarks do not outperform CPU F16.** The repository provides inspectable v68 port source, numerical validation and reproduction instructions.

## What changed from upstream to run on Snapdragon 888?

The upstream README requires Snapdragon 8 Gen 2 or newer and recommends `DSP_ARCH=v73`. This port required more than changing the compilation target to `v68`:

| Component | Upstream v73 path | v68 adaptation and source |
| --- | --- | --- |
| HMX output | `cvt.hf = acc`, followed by `mxmem = cvt` | Legacy `mxmem(..., 2047):after.hf = acc` with explicit accumulator clearing; [hmx_utils.h](htp-ops-lib/include/dsp/hmx_utils.h) |
| HMX bias / conversion control | `bias = mxmem2(...)` with FP16 scale initialization | `bias = mxmem(...)` with a zero-filled legacy control region verified on hardware. Numerical scaling is applied separately; the legacy fields must not be interpreted as FP16 scales; [hmx_utils.h](htp-ops-lib/include/dsp/hmx_utils.h) |
| HMX ownership | Shared `HAP_compute_res_hmx_lock2` API | Acquire and release the legacy exclusive lock on the worker that executes the matrix instructions; [hmx_mgr.c](htp-ops-lib/src/dsp/hmx_mgr.c) |
| VTCM | Multiple 1 MiB work areas | 512 KiB work areas, capacity checks and synchronization barriers for the measured 4 MiB VTCM; [mat_mul.c](htp-ops-lib/src/dsp/ops/mat_mul.c), [vtcm_mgr.cc](htp-ops-lib/src/dsp/vtcm_mgr.cc) |
| Conversion and dequantization | qfloat conversion/multiplication chains | Integer HVX operations preserve IEEE FP16/FP32 conversion and quantized-product rounding, with corrected lane pairing; [hvx_convert.h](htp-ops-lib/include/dsp/hvx_convert.h), [mat_mul.c](htp-ops-lib/src/dsp/ops/mat_mul.c) |
| FlashAttention | Original v73 implementation | A v68 QK/PV HMX branch with explicit scaling, FP32 online softmax, GQA, masks and tail handling; [flash_attn.c](htp-ops-lib/src/dsp/ops/flash_attn.c) |
| Host and communication | Some failures could continue through CPU paths; requests lacked bounded waits | Route ordinary and packed weights precisely; stop on packed-matrix failures; check initialization, RPC status and timeouts; [ggml-htp](llama.cpp-npu/ggml/src/ggml-htp), [commu.c](htp-ops-lib/src/dsp/commu.c) |
| Build and validation | Original HMX build and tests | HMX/HVX selection, a retained v68 HVX reference backend, independent numerical references and real-device RPC tests; [CMakeLists.txt](htp-ops-lib/CMakeLists.txt), [tests/v68](htp-ops-lib/tests/v68) |

Host error propagation, layout routing and thread cleanup are general correctness and robustness fixes, rather than v68-specific ISA requirements. See the **[Snapdragon 888 port notes](docs/888-PORT.md)** for the detailed explanation.

**The model architecture, tokenizer, upstream HTP converter and quantizer are unchanged.** The port retains the upstream 32x32 permutation of seven layer-weight families and the quantized packing format selected by `REPACK_FOR_HVX=1`. Quantized weights are dequantized to FP16 before HMX execution; this is not native INT4 HMX computation. Ordinary GGUF and HTP-packed GGUF files are not interchangeable.

## Quick start

See **[BUILD.md](docs/BUILD.md)** for environment setup, compilation, conversion and deployment. The parameterized entry points are [tools/build.py](tools/build.py), [tools/deploy.py](tools/deploy.py) and [tools/device.py](tools/device.py). Supply the SDK, NDK, model weights and device runtime separately.

Tested toolchain: Linux x86-64, Android NDK r26d, Hexagon SDK 6.6.0.0 and Hexagon Tools 19.0.07. Tested device: OnePlus 9 LE2115, Snapdragon 888, Android 14. These runs used ordinary ADB shell and unsigned cDSP FastRPC, without invoking `su` or modifying system/vendor partitions.

| Mode | Key build settings | Purpose |
| --- | --- | --- |
| v68 HMX | Script sets `DSP_VERSION=v68`, `HTP_USE_HMX=ON`; llama `GGML_HTP=ON` | This HMX implementation |
| v68 HVX | Script sets `DSP_VERSION=v68`, `HTP_USE_HMX=OFF`; llama `GGML_HTP=ON` | Retained correctness reference backend |
| CPU | llama `GGML_HTP=OFF`, with separate executable and library directories | CPU comparison using ordinary-layout GGUF |

`HTP_USE_HMX` defaults to OFF on v68; the `--backend hmx` build script explicitly enables it. Set `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT` and `ANDROID_NDK` as described in the build guide, and prepare a converted model before running:

```sh
python3 tools/build.py --backend hmx
python3 tools/deploy.py --backend hmx --model /path/to/qwen2.5-0.5b-f16-hmx.gguf
python3 tools/device.py --backend hmx -- ./htp_v68_test --hmx --full
python3 tools/device.py --backend hmx -- ./htp_v68_test --pipeline
python3 tools/device.py --backend hmx -- ./htp_v68_test --hmx-attention

python3 tools/device.py --backend hmx --trace -- \
  ./llama-cli -m qwen2.5-0.5b-f16-hmx.gguf \
  -p 'The capital of France is' -n 8 -c 128 -b 16 -ub 16 \
  -t 4 -tb 4 -fa --temp 0 --seed 1234 --no-display-prompt
```

HTP automatically registers as an ACCEL backend in this fork. `-ngl 0` or `--device none` is insufficient to establish a CPU-only comparison. Use a separate `GGML_HTP=OFF` build and ordinary GGUF. `HTP_TRACE=1` records host-observed DSP request counts, timing and status; **it does not measure NPU utilization**.

## Validated results

**New: [HMX feature validation against the V81 manual](docs/HMX-FEATURES.md).**
87 independent Snapdragon 888 probes establish partial K, K=1024 deep
matrices, 64-channel weight deep, shifted windows, accumulator control and
legacy bias/ReLU behavior. The report also retains incompatible V81 controls
and the modern CVT architecture gates, with standalone source and raw logs.

The final vector implementation passed **40 real-device FastRPC operator cases**:

| Validation | Result |
| --- | --- |
| 27 F16 / Q8_0 / IQ4_NL GEMM cases + 4 Q8_0 / IQ4_NL pipeline cases | Maximum absolute error of 0 in every case against independent references that include the actual FP16 rounding boundaries |
| 9 HMX attention cases | Maximum absolute error of 0.000559777 against a full-precision softmax reference |
| Full Qwen2.5-0.5B F16 model, 32 generated tokens | Successful exit; 5,376 F16 matrix requests and 768 attention requests |
| Full Qwen2.5-0.5B IQ4_NL + Q8_0 model, 32 generated tokens | Successful exit; 3,840 IQ4_NL, 1,536 Q8_0 matrix requests and 768 attention requests |
| Fixed 128-token F16 input, compared with CPU over 63 positions | Mean KLD 0.00001; RMS probability difference 0.099%; top-token agreement 100% |

GEMM tests cover long K, multiple output tiles, M tails and quantized pipelines. Attention tests cover GQA=7, additive/null masks, fully masked rows and tails. The probability comparison uses this fork's scaled uint16 log probabilities, rather than raw FP32 logits. Request counts cover the corresponding complete test process and must not be interpreted directly as operator counts per generated token.

These are functional and numerical checks on a small model, rather than general model-accuracy conclusions. Host tests additionally cover memory layout, IEEE conversion and all 16,777,216 INT8 x FP16 scale bit-pattern combinations; host models do not replace real DSP validation. Public records are in [results/README.md](results/README.md). Methods and reproduction entry points are in [TESTING.md](docs/TESTING.md) and the [operator test notes](htp-ops-lib/tests/v68/README.md).

## Current performance

Short-sequence `llama-bench` runs use the same Qwen2.5-0.5B model with `-p 32 -n 8 -b 32 -ub 32 -t 4 -fa 1`, default warmup and `HTP_TRACE=0`. CPU/HMX use three repetitions each; the slower HVX reference uses one.

| Backend / model | Prefill pp32, tokens/s | Decode tg8, tokens/s |
| --- | ---: | ---: |
| CPU / F16 | 42.58 | 8.88 |
| Retained HVX / F16 | 0.42 | 0.10 |
| HMX / F16 | 27.57 | 4.42 |
| HMX / IQ4_NL + Q8_0 | 14.75 | 1.01 |

HMX is faster than this repository's correctness-oriented HVX reference, but **does not yet outperform CPU F16**. The quantized row uses a different precision and has no matched CPU quantized baseline. These are complete hybrid-backend timings, including CPU work, communication, packing and dequantization; they do not represent peak HMX throughput. Temperature was not controlled, and CPU ARM compiler options and thread affinity were not tuned. No power or hardware-utilization conclusions are established.

## Provenance, layout and scope

| Directory | Pinned upstream version |
| --- | --- |
| [llama.cpp-npu/](llama.cpp-npu) | [haozixu/llama.cpp-npu @ 57e34a34](https://github.com/haozixu/llama.cpp-npu/tree/57e34a34e0293894bd026280875703561c0106ef) |
| [htp-ops-lib/](htp-ops-lib) | [haozixu/htp-ops-lib @ 85eb88ed](https://github.com/haozixu/htp-ops-lib/tree/85eb88edcafd35afff1a43606a4c47eec9a0ca0b) |

The original implementation accompanies [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/abs/2509.23324). This repository is an independent port. The initial commit preserves the pinned upstream source snapshots; subsequent commits show the adaptation for review. See [UPSTREAM.json](UPSTREAM.json) for revisions and [NOTICE.md](NOTICE.md) for attribution and licensing. No replacement blanket license is assigned to the upstream subtree that lacks a top-level license.

**[View all adaptation changes against the original snapshots](https://github.com/QiangWu769/llama-cpp-snapdragon888/compare/13a0dbb...main)**, including source, tests, tools and documentation. Locally, run `git diff 13a0dbb HEAD -- htp-ops-lib llama.cpp-npu` to inspect changes within the two original projects.

Validation covers the stated device, firmware, model and short contexts. Unsigned cDSP availability and firmware runtime compatibility on other Snapdragon 888 phones, as well as correctness for other models and long contexts, require independent verification. The repository does not distribute the Qualcomm SDK, QNN/firmware libraries, model weights or compiled device binaries.
