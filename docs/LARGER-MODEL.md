# Qwen2.5-1.5B on Snapdragon 888

Qwen2.5-1.5B-Instruct completed single-sequence and 32-sequence HMX inference on
the OnePlus 9 LE2115 / SM8350 / Hexagon v68. The successful model uses the
original fork's **mixed Q4_0 + F16 storage**, with HVX dequantization to FP16
before HMX matrix computation. Full F16 inference failed during shared-buffer
mapping in the tested configurations. The larger model completed 12-token
single-sequence decoding and two tokens in each of 32 independent sequences.
Phone-local activity captures do not establish HMX saturation. Numerical
checks identified drift dominated by the HMX attention path in this fixture;
switching attention to CPU while retaining HMX GEMMs substantially reduced it.

The experiment was conducted on 2026-09-30 using the existing Android 14 phone.
The development computer compiled, staged and retrieved artifacts. Inference
and the `qcphoneperf` Python/C++ collector execute on the phone; the runtime
does not require a computer service, HTTP endpoint, collector CLI or APK.

## Model and file identity

The source is the official
[Qwen2.5-1.5B-Instruct repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
pinned to revision
[`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commit/989aa7980e4cf806f80c7fef2b1adb7bc71aa306).
Its configuration has 28 layers, hidden width 1536, FFN width 8960, 12 query
heads, two KV heads, head dimension 128 and a tied 151936-token embedding/output
matrix. The original Safetensors data is converted without executing external
model code.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Source `model.safetensors` | 3,087,467,144 | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| HTP-permuted F16 GGUF | 3,093,669,376 | `09cdc85993ca1dff81f919ba3338674aac6b6fb1e302832335f75968f0a6092e` |
| Ordinary CPU F16 GGUF | 3,093,669,376 | `b6eaec3509f1d0373d1f4802654c4a7bfcde0768645d2d45e8885f6922b428ee` |
| HTP-permuted, HVX-superpacked Q4_0 + F16 GGUF | 1,504,724,992 | `eb4e9683beac9d347fed3444ceebe24994e3005004caa09fe2a7557698c8a72f` |

The HTP and ordinary GGUF files have different storage layouts. A standard
GGUF cannot be substituted for an HTP-permuted model. GGUF does not encode a
marker for this private layout, so the conversion and quantization provenance
are part of the model contract.

## Changes needed for the wider FFN

FFN-down has logical `K=8960`. Even when logical `M=1`, HMX FP16 storage reserves
one physical 32-row activation tile. That requires
`32 × 8960 × 2 = 573440` bytes, exceeding the old 512 KiB preferred activation
area. The old size calculation could therefore select zero rows.

The [F16 VTCM planner](../htp-ops-lib/include/dsp/f16_vtcm_layout.h) now permits
the minimum physical activation tile for wider K, then fits weights,
activations, output and 256 bytes of control storage together inside the
manager's actual usable VTCM. For `M=32, K=8960, N=1536` and 3.75 MiB of usable
VTCM, it selects physical chunks of **32 rows and 160 columns**. The 160 columns
are a chunk of the output width, not the model's total N. The planner also checks
alignment, integer bounds and the DMA descriptor length limit. Host tests passed
780 shape/capacity cases under ASan and UBSan.

Enable the existing opt-in `HTP_F16_M_AWARE_VTCM=ON` build option; its
unset default remains OFF.

This adjustment is confined to the F16 path. It does not enlarge the shared
fixed areas used by the quantized path. Keeping FFN-down as F16 in the mixed
model avoids sending its wider K through the existing small-M, full-K
quantized path, which has a separate capacity limitation.

## Full F16 mapping failures

The following attempts did not complete inference. Layer numbers are the
zero-based layer IDs reported by the backend.

| Active mapping-cache budget | Whole-tensor buffer packing cap | Reported failure layer |
| --- | --- | ---: |
| Default 3 GiB | Default 256 MiB | 13 |
| 512 MiB | Default 256 MiB | 18 |
| 256 MiB | Default 256 MiB | 18 |
| 256 MiB | 64 MiB | 20 |

The observed kernel message reports `dma_buf_map_attachment` failing with
`ret -12` (`-ENOMEM`) in `fastrpc_mmap_create`. This identifies a host
DMA/IOMMU mapping allocation failure. It does not establish an exact IOVA
capacity, a DSP virtual-address limit, or ordinary RAM exhaustion on the
installed kernel.

Qualcomm's downstream FastRPC/ION sources document delayed IOMMU unmapping
that can retain a mapping until its DMA-BUF allocation is destroyed. An active
userspace LRU budget therefore need not bound all retained kernel mappings.
The original upstream author's
[FastRPC issue #137](https://github.com/qualcomm/fastrpc/issues/137) reports a
similar cumulative-mapping failure on a different OnePlus device: explicit
unmaps did not restore capacity, while freeing old RPCMEM allocations did.
That report and the
[lazy mapping API](https://android.googlesource.com/kernel/msm/+/a4f7977a70291f0f273aea4de415861f6b768746/include/linux/msm_dma_iommu_mapping.h)
support a possible explanation; they do not prove that the installed 888 kernel
uses an identical implementation. Aperture fragmentation or other kernel
allocation pressure remains possible.

Two optional settings make these experiments reproducible:

| Variable | Experiment value | Unset behavior | Scope |
| --- | --- | --- | --- |
| `GGML_HTP_MAP_BUDGET_MB` | `256` | Existing 3 GiB active mapping-cache budget | Host LRU accounting; pending retirements finish after the synchronous DSP request |
| `GGML_HTP_MAX_BUFFER_MB` | `64` | Existing 256 MiB packing cap | Whole-tensor model/context/KV buffer packing through the GGML maximum-size hook |

The second setting does not split individual tensors and does not guarantee
that every RPCMEM allocation is at most 64 MiB. Generic graph allocations can
bypass that hook; separately packed output-head allocations retain their own
contract. The observed single-sequence graph buffer was approximately 7 MiB.
The cap is cached at its first use in the process, so configure it before
starting the program rather than changing it between model loads. Both unset
defaults remain unchanged.

## Precision in the successful mixed model

The existing `Q4_0+F16` quantization policy chooses:

| Tensor family | Stored precision/layout | Computation in this experiment |
| --- | --- | --- |
| Attention Q/K/output; FFN up/gate | Q4_0, HTP tiles and HVX superblocks | HVX unpacks and dequantizes to FP16; HMX uses FP16 tiles |
| Attention V; FFN down | F16, HTP tiles | HMX FP16 matrix computation |
| Tied token embedding/final output | Q6_K, ordinary GGUF layout | Existing CPU path |
| Norms, biases and other ordinary tensors | Types selected by the converter | Existing hybrid CPU path |

HMX is performing FP16 matrix computation here, rather than native INT4 GEMM.
The quantized model reduces shared-weight storage while retaining the larger
1.5B matrix dimensions. `HTP_OFFLOAD_OUTPUT=0` keeps the final projection on
the ordinary CPU path; the optional packed-head path is not enabled for this
larger model. This backend does not use the GPU.

For one sequence, logical decode M remains one. A larger K or N increases work
per matrix operation but does not create 32 useful independent rows. Activity
counters do not reveal a physical row-enable mask. Testing 32 independent
sequences can supply 32 real decode inputs; it is a separate workload. The
tested 32-sequence case completed, but its activity does not establish saturation.

## Verified smoke test

The library caller submitted one sequence with the raw prompt
`The capital of France is`, requested four greedy output tokens, ignored EOS,
used four CPU threads, requested 128 context tokens per sequence, enabled flash
attention and attached a persistent ordinary CPU threadpool.

| Observation | Measured result |
| --- | --- |
| Process exit status | `0` |
| Generated token IDs | `[12095, 11, 323, 279]` |
| Generated bytes as text | ` Paris, and the` |
| Generated bytes as hexadecimal | `2050617269732c20616e6420746865` |
| Prefill time | 15.219575776 s |
| Generation window | 9.364053590 s |
| Backend decode time, three calls | 9.350432288 s |
| Timed decode input tokens/s | 0.32084078121715176 |

The first output token uses prefill logits; four generated tokens therefore
require three timed decode inputs. The reported decode-input rate is
`3 / decode_backend_seconds`; the generation window also includes sampling.
Output-token throughput instead counts the first token from prefill.
**Tracing was enabled (`HTP_TRACE=1`) in this diagnostic run.** This single
short run is not a throughput benchmark, a matched CPU comparison, or an HMX
utilization measurement. Generated text alone does not validate numerical
accuracy.


## Phone-local activity measurements

Eight captures completed: two runs for each size/width, with the repeat series
reversing width and model order. Both sizes use the same Q4_0+F16 policy,
CPU final projection, four ordinary CPU workers, one hybrid worker and the
same optional mapping settings. Tracing is disabled. M1 generates 12 output
tokens, while M32 generates two per sequence; compare model sizes within a
width. Each M32 measurement contains one timed decode call with 32 real tokens.

The primary activity window is generation, including host sampling. Only
complete corrected SysMon intervals inside that window are integrated.
Ratios below divide measured activity MCPS by the matching continuous
single-context control from [LOAD-CALIBRATION.md](LOAD-CALIBRATION.md),
at the same observed QDSP clock of 1420.8 MHz. These ratios are not peak MAC
capacity, hardware lane occupancy or the fraction of useful physical rows.

| Model | Real decode width | Timed aggregate decode inputs/s | HVX activity / continuous control | HMX activity / continuous control | Generation coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 1 | 0.484–1.125 | 15.13–35.19% | 0.210–0.489% | 98.67–99.71% |
| 1.5B | 1 | 0.328–0.328 | 37.31–37.35% | 0.504–0.504% | 99.53–99.85% |
| 0.5B | 32 | 14.588–14.626 | 14.65–14.72% | 0.248–0.249% | 95.83–95.84% |
| 1.5B | 32 | 5.409–5.429 | 19.32–19.60% | 0.308–0.313% | 98.29–99.59% |

Ranges show the two observed runs, not statistical confidence intervals.
Generation contains 22–335 complete approximately 100 ms intervals per run.
Decode throughput excludes sampling and counts 11 timed M1 inputs or 32
timed M32 inputs, excluding the first output token obtained from prefill.
The export also retains the separate prefill and decode-call-union summaries.

**Increasing the model size did not establish HMX saturation.** The stable
1.5B M1 result stays near 0.50% of the continuous HMX control; M32 is near
0.31%. HVX activity is substantially higher. The backend still performs
weight dequantization, activation conversion, packing, transport and ordinary
CPU work around HMX. These observations do not isolate every component's
critical-path duration.

0.5B M1 changes from 0.484 to 1.125 timed inputs/s and from 0.210% to 0.489%
HMX relative activity between runs. Its first capture has early decode calls
around 4.7 seconds and later calls around 0.89 seconds. That variability is
retained rather than attributed to an unmeasured cause. CPU frequency/affinity,
cache state and temperature were not experimentally controlled. Thermal
readings are preserved; a larger-model speedup or architecture-only causal
claim is not supported. The 0.5B source is the base variant, while 1.5B is
Instruct.

M1 still provides one useful logical row per matrix operation; M32 provides
32. The independent caller records separate sequence IDs, an actual physical
microbatch of 32 and a decode histogram of `{32: 1}`. That establishes the
submitted workload, not a physical HMX row-enable mask. A larger K/N at M1
does not create more useful rows.

## Reproduction through the libraries

Follow [BUILD.md](BUILD.md) for the v68 HMX library build and matching ARM64/DSP
artifacts. Convert the pinned model using this checkout's HTP converter, then
quantize that converted file with the same checkout's quantizer:

```sh
python3 tools/build.py --backend hmx --m-aware-vtcm --jobs 6

PYTHONPATH=llama.cpp-npu/gguf-py python3 llama.cpp-npu/extras/convert_hf_to_gguf_htp.py \
  --outfile models/qwen2.5-1.5b-instruct-f16-hmx.gguf --outtype f16 \
  /path/to/pinned/Qwen2.5-1.5B-Instruct

REPACK_FOR_HVX=1 llama.cpp-npu/build-host/bin/llama-quantize \
  models/qwen2.5-1.5b-instruct-f16-hmx.gguf \
  models/qwen2.5-1.5b-instruct-q4_0-f16-hmx.gguf 'Q4_0+F16'
```

Before launching the phone program, set its loader paths to the staged matching
libraries and configure the backend. The following uses an example isolated
deployment directory:

```sh
export LD_LIBRARY_PATH=/data/local/tmp/larger-model/payload:/vendor/lib64
export ADSP_LIBRARY_PATH='/data/local/tmp/larger-model/payload/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp;/system/lib/rfsa/adsp'
export DSP_LIBRARY_PATH="$ADSP_LIBRARY_PATH"
export GGML_HTP_MAP_BUDGET_MB=256
export GGML_HTP_MAX_BUFFER_MB=64
export HTP_REUSE_THREADPOOL=1
export HTP_SPLIT_CPU_OPS=1
export HTP_FAST_CONCAT=1
export HTP_HYBRID_THREADS=1
export HTP_OFFLOAD_OUTPUT=0
export HTP_TRACE=0
```

Use `HTP_TRACE=1` only when reproducing the diagnostic trace. The ordinary CPU
threadpool is also attached through `llama_attach_threadpool`; it is separate
from the HTP hybrid threadpool. The existing
[C++ caller](../experiments/load-calibration/inference/batch_decode.cpp) uses
`llama_load_model_from_file`, `llama_new_context_with_model`, `llama_decode` and
independent greedy samplers. Its context configuration is:

```cpp
auto parameters = llama_context_default_params();
parameters.n_ctx = parallel * 128;
parameters.n_seq_max = parallel;
parameters.n_batch = std::max({32, total_prompt_tokens, parallel});
parameters.n_ubatch = parallel;
parameters.n_threads = 4;
parameters.n_threads_batch = 4;
parameters.flash_attn = true;
```

Use `parallel=1` for the smoke case. Record the context and microbatch sizes
returned by the library rather than assuming every requested size is exact.
For independent batched decode, each active sequence contributes a real token
with its own sequence ID and position.

The collector is an importable phone-local function. After staging the Python
bindings, native adapter, clock helper and authorized Qualcomm SDK dependencies,
call `collect(Path(fresh_directory), config, Path(clock_library))` from
[`capture_workload.py`](../experiments/load-calibration/capture_workload.py).
It uses `qcphoneperf.Profiler.snapshot_metrics()` in the same phone process.
The optional research inference executable is a workload caller, not a collector
CLI. The tested profiler access requires the phone's existing root permissions.
See [the library capture configuration](../experiments/load-calibration/README.md)
and [activity calibration](LOAD-CALIBRATION.md) for API setup, event semantics
and version-specific timestamp handling.

## Numerical validation

Independently quantizing an ordinary F16 GGUF and its HTP-permuted counterpart
does not preserve identical effective weights: the permutation changes which
32 values share each Q4 scale. A comparison between those independently
quantized models combines backend differences with quantization differences.

The reconstruction helper at
[`experiments/larger-model/reconstruct_reference.py`](../experiments/larger-model/reconstruct_reference.py)
instead reads the actual HTP mixed model, reverses its HVX superblocks,
dequantizes Q4 values to the same rounded FP16 products supplied to HMX, and
reverses the HTP tiles. It preserves the ordinary Q6_K embedding/output bytes.
The resulting ordinary CPU reference is for numerical checking; its larger
F16 layer storage makes it unsuitable as a same-storage performance baseline.
CPU activation and accumulation precision still differ from HMX. The actual
CPU runtime feature line reports NEON and ARM_FMA but no FP16_VA. In this
checkout that feature is emitted only when native FP16 vector arithmetic is
compiled in. The generic F16 dot fallback widens half operands for FP32
accumulation; the enabled LLAMAFILE path can instead accept FP32 activations
with F16 weights. CPU activation precision therefore depends on dispatch.
HMX rounds activations and matrix outputs to FP16. The v68 HMX accumulator's
exact internal precision has not been established.

The same-effective-weights CPU and HMX smoke runs match all four output
tokens and bytes. In the 12-token M1 capture, the first eight tokens agree;
token nine differs: CPU chooses `also` (1083), while HMX chooses `the` (279).
Subsequent values then reflect different token histories. This failed full
greedy comparison is retained; matching the smoke prefix does not establish
complete numerical equivalence. All 32 two-token sequence outputs match the
CPU prefix `[12095, 11]`.

A teacher-forced comparison uses the committed 2039-byte English fixture
(328 tokenizer tokens), one 128-token chunk, physical microbatches of 32, and
63 scored next-token positions. Every backend receives the same token history.
The CPU baseline file contains scaled uint16 log probabilities; it is not raw
FP32 logits. The perplexity tool decodes that baseline and truncates very small
baseline probabilities when computing CPU-to-HMX KL. Its `Delta p RMS` metric
is the error in the **ground-truth next-token probability** across scored
positions, rather than an RMS over every vocabulary probability.

| Configuration | PPL | Mean KL versus CPU | Ground-truth probability RMS error | Same top token |
| --- | ---: | ---: | ---: | ---: |
| Same-effective-weights CPU | 120.9477 | Baseline | Baseline | Baseline |
| HMX GEMMs and HMX attention | 118.4158 | 0.01407 ± 0.00099 | 1.956 ± 0.428% | 58/63 (92.063%) |
| HMX GEMMs and CPU attention | 120.7298 | 0.00026 ± 0.00003 | 0.353 ± 0.118% | 63/63 (100%) |

The printed uncertainties are the tool's estimates across scored positions,
not repeated-run confidence intervals. PPL on this short custom fixture is a
diagnostic, not a model-quality benchmark. The CPU-attention isolation sets
only `HTP_DISABLE_FLASH_ATTN=1`; model weights, GEMMs, input and other settings
remain fixed. Trace logs contain successful HMX FFN-down operations with
`M=32, K=8960, N=1536`, and no DSP attention dispatches in that isolation.

The same-history greedy check also inspects all vocabulary logits for finite
values before selecting each token. At the ninth output, CPU gives `also`
20.2140 and `the` 19.7900 (margin 0.4240); default HMX attention gives `the`
20.1714 and `also` 20.1075 (margin 0.0639). This is not an exact tie. With CPU
attention, all nine checked output tokens match CPU, and the ninth-token margin
is 0.3759 for `also`. The diagnostics and first difference are retained.

This isolation localizes most measured drift to the HMX attention path under
these conditions. It does not establish a particular indexing defect,
attribute the difference solely to FP16 rounding, or prove equivalence on
other prompts and lengths. HMX QK, probability and PV conversion boundaries
remain candidates for further real-input operator testing. For research that
requires closer agreement with this CPU reference, the **tested workaround**
is to set `HTP_DISABLE_FLASH_ATTN=1` before creating the context. GEMMs still
run on HMX. The activity table above uses HMX attention; it must not be reused
as a measurement of the CPU-attention configuration.

Run probability and logit diagnostics separately from activity captures.
The perplexity caller needs a writable current directory for its relative
`./cache`; `LLAMA_CACHE` alone did not prevent a cache permission failure.
This checkout's `llama-perplexity` does not accept `--no-warmup`. Two cache
setup failures and one unsupported-option failure occurred before model
inference; they are preserved separately from arithmetic and mapping results.

The [compact evidence export](../results/LARGER-MODEL-20260930.json)
contains phase records, token IDs, telemetry, provenance, numerical diagnostics
and failed trial metadata. All eight activity captures completed with successful
workload and collector cleanup. The final device check found no active experiment
workload and verified the profiler API hash used for timestamp correction.
No physical-capacity or row-occupancy percentage is established. Qualcomm SDK
binaries and raw proprietary databases are not included in the public export.
