# Snapdragon 888 HMX utilization experiments

The subsequent [activity calibration](LOAD-CALIBRATION.md) tests the SDK percentage
against controlled DSP duty cycles and corrects an SDK timestamp scale error for
phase alignment. The historical whole-capture statistics below remain unchanged.

The target is OnePlus 9 LE2115, Snapdragon 888 / SM8350, Hexagon v68,
Android 14, with converted Qwen2.5-0.5B F16 weights. These experiments
measure useful decoding throughput and decoded HMX/HVX activity together.
The optional backend changes are disabled by default.

## Bottlenecks and changes

Increasing `llama-bench -b` does not turn its one-token generation loop into
32 independent streams. The [C++ library caller](../experiments/utilization/batch/README.md)
gives each stream a separate KV sequence ID and greedy sampler, submits one
token per active stream per decode call, and records output IDs, exact bytes
and actual batch widths. Separate traces confirm matrix M=32. The comparison
prompt is identical across streams; dummy rows are not counted as generated
tokens. Aggregate throughput does not imply lower single-stream latency.

Separately traced baseline runs put the CPU output projection at 13.75% of
M1 and 61.29% of M32 decode wall time. The M32 projection takes 1.8165–1.8244
seconds. Two packed vocabulary halves reduce observed HMX request time to
approximately 11.2 ms each. Request time includes transport and DSP conversion;
it is not isolated HMX execution time. Whole-model gains are smaller because
other operations remain.

The original hybrid executor creates and joins CPU workers per graph and uses
RPCMEM buffers for ordinary operations. Splitting those operations onto the
standard CPU executor, attaching its worker pool through the llama library
API, and keeping one hybrid worker can reduce overhead. Pool reuse alone does
not establish a speedup.

The original FP16 path reserves fixed 512 KiB matrix areas. The optional VTCM
planner uses actual usable capacity and full physical 32-row tiles for logical
M. It changes M/N chunks while retaining K accumulation order, conversion,
DMA completion and HMX ownership.

The output CONCAT initially distributes work over its third dimension. For
the vocabulary tensor that dimension is one: all 32 rows use one worker and
scalar copies. A separately traced combined run spends 800.9–821.2 ms per
decode CONCAT. The optional fast path distributes token rows across workers
and copies each vocabulary half with `memcpy`. Other axes and incompatible
layouts retain the generic path.

### Remaining critical path

In the final traced M32 decode graphs, CPU MUL totals 322–326 ms, ADD
201–203 ms, and UNARY approximately 75 ms. DSP flash-attention requests total
195–200 ms and matrix-product requests approximately 143 ms. CPU-node times
include worker synchronization; DSP-request times include transport, conversion
and DSP computation. These diagnostic timings identify serial host work, not
pure HMX execution or a wall-time utilization percentage.

The split schedule can still give CPU operations uncached RPCMEM inputs.
RPCMEM is advertised as host-accessible, the CPU backend accepts that buffer
type, and the scheduler need not copy an accepted input. Thus FFN SiLU and
multiply can read DSP products directly even when their CPU outputs use normal
cached storage. This makes uncached-input latency a plausible contributor;
no matched cache-policy experiment or cache-miss measurement establishes it
as the cause.

Two **unimplemented** next experiments are an explicit cache-stage microbenchmark
that times input copies and CPU arithmetic separately, and an HVX fusion of
FFN SiLU with its elementwise multiply before the down projection. Fusion
could remove CPU reads and a return copy while retaining F32 intermediates and
the existing downstream F16 conversion. An HVX exponential approximation or
changed rounding requires independent numerical validation, including distinct
prompts; a higher activity sample alone is not a benefit.

Changing the allocation flag alone is unsafe. The normal transport polls a
shared message channel and uses file-descriptor mappings with caller-managed
coherency. DSP cache maintenance does not make dirty or stale ARM cache lines
coherent. A cached tensor path needs explicit CPU/DSP ownership transfers,
clean/invalidate ordering and correct ranges for aliases and views; the control
channel needs its own protocol. Matching host and DSP changes must be validated
together before measuring throughput.

## Controls

| Control | Effect |
| --- | --- |
| `HTP_SPLIT_CPU_OPS=1` | Advertise packed matrix products and supported flash attention to HTP; route ordinary operations to the standard CPU executor. |
| `HTP_REUSE_THREADPOOL=1` | Reuse the hybrid worker pool. Read once during shared HTP context initialization. |
| `HTP_HYBRID_THREADS=1` | Override hybrid workers only; ordinary CPU workers retain their configured count. Valid values: 1–8. |
| Caller `--reuse-cpu-threadpool` | Attach an ordinary CPU pool through `llama_attach_threadpool`; independent of the hybrid pool. |
| `HTP_OFFLOAD_OUTPUT=1` | Create two packed HMX vocabulary projections for the validated tied Qwen2 F16 `[896,151936]` weights. |
| `HTP_FAST_CONCAT=1` | Copy dimension-zero F32/I32 concat rows in parallel when scalar strides are 4 bytes, destination rows are disjoint and destination storage does not overlap either input. |
| Build `HTP_F16_M_AWARE_VTCM=ON` | Enable shape-aware FP16 allocation in the v68 HMX DSP build; selected by `tools/build.py --backend hmx --m-aware-vtcm`. |
| `HTP_TRACE=1` / `HTP_CPU_TRACE=1` | Diagnose DSP matrix shapes and host CPU-node time. Clear both for throughput measurements. |

Keep environment choices fixed before loading a model and for its lifetime.
The splitter checks structural support during loading/scheduling and actual
RPCMEM mappings during execution. Packed weights never fall back to ordinary
CPU multiplication. Its flash-attention capability guard respects the v68
512-element head limit.

The output option preserves the original embedding/output layout. It adds
**272,269,312 bytes (259.66 MiB)** of persistent model-owned caches, split into
two 75,968-column buffers below the allocator's 256 MiB limit. This is allocated
cache capacity, not a measured RSS difference. Only the stated tied storage,
shape, F16 type, actual HTP layer buffers and absence of LoRA adapters are
accepted. Startup packing remains a cost. HMX rounds activations/output to
FP16; CPU FP32 logit equality is not assumed. Checked mapping retirement and
MAP error propagation must ship in both matching host and DSP libraries.

For the tested M32 configuration, build the HMX DSP with
`tools/build.py --backend hmx --m-aware-vtcm`, stage matching host/DSP libraries,
and build the library caller described in the batch example. The phone-local
capture function accepts the following explicit settings in addition to the
caller-supplied model, inference, SDK, adapter and fresh output paths:

```python
width=32,
tokens=12,
sample_ms=60000,
reuse_threadpool=True,
reuse_cpu_threadpool=True,
split_cpu_ops=True,
hybrid_threads=1,
offload_output=True,
fast_concat=True,
shape_aware_vtcm=True,
```

`shape_aware_vtcm` records a build choice; it does not change the DSP binary.
Use the same compiled libraries for the matched control, keeping only this
metadata setting and the same width/token count. Other controls default off.
Check activation markers, output IDs/bytes, observation coverage and cleanup
before comparing counters. For M1, the split/pool configuration without output
offload avoids the additional vocabulary cache and has a separate measured
result; the maximum-throughput M32 configuration need not be preferable for a
single stream.

## Validation

Portable regressions cover [threadpool](../experiments/utilization/threadpool/),
[split](../experiments/utilization/split/),
[output](../experiments/utilization/output/), and
[fast concat](../experiments/utilization/concat/).
The pool regression covers 600 fresh graphs, varying active/worker counts,
bitwise comparison, 30 abort/recovery cycles, immediate graph disposal and
teardown. Output checks independently validate 136,317,952 half words, actual
vocabulary halves, all half encodings, guards, concat ordering and mapping
failure retention. Fast-concat checks run 150 actual-executor graph cases,
including F32/I32, full vocabulary M1/M32, unequal halves, padded/permuted
rows, incompatible-stride and overlap fallbacks, and untouched guards. Host
tests do not replace physical DSP validation.

On the phone, the 27 original GEMM, four quantized-pipeline and nine attention
cases pass with the optional DSP layout. Both layouts additionally pass 28
F16 cases with M=1/31/32/33/64/65/128/129 and four rejection guards. Valid
outputs have zero reference error and identical cross-build digests. The host
layout planner checks 546 shape/capacity cases.

The packed-head and combined scheduling stacks complete M1/M32 inference
with expected tokens and physical head completions. Each passes two model
loads in one backend process, actual head execution, context-before-model
destruction, finite logits and identical cross-reload full-logit hashes. This
is a bounded lifecycle check, not a leak stress test.

Against the saved ordinary CPU F16 reference over 63 fixed positions, the
packed-head and combined stacks report mean KLD **0.00002**, RMS probability
difference **0.121%**, and top-token agreement **100%**. This fork stores scaled
uint16 log probabilities for that comparison. The fixture does not establish
accuracy for all prompts, models or long contexts.

A separate M32 test uses four different prompts, each repeated eight times,
and compares each sequence with its prompt decoded alone at width 1. **29 of
32 sequences match exactly** over eight generated tokens. Zero-based rows
6, 14 and 18 first diverge at token 8: the width-1 code-prompt reference emits
`*` (353), while M32 emits `^` (61). This failed comparison is retained.

The separate [pre-greedy diagnostic caller](../experiments/utilization/logits/)
records finite logits before sampling. Fast concat off/on match all **256
full-vocabulary F32 hashes**, output IDs and bytes. Disabling only the packed
output projection, with splitting and both pools still enabled, restores all
32 width-1 outputs. At token 8, CPU-head margins favoring 353 over 61 are
0.0086917877, 0.0124855042 and 0.0124855042 for rows 6/14/18. The packed head
returns **20.234375 for both candidates**, creating a zero-margin tie; greedy
selection chooses the lower ID 61. The packed-head width-1 case retains a
0.03125 margin and selects 353.

These controls support the inference that the packed FP16 projection path
erases narrow margins in this fixture. Final-output rounding alone does not
explain row 6: its CPU value for 353 rounds to 20.25 in F16, while the packed
head returns 20.234375. Activation rounding, HMX arithmetic and output conversion
were not isolated separately. Width-1/M32 logit hashes can differ even when
tokens agree. The same-prompt, reload and fixed-position checks therefore do
not establish exact greedy equivalence for arbitrary independent prompts.
Diagnostic scans add CPU work; their timings are excluded from throughput.

## Measurement method

The [phone-local Python example](../experiments/utilization/batch/phone_capture_case.py)
calls the `qcphoneperf` library inside Android. Its native adapter initializes
the local SDK with a null server configuration. A development computer builds,
stages and retrieves evidence; it does not collect runtime metrics. No new
profiling CLI or APK is introduced. Tested telemetry access requires root.

Fresh inference processes use four CPU workers, flash attention, context 128
per sequence, 12 greedy output tokens and explicitly ignored EOS. Primary
throughput is synchronized decode input tokens divided by backend decode wall
time. Eleven calls are timed; the first generated token comes from prefill.
Generation including sampling, prefill and process lifetime are recorded
separately. Separate matrix traces are cleared during throughput measurements.

Selected SysMon IDs are 4097 QDSP6 Load, 4182 QDSP Clock, 4352/4377 HVX
utilization/activity, 4480/4481 HMX utilization/activity, and 4521 HMX Clock.
Observed matching-timestamp ratios are `4480 ~= 100*4481/4097` and
`4352 ~= 100*4377/4097`. ID4521 returns zero: physical HMX frequency remains
unresolved. These metrics do not measure useful row occupancy or aggregate
NPU utilization.

Counter statistics span initialization, prefill, decode and observed idle.
Positive-sample means are conditional arithmetic means, not wall-time busy
fractions. Source-clock conversion is unresolved; callback receipt gaps can
include buffering. Coverage, API start/stop/close, invalid/dropped records,
callback release, binary/DSP hashes and thermal observations are retained.
Runs are serialized; temperature is not controlled.

## Recorded results

The final matched build has shape-aware DSP allocation in both configurations.
The optimized configuration also enables split CPU operations, both reused
pools, one hybrid worker, the packed output head and fast concat. R1 runs the
control before the optimized case; R2 reverses that order for M32. All eight
final cases cover their full process lifetime, return the seven requested
metrics with successful cleanup, and match the fixed comparison prompt's
token IDs and output bytes. The separate distinct-prompt failure described
above remains a numerical limitation.

| Real streams | Control R1 / R2, tokens/s | Optimized R1 / R2, tokens/s | Paired speedup |
| ---: | ---: | ---: | ---: |
| 1 | 4.5545 / 4.5326 | 5.1233 / 5.1379 | 1.12–1.13x |
| 32 | 10.1063 / 10.2410 | 23.4585 / 23.4296 | 2.29–2.32x |

Throughput counts decode input tokens over synchronized backend time. The M32
numbers are aggregate throughput across 32 useful independent streams, rather
than one stream producing 23 tokens/s. Two repetitions check the observed
gain; they do not characterize variance or thermal stability.

| M32 activity statistic, range across R1/R2 | Control | Optimized |
| --- | ---: | ---: |
| Positive-sample HMX mean, MCPS | 7.789–7.827 | 10.300–10.348 |
| Peak HMX activity, MCPS | 10.481–10.485 | 44.434–44.439 |
| Peak SDK HMX ratio, % | 0.7377–0.7380 | 3.1274–3.1278 |
| Peak SDK HVX ratio, % | 1.4887–1.4923 | 4.2871–4.2976 |

The separately traced output concat decreases from **800.9–821.2 ms** to
**39.2–48.1 ms**. These traces are separate from the untraced throughput runs.
The activity statistics span the whole capture and do not establish sustained
busy percentages, decode-only utilization or hardware row gating.

An earlier combined stack, before fast concat, improves matched M32 throughput
from 10.1369 to 15.0046 tokens/s. All earlier stages, negative results and
exclusions remain in the [evidence JSON](../results/UTILIZATION-20260930.json).
The [selected figure](../results/UTILIZATION-20260930.png) and
[PDF](../results/UTILIZATION-20260930.pdf) show the eight final cases.

Single-option head offload slows M1 from 4.5415 to 4.2658 tokens/s. Splitting
ordinary operations with reused pools gives approximately 5.0 tokens/s in
tested M1 cases. VTCM alone is approximately unchanged at M1; reusing only the
hybrid pool or reducing only its worker count does not help. Retain these
negative results when selecting a configuration.

The initial combined-M1 capture misses a 19.412-second tail and is
excluded from counter comparisons. An initial split-M32 observation overlaps
a deployment interval and is conservatively excluded from comparisons. The
failed initial loader gate is retained separately; corrected tied-storage
checks pass before timing runs.

The GPU is unused by this inference backend. No energy savings, peak TOPS or
per-row hardware gating is claimed. Work is not added merely to increase a
utilization number.
