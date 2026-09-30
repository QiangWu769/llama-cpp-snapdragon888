# Recorded Snapdragon 888 results

These are existing measurements from one OnePlus 9 LE2115, SM8350
(Snapdragon 888 / Hexagon v68), running Android 14. The model is
Qwen2.5-0.5B with 494,032,768 parameters. This directory contains text evidence;
it does not distribute model weights, SDK files or executable binaries.

The HMX backend is a hybrid: HMX performs supported layer matrix products and
attention QK/PV products on the cDSP; HVX and scalar DSP code perform conversion,
packing and other work. Embedding lookup, output projection and ordinary
unsupported operations run on the ARM CPU. The backend does not use the GPU.

## Identity and provenance

[identity.json](identity.json) records the device model, toolchain, upstream
source bases and the measured HMX DSP skeleton's SHA-256:

```text
d4f3d9516c00509300f903a0505301d77385d7829c9e1028804d398ccea46db5
```

The recorded host and deployed phone skeletons were identical. The
[ELF header](hmx-final-elf-header.txt) identifies Hexagon V68, and the
[instruction excerpt](hmx-final-instructions.txt) contains the legacy HMX
FP16 instructions from that skeleton. The identity record also retains the
exclusive HMX lock/unlock symbol evidence. These identify the measured build;
the instruction excerpt alone is not a runtime utilization measurement.

Toolchain: Hexagon SDK 6.6.0.0, Hexagon Tools 19.0.07 and Android NDK r26d.
The binary SHA identifies the measured artifact, not a promise that every
rebuild produces identical bytes.

The benchmark's `build_commit: "57e34a3"` is the **upstream llama.cpp-npu
base**, not the full port source revision. The port includes local changes
on top of the bases recorded in `identity.json` and the repository's
`UPSTREAM.json`. Do not use the benchmark field alone to identify this port.
The phone clock was inaccurate: raw `test_time` values are retained unchanged
and must not be treated as reliable measurement dates. `host_recorded_utc`
records identity collection time, not the start of every test.

Raw operator logs, ELF/instruction text and benchmark JSON retain their
original bytes; benchmark files were renamed from `.stdout` to `.json`.
Summary JSON omits command, deployment and path fields. Private host paths,
device serials and network addresses are not included.

## Forty real-phone operator cases

| Suite | Cases | Maximum absolute error | Raw log |
| --- | ---: | ---: | --- |
| F16 / Q8_0 / IQ4_NL matrix multiplication | 27 | 0 | [full](hmx-final-full.log) |
| Quantized pipeline and output-stationary matrix multiplication | 4 | 0 | [pipeline](hmx-final-pipeline.log) |
| HMX attention | 9 | 0.00055977702 | [attention](hmx-final-attention.log) |

All three processes exited successfully; the sanitized aggregate is
[hmx-final.summary.json](hmx-final.summary.json). Matrix references explicitly
round activations and dequantized weights to FP16, accumulate with an
independent reference, and round outputs to FP16. The output-stationary
reference additionally rounds between 512-element K chunks. Zero error here
means exact numerical agreement with those conversion boundaries on these
fixtures, not agreement with unrestricted FP32 matrix multiplication.

The full suite includes a small `(M,K,N)=(5,96,96)` case and combinations
of `M={1,5,32,33}`, `K={896,4864}`, `N=896`. Pipeline cases use
`(128,896,896)` and `(129,1568,1056)` for Q8_0 and IQ4_NL. Attention includes
GQA, odd query/KV lengths, null/additive masks, entirely masked rows and
poisoned padding. Its independent reference uses full-precision softmax;
the acceptance bound is 0.003 absolute error for the tested input range.
See the [operator test documentation](../htp-ops-lib/tests/v68/README.md)
for full shapes, input ranges, guards and acceptance rules.

Q4_0 is **not included** in these 40 HMX cases. The retained HVX path has
separate Q4_0 coverage; no full Q4_0 model result is claimed here. Per-case
`rpc_wall_ms` includes transport and operator overhead and is a diagnostic,
not isolated HMX throughput.

## End-to-end generation and a small numerical comparison

Both 32-token generation runs exited with status 0 using context 128,
batch/microbatch 16, four threads, flash attention, temperature 0, seed 1234
and no warmup. The prompt was `The capital of France is`.

| Model | Completed matrix requests | Completed attention requests | Summary |
| --- | --- | ---: | --- |
| F16 | 5,376 F16 | 768 | [F16 generation](qwen-f16-hmx-final32.summary.json) |
| IQ4_NL + Q8_0 | 3,840 IQ4_NL + 1,536 Q8_0 | 768 | [quantized generation](qwen-quant-hmx-final32.summary.json) |

These summaries preserve generated text and process timing. Text is model
output, not a factual-answer evaluation. Opcode counts record successful
DSP requests: `1=F16 matmul`, `3=Q8_0 matmul`, `4=IQ4_NL matmul`,
`5=attention`. Counts and host-observed latency do not measure HMX occupancy,
CPU/GPU/NPU utilization or energy consumption.

The [fixed-token comparison](hmx-f16-logprobs-c128-final.summary.json) compares
the HMX F16 model with the separately built, ordinary-layout CPU F16 model
over **63 positions in one 128-token input**. It records the exact input
token IDs and the reference file's hash. Reported mean KL divergence is
0.00001, RMS probability difference is 0.099%, and top-token agreement is
100%. These values have the tool's printed precision; printed zero
uncertainty is not evidence of zero underlying uncertainty. The reference
stores per-position scaled 16-bit log probabilities, not raw FP32 logits.
This small comparison is not general model-quality or long-context validation.

## Short end-to-end benchmark

Recorded `llama-bench` settings were `-p 32 -n 8 -b 32 -ub 32 -t 4 -fa 1`,
default warmup and `HTP_TRACE=0`. CPU and HMX use three repetitions each;
the slow retained HVX reference uses one.

| Backend / model | Prefill pp32, tokens/s | Decode tg8, tokens/s | Raw JSON |
| --- | ---: | ---: | --- |
| CPU / F16 | 42.58 | 8.88 | [CPU](bench-cpu-f16-final.json) |
| Retained HVX / F16 | 0.42 | 0.10 | [HVX](bench-hvx-f16-current.json) |
| HMX / F16 | 27.57 | 4.42 | [HMX F16](bench-hmx-f16-final.json) |
| HMX / IQ4_NL + Q8_0 | 14.75 | 1.01 | [HMX quantized](bench-hmx-quant-final.json) |

This HMX implementation is faster than the correctness-oriented HVX reference
but **does not outperform the CPU F16 baseline**. These are whole-backend
timings including ARM work, transport, conversion and dequantization. Runs
were short, sequential and not thermally controlled; thread affinity and ARM
CPU flags were not tuned. A single HVX sample cannot characterize variance,
even though its raw standard deviation is printed as zero. The mixed
quantized row changes model precision and has no matched CPU quantized
baseline, so it does not establish a quantized CPU-versus-HMX speedup.

The raw generic model label `qwen2 1B` refers to the 494,032,768-parameter
Qwen2.5-0.5B model recorded in those same files. `MyHTP` and
`Unknown Hexagon Processor` are generic backend labels; the device identity
is recorded separately. Generic fields such as `n_gpu_layers` remain in
the benchmark schema and do not demonstrate GPU execution. Likewise, the
HVX file's model filename contains `hmx` because both DSP implementations
consume the same packed weights; the filename does not select the backend.

These measurements cover this phone, firmware, model and tested shapes only.
They do not establish compatibility with every Snapdragon 888 device, other
models, long contexts or other Hexagon generations.
