# Independent sequence decoding and phone-local profiling

This research example calls the existing Android llama library with one real token from each independent sequence in every decode batch. It also demonstrates calling the [qcphoneperf Python library](https://github.com/QiangWu769/qcphoneperf/tree/add-native-profiling-api/api) inside the phone's Python process. The inference caller and capture script do not install a collector command or an APK.

Each sequence owns its greedy sampler and KV sequence ID. The caller records every output token ID and the exact output bytes, so width 1 can be compared with every sequence at widths 4, 16, and 32. It sets the physical microbatch capacity to the requested width and reports the submitted decode batch histogram. That histogram establishes the llama API submission width; a separate backend trace is needed to establish an individual matrix operation's actual `M`.

## Build against the matching Android libraries

Build the repository's Android HMX backend first. Supply its build directory to this example; do not mix source headers and libraries from different revisions. From the repository root:

```sh
cmake -S experiments/utilization/batch -B build-batch-android \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_DIR=/path/to/matching/llama.cpp-npu/build-android
cmake --build build-batch-android -j4
```

`LLAMA_SOURCE_DIR` defaults to this repository's `llama.cpp-npu` directory and can be overridden for another matching checkout. CMake imports that build's `libllama.so`, `libggml-cpu.so`, and `libggml-base.so`; it does not rebuild the backend. The runtime also needs the backend's matching dependent libraries, DSP skeleton and resources, and a correctly converted HMX FP16 model. Stage these and `hmx-batch-decode` in a directory of your choice on the phone.

The example uses the existing llama library's static C++ runtime arrangement. A backend built with another runtime configuration needs a matching caller configuration.

## Call the capture function on the phone

Install or stage the qcphoneperf Python bindings and Android ARM64 adapter separately. Follow that library's setup instructions for the authorized Qualcomm SDK, its matching metadata database, handler directories, vendor dependencies, and caller identity. Proprietary SDK dependencies are external to this repository. Set Android linker and Python module paths before starting Python. Runtime collection executes on the phone, without a remote profiler endpoint.

Put `phone_capture_case.py` on the phone's Python module path, then call its function from your research application:

```python
from pathlib import Path
from phone_capture_case import CaptureConfig, capture_case

case = capture_case(CaptureConfig(
    directory=Path("/data/local/tmp/llama-hmx"),
    model=Path("/data/local/tmp/models/model-f16-hmx.gguf"),
    output_dir=Path("/data/local/tmp/captures/baseline-m4"),
    profiler_payload=Path("/data/local/tmp/qualcomm-profiler"),
    adapter_library=Path("/data/local/tmp/qcphoneperf/lib/libqcphoneperf.so"),
    width=4,
    tokens=12,
    sample_ms=60000,
    timeout_seconds=90,
))

if case["exit_status"] != 0:
    raise RuntimeError("Inspect the retained workload and profiling errors")
if not case["coverage"]["full_workload_lifetime_inside_observation"]:
    raise RuntimeError("Counter comparisons need a fully observed workload")
```

Every call requires a new output directory. Run cases sequentially after the previous profiling session has closed. `baseline_directory` optionally supplies fallback inference libraries; the selected `directory` takes precedence. `sdk_library` optionally overrides the payload's `libs/libQualcommProfilerApi.so`. All paths are supplied by the caller.

The controlled capture uses the prompt `The capital of France is`, independent greedy sampling, ignored EOS, 128 context slots per sequence, and four CPU threads. Defaults are 12 output tokens per sequence, a 1-second workload start delay, a 60-second observation, and a 90-second workload timeout. Width is bounded to 1–32, token count to 1–120 for this fixed prompt/context, and observation requests to 1–60000 ms. The native API applies its minimum observation duration. A workload may still extend beyond the actual observation; the returned coverage fields identify an unobserved prefix or tail.

The capture clears inherited hardware traces and experimental backend controls for control cases. The inference subprocess receives the selected inference library and DSP paths. Profiling retains the caller process's separately configured SDK environment.

## Explicit experiment controls

All controls below default off. They request behavior from a matching compiled backend; setting a field does not rebuild it.

| `CaptureConfig` field | Behavior and verification |
| --- | --- |
| `reuse_threadpool=True` | Requests `HTP_REUSE_THREADPOOL=1` for the HTP hybrid executor; records its persistent-pool runtime marker. |
| `reuse_cpu_threadpool=True` | Adds the caller's `--reuse-cpu-threadpool` argument. The caller creates an ordinary `ggml-cpu` pool and attaches it with `llama_attach_threadpool`; the context is destroyed before its pool. Records the attachment marker and pool thread count. |
| `offload_output=True` | Requests `HTP_OFFLOAD_OUTPUT=1`; requires the packed output-projection activation marker. |
| `split_cpu_ops=True` | Requests `HTP_SPLIT_CPU_OPS=1` for the selected backend. |
| `hybrid_threads=N` | Requests `HTP_HYBRID_THREADS=N`, with `N` in 1–8; the ordinary CPU executor retains four configured workers. |
| `shape_aware_vtcm=True` | Records that the selected DSP skeleton was built with the shape-aware layout. This metadata field does not change an installed binary. |
| `fast_concat=True` | Requests `HTP_FAST_CONCAT=1`; requires the actual `HTP: fast dim0 concat enabled` execution marker. |

The ordinary CPU pool and the HTP hybrid pool have separate lifecycles. Requested ordinary-pool, output-projection, and fast-concat activation without the required marker produces a failure result. A concat marker proves that an eligible fast path executed; it does not establish that every concat operation used that path.

## Retained data and interpretation

The default NPU IDs are 4097, 4182, 4352, 4377, 4480, 4481, and 4521. The capture checks that the installed SDK advertises every selected ID before starting the workload. Supply `metric_ids` for another selection of up to eight unique unsigned 32-bit IDs, and inspect the returned names and units. Advertisement alone does not guarantee a measurable or compatible counter combination.

Each call retains:

- `process.json`: configuration, Android process identity, selected executable/library/DSP skeleton SHA-256 hashes.
- `capabilities.json`, `capture.json`, and `capture-call.json`: raw catalog, returned records/statuses, observation bounds, and call timing.
- `benchmark-stdout.json` and `benchmark.json`: raw and parsed inference JSON; `workload-stderr.log` remains separate.
- `workload-status.json`, `case-summary.json`, and `close.json`: workload exit/timeout, runtime markers, coverage, and completed profiler close.

Initialization exceptions propagate to the caller. A completed call can return failure because collection, the workload, a required activation marker, or cleanup failed. Preserve per-domain status and errors; missing values must not be replaced with zeros. A zero is a reported sample only when the API says so.

Use `decode_input_tokens_per_second` as the primary synchronized backend decode throughput. It counts only inputs to timed `llama_decode` calls. `aggregate_output_tokens_per_second` includes completed output tokens and sampling wall time; its first token per sequence comes from the final prefill logits. The JSON separates prefill, generation wall, and backend decode durations and retains actual batch widths and every sequence's output.

`HMX active` ID 4481 is active MCPS, rather than a row-occupancy measurement. SDK percentage metrics and their source timestamps must retain their original semantics. Comparing a same-timestamp active/load ratio does not establish wall-time engine utilization, and peak samples from a capture including initialization and prefill do not describe steady decoding alone. Check token IDs, output bytes, API status, coverage, and provenance before interpreting a throughput or counter comparison.
