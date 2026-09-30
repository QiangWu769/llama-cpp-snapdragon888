# Same-process packed output reload caller

This standalone library caller loads the same supported Qwen2 model twice in
one process. Each iteration creates a 128-token context with four threads and
flash attention, tokenizes `The capital of France is`, performs prompt decode
and one greedy decode, then frees the inference context before the model.
Backend initialization and shutdown surround both loads together.

Build against existing matching Android libraries in an independent directory:

```sh
cmake -S experiments/utilization/output/reload -B build-output-reload \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DLLAMA_SOURCE_DIR="$PWD/llama.cpp-npu" \
  -DLLAMA_BUILD_DIR=/path/to/matching/android-build
cmake --build build-output-reload --target hmx-output-reload -j4
```

On the phone, stage the caller beside the matching llama, GGML, HTP and DSP
companion libraries and use the established library-search environment. Invoke:

```sh
HTP_OFFLOAD_OUTPUT=1 HTP_TRACE=1 ./hmx-output-reload /path/to/supported-model.gguf
```

A public evaluation callback verifies that both reserved output projection
nodes completed with RPCMEM weight buffers during each decode. Confirm the
corresponding `HTP DSP completed` lines in stderr: the callback alone does not
inspect DSP mapping state. The callback synchronizes at selected graph nodes,
so this caller measures lifecycle correctness rather than representative
throughput.

JSON on stdout records finite logit ranges, greedy token IDs and FNV-1a hashes
of all vocabulary logits encoded as little-endian binary32. Exit 0 requires both
iterations, both head nodes, successful cleanup and identical logits across
reloads. Exit 2 reports differing logits; ordinary caught failures exit 1. A
backend abort or timeout can terminate before JSON; retain stderr and the
process status. Stage markers identify loading, decoding, context free and
model free. A successful separate-process run does not establish this
same-process reload property.
