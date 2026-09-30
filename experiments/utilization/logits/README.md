# Optional pre-greedy logit diagnostics

This separate llama library caller retains the independent sequence IDs,
greedy samplers and batch behavior of the [measurement caller](../batch/).
The measurement caller and its source are unchanged. `--diagnostic-logits`
is disabled by default.

When enabled, each active sequence records the full vocabulary's little-endian
binary32 FNV-1a64 hash, top two token IDs and values, their gap, and the values
of tokens 353 and 61, immediately before sampling at every generation step.
Those two IDs are fixture-specific watch candidates, not a portable meaning
across tokenizers. Non-finite logits fail the run. Selected tokens must agree
with the independently scanned top1; the lowest ID wins an exact tie.

The diagnostic reads logits without changing them. Scanning the vocabulary
adds CPU work and can change cache state and timing. Use it for numerical
diagnosis; exclude all diagnostic timing fields from throughput comparisons.
Matching 64-bit hashes are useful comparison signatures, rather than a saved
full-logit array or a collision-free equality proof.

## Build and run

Supply an existing matching Android build and source tree. The CMake file has
no host, device, model or SDK paths embedded in it. It imports `libllama`,
`libggml-cpu` and `libggml-base` from that supplied build; do not mix versions.

```sh
cmake -S experiments/utilization/logits -B build/logits \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DLLAMA_SOURCE_DIR="$LLAMA_SOURCE_DIR" \
  -DLLAMA_BUILD_DIR="$LLAMA_BUILD_DIR"
cmake --build build/logits --target hmx-batch-decode-logits
```

Stage the separate executable beside a compatible payload. On Android, supply
the matching host/DSP library environment used by the measurement caller, keep
the model and feature flags fixed for each fresh process, and add
`--diagnostic-logits` to the otherwise identical arguments. For example:

```sh
./hmx-batch-decode-logits --model "$MODEL" --parallel 32 --tokens 8 \
  --context-per-sequence 128 --threads 4 --ignore-eos \
  --reuse-cpu-threadpool --prompts-file prompts.txt --diagnostic-logits \
  > diagnostics.json 2> diagnostics.log
```

Each prompt-file line identifies one sequence. Keep per-sequence KV IDs and
samplers independent. Use a separate width-1 reference for each different
prompt, and preserve stdout, stderr and exit status for failed comparisons.
Do not overwrite the measurement executable with this caller.

## Output and comparison

The top-level `diagnostic_logits` flag must be true. Each
`sequences[].logit_diagnostics[]` entry contains:

| Field | Meaning |
| --- | --- |
| `step` | Zero-based generated-token step. |
| `logits_index` | The requested row in the preceding decode output. |
| `finite` | True after every vocabulary value passes the finite check. |
| `f32_le_fnv1a64` | Hash of every vocabulary logit's binary32 encoding. |
| `top1_id`, `top1_value` | Independently scanned greedy winner and value. |
| `top2_id`, `top2_value`, `gap` | Next candidate and the margin between them. |
| `token353_value`, `token61_value` | The two watched candidate values. |

Compare a width-1 reference with all matching prompt rows in a width-32 run:

```sh
python3 compare_logits.py reference.json batch32.json > comparison.json
```

Compare corresponding sequence IDs from two same-width controls:

```sh
python3 compare_logits.py repeat-a.json repeat-b.json --by-sequence \
  > comparison.json
```

The comparison separates the first hash change from the first greedy-token
change and retains each step's actual values. Inspect margins and watched
candidates before attributing a token change to rounding. After token histories
diverge, subsequent logits also reflect different inputs. The documented
eight-token diagnosis is a bounded fixture, not a general accuracy result.
