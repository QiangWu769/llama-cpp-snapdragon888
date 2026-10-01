# Build and Reproduce

Run the following commands from the repository root. The code uses the build options already validated on the phone; this release replaces personal paths in the experimental scripts with parameters. The new wrapper scripts have passed syntax, command-generation, and local checks, but a complete cross-compilation has not yet been repeated with these wrappers. See [results](../results/README.md) for hardware validation and build artifact identities.

## 1. Environment

The tested build environment is Ubuntu 24.04 x86-64, Hexagon SDK **6.6.0.0**, Hexagon Tools **19.0.07**, and Android NDK **r26d**. Python 3, CMake, Ninja, Clang (for host tests), and adb are also required. Obtain the SDK, toolchains, and models separately; they are not included in this repository.

The tested device is a OnePlus 9 LE2115 / SM8350 running Android 14. Tests used a normal adb shell and an unsigned cDSP FastRPC session, without running `su` or modifying system/vendor. Whether other firmware allows the same session must be verified on the device.

```sh
git clone https://github.com/QiangWu769/llama-cpp-snapdragon888.git
cd llama-cpp-snapdragon888

export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.6.0.0
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07"
export ANDROID_NDK=/path/to/android-ndk-r26d
# Set this when multiple devices are connected; omit it for a single device.
export ANDROID_SERIAL=YOUR_DEVICE_SERIAL
adb devices
```

## 2. Build HMX, HVX, or CPU

```sh
python3 tools/build.py --backend hmx --jobs 6
# Print the build commands without executing them:
python3 tools/build.py --backend hmx --dry-run
```

The HMX build explicitly sets `DSP_VERSION=v68` and `HTP_USE_HMX=ON`, producing:

| Artifact | Directory |
| --- | --- |
| DSP skeleton | `htp-ops-lib/hexagon_ReleaseG_toolv19_v68_hmx/ship/` |
| ARM64 stub and `htp_v68_test` | `htp-ops-lib/android_ReleaseG_aarch64_hmx/ship/` |
| Android llama executables | `llama.cpp-npu/build-android/bin/` |
| Logs for individual build steps | `logs/` |

```sh
# Retained HVX reference backend: DSP and stub directories have no _hmx suffix.
python3 tools/build.py --backend hvx
# Separate CPU baseline: GGML_HTP=OFF, with output in build-android-cpu.
python3 tools/build.py --backend cpu
```

HMX and HVX share the same Android llama HTP transport and therefore use the same `build-android` directory. Their DSP skeletons are deployed to separate directories. The CPU build and shared libraries are separate; `-ngl 0` / `--device none` do not reliably disable the automatically registered HTP ACCEL backend in this fork.

## 3. Model Conversion

**Qwen2.5-0.5B** has been validated. The later
[Qwen2.5-1.5B experiment](LARGER-MODEL.md) uses mixed Q4_0+F16 storage,
`--m-aware-vtcm` and optional mapping settings; full F16 mapping did not
complete on this phone. Start with a directory containing its original Hugging Face weights. HTP requires the original project's 32×32 permuted weight layout; an arbitrary standard GGUF cannot be substituted directly. This port does not modify the converter or quantizer and does not add a GGUF layout marker.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r llama.cpp-npu/requirements/requirements-convert_hf_to_gguf.txt
mkdir -p models
export HF_MODEL=/path/to/Qwen2.5-0.5B

PYTHONPATH=llama.cpp-npu/gguf-py python3 llama.cpp-npu/extras/convert_hf_to_gguf_htp.py \
  --outfile models/qwen2.5-0.5b-f16-hmx.gguf --outtype f16 "$HF_MODEL"

# Create a separate GGUF with the standard layout for the CPU baseline.
PYTHONPATH=llama.cpp-npu/gguf-py python3 llama.cpp-npu/convert_hf_to_gguf.py \
  --outfile models/qwen2.5-0.5b-f16-cpu.gguf --outtype f16 "$HF_MODEL"
```

Optional mixed IQ4_NL/Q8_0 quantization uses the original fork's quantizer included in this repository. First build it for the Linux host:

```sh
cmake -S llama.cpp-npu -B llama.cpp-npu/build-host \
  -DGGML_HTP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build llama.cpp-npu/build-host --target llama-quantize -j 6
REPACK_FOR_HVX=1 llama.cpp-npu/build-host/bin/llama-quantize \
  models/qwen2.5-0.5b-f16-hmx.gguf models/qwen2.5-0.5b-iq4-q8-hmx.gguf IQ4_NL+Q8_0
```

HMX and HVX accept the same HTP packed files. The embedding and final output weights use the standard layout; the seven designated types of layer weights are permuted according to the original ABI. Within the HMX backend, HVX dequantizes weights to FP16 before passing them to HMX; this is not native INT4 matrix computation.

## 4. Deployment

```sh
python3 tools/deploy.py --backend hmx --model models/qwen2.5-0.5b-f16-hmx.gguf
# Add a second model; existing models with different names are not removed.
python3 tools/deploy.py --backend hmx --model models/qwen2.5-0.5b-iq4-q8-hmx.gguf
```

The default deployment directories are `/data/local/tmp/llama-v68-hmx`, `/data/local/tmp/llama-v68-hvx`, and `/data/local/tmp/llama-v68-cpu`. Use `--phone-dir` to change the path, and pass the same path when running subsequent commands. `--serial` overrides `ANDROID_SERIAL`. `--dry-run` checks local artifacts and prints adb commands without writing to the device.

The tool copies the ARM64 executables, `libllama.so`, `libggml*.so`, the HTP stub, and the NDK's ARM64 `libc++_shared.so`. The DSP skeleton goes in `dsp/`. **The DSP uses the phone firmware's C++ runtime**: do not also copy SDK 19's DSP `libc++.so.1` or `libc++abi.so.1`, because the tested firmware does not provide the `aligned_alloc` symbol they require. These DSP libraries are separate from the ARM64 `libc++_shared.so` required on the Android side.

## 5. On-Device Operator Tests and Generation

```sh
# Three groups containing 40 cases in total; each has a default 300-second timeout.
python3 tools/device.py --backend hmx -- ./htp_v68_test --hmx --full
python3 tools/device.py --backend hmx -- ./htp_v68_test --pipeline
python3 tools/device.py --backend hmx -- ./htp_v68_test --hmx-attention

python3 tools/device.py --backend hmx --trace -- ./llama-cli \
  -m qwen2.5-0.5b-f16-hmx.gguf -p 'The capital of France is' \
  -n 32 -c 128 -b 16 -ub 16 -t 4 -tb 4 -fa \
  --temp 0 --seed 1234 --no-warmup --no-display-prompt
```

The tests should report `V68_RPC_TEST PASS failures=0`, with 27, 4, and 9 cases respectively. For whole-model generation, you can substitute `qwen2.5-0.5b-iq4-q8-hmx.gguf`. `--trace` enables `HTP_TRACE=1`, recording successful request counts and latency observed by the host; it is not a CPU/GPU/NPU utilization counter. This backend does not use the GPU.

For HVX, use `--backend hvx` and run `./htp_v68_test` without the HMX options. Only `--dry-run` prints commands without running them; normal `device.py` execution runs the command and preserves its exit code. Its stdout can be redirected directly to a log or JSON file.

## 6. Performance and CPU Comparison

```sh
python3 tools/device.py --backend hmx -- ./llama-bench \
  -m qwen2.5-0.5b-f16-hmx.gguf -p 32 -n 8 -b 32 -ub 32 -t 4 -fa 1 -r 3 -o json

python3 tools/build.py --backend cpu
python3 tools/deploy.py --backend cpu --model models/qwen2.5-0.5b-f16-cpu.gguf
python3 tools/device.py --backend cpu -- ./llama-bench \
  -m qwen2.5-0.5b-f16-cpu.gguf -p 32 -n 8 -b 32 -ub 32 -t 4 -fa 1 -r 3 -o json
```

The benchmark keeps the default warmup and disables tracing. Compare runs using the same original model, thread counts, and sequence lengths, and record temperature, clock frequencies, and repetition counts. The existing short runs in this repository did not control temperature and must not be interpreted as peak hardware throughput.

For a small numerical comparison, use `llama-perplexity` with standard CPU F16 and the corresponding HTP F16 model. Save a CPU baseline for fixed text with `-c 128 -b 32 -ub 32 -t 4 -tb 4 -fa --chunks 1 --save-all-logits <absolute-path-on-phone>`, then use `--kl-divergence --kl-divergence-base <same-path>` on HMX to read the same tokens. The tested `llama-perplexity` does not accept the CLI-only `--no-warmup` flag; run it from a writable directory for its relative cache. The input text must encode to at least 256 tokens. This fork saves scaled uint16 log probabilities, not raw FP32 logits; one 128-token chunk compares 63 positions. See [results](../results/README.md) for the recorded token IDs and summaries.

## 7. Troubleshooting

- Initialization failure: check DSP session, power, VTCM, and HMX lock errors first. Verify firmware support for unsigned cDSP sessions and that `libhtp_ops.so` and the skeleton are a matching pair.
- Missing `aligned_alloc`: check whether SDK DSP C++ runtimes were copied into the deployment.
- Incorrect model output: use this fork's HTP converter. Quantization requires `REPACK_FOR_HVX=1`; the CPU baseline requires a standard GGUF.
- Packed matmul error: the code fails explicitly. Packed weights cannot be passed to standard CPU matmul as a fallback.
- Request timeout: `HTP_OP_TIMEOUT_MS` defaults to 120000 ms. Diagnose with `tools/device.py --trace -- env HTP_OP_TIMEOUT_MS=... ./llama-cli ...`; changing the timeout cannot fix a DSP deadlock or an unsupported operator.

See [TESTING.md](TESTING.md) for host tests that do not require the SDK.
