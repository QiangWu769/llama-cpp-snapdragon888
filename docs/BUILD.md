# 构建与复现

以下命令从仓库根目录执行。代码沿用已经真机验证的编译参数；本次发布将实验脚本中的个人路径改成了参数。新的通用包装脚本经过语法、命令生成和本地检查，尚未用这组包装脚本重新完成一次全量交叉编译。实际硬件验证和构建产物身份见 [results](../results/README.md)。

## 1. 环境

已测构建环境为 Ubuntu 24.04 x86-64、Hexagon SDK **6.6.0.0**、Hexagon Tools **19.0.07**、Android NDK **r26d**。另外需要 Python 3、CMake、Ninja、Clang（主机测试）和 adb。SDK、工具链和模型需自行取得，本仓库不包含它们。

已测设备为 OnePlus 9 LE2115 / SM8350 / Android 14。测试使用普通 adb shell 和 unsigned cDSP FastRPC session，没有执行 `su` 或修改 system/vendor；其他固件是否允许同样的会话，需要实际验证。

```sh
git clone https://github.com/QiangWu769/llama-cpp-snapdragon888.git
cd llama-cpp-snapdragon888

export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.6.0.0
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07"
export ANDROID_NDK=/path/to/android-ndk-r26d
# 多台设备连接时设置；只有一台时可省略。
export ANDROID_SERIAL=YOUR_DEVICE_SERIAL
adb devices
```

## 2. 编译 HMX、HVX 或 CPU

```sh
python3 tools/build.py --backend hmx --jobs 6
# 只查看将运行的编译命令：
python3 tools/build.py --backend hmx --dry-run
```

HMX 构建会显式设置 `DSP_VERSION=v68`、`HTP_USE_HMX=ON`，生成：

| 产物 | 目录 |
| --- | --- |
| DSP skeleton | `htp-ops-lib/hexagon_ReleaseG_toolv19_v68_hmx/ship/` |
| ARM64 stub 和 `htp_v68_test` | `htp-ops-lib/android_ReleaseG_aarch64_hmx/ship/` |
| Android llama 程序 | `llama.cpp-npu/build-android/bin/` |
| 分步构建日志 | `logs/` |

```sh
# 保留的正确性导向 HVX 后端：DSP 与 stub 目录没有 _hmx 后缀。
python3 tools/build.py --backend hvx
# 独立 CPU 基线：GGML_HTP=OFF，输出到 build-android-cpu。
python3 tools/build.py --backend cpu
```

HMX 和 HVX 共享同一套 Android llama HTP transport，因此使用同一个 `build-android`。不同的 DSP skeleton 部署到独立目录。CPU 构建和动态库独立；`-ngl 0` / `--device none` 不能在此 fork 中保证关闭自动注册的 HTP ACCEL 后端。

## 3. 模型转换

已验证 **Qwen2.5-0.5B**，需要 Hugging Face 原始权重目录。HTP 需要原项目的 32×32 权重置换布局，不能直接用任意普通 GGUF 替代。本次没有修改 converter 或 quantizer，也没有增加 GGUF 布局标记。

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r llama.cpp-npu/requirements/requirements-convert_hf_to_gguf.txt
mkdir -p models
export HF_MODEL=/path/to/Qwen2.5-0.5B

PYTHONPATH=llama.cpp-npu/gguf-py python3 llama.cpp-npu/extras/convert_hf_to_gguf_htp.py \
  --outfile models/qwen2.5-0.5b-f16-hmx.gguf --outtype f16 "$HF_MODEL"

# CPU 基线必须另做普通布局的 GGUF。
PYTHONPATH=llama.cpp-npu/gguf-py python3 llama.cpp-npu/convert_hf_to_gguf.py \
  --outfile models/qwen2.5-0.5b-f16-cpu.gguf --outtype f16 "$HF_MODEL"
```

可选的 IQ4_NL/Q8_0 混合量化使用本仓库内原 fork 的 quantizer。先编译 Linux 主机版本：

```sh
cmake -S llama.cpp-npu -B llama.cpp-npu/build-host \
  -DGGML_HTP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build llama.cpp-npu/build-host --target llama-quantize -j 6
REPACK_FOR_HVX=1 llama.cpp-npu/build-host/bin/llama-quantize \
  models/qwen2.5-0.5b-f16-hmx.gguf models/qwen2.5-0.5b-iq4-q8-hmx.gguf IQ4_NL+Q8_0
```

HMX 与 HVX 接受相同的 HTP packed 文件。embedding 和最终 output 使用普通布局，其余指定的七类层权重按原 ABI 置换。量化权重由 HVX 还原 FP16 后交给 HMX，不是原生 INT4 矩阵运算。

## 4. 部署

```sh
python3 tools/deploy.py --backend hmx --model models/qwen2.5-0.5b-f16-hmx.gguf
# 追加第二个模型；已有不同名称的模型不会被删除。
python3 tools/deploy.py --backend hmx --model models/qwen2.5-0.5b-iq4-q8-hmx.gguf
```

默认部署位置是 `/data/local/tmp/llama-v68-hmx`、`llama-v68-hvx`、`llama-v68-cpu`。可用 `--phone-dir` 改路径；随后执行命令也要传同一路径。`--serial` 覆盖 `ANDROID_SERIAL`。`--dry-run` 检查本地产物并打印 adb 命令，不写设备。

工具复制 ARM64 可执行程序、`libllama.so`、`libggml*.so`、HTP stub 和 NDK 的 ARM64 `libc++_shared.so`；DSP skeleton 放在 `dsp/`。**DSP 使用手机固件中的 C++ runtime**：不要额外复制 SDK 19 的 DSP `libc++.so.1` 或 `libc++abi.so.1`，已测固件不提供它们依赖的 `aligned_alloc`。这与 Android 侧必需的 ARM64 `libc++_shared.so` 是两回事。

## 5. 真机算子测试与生成

```sh
# 三组共 40 个案例，默认每组超时 300 秒。
python3 tools/device.py --backend hmx -- ./htp_v68_test --hmx --full
python3 tools/device.py --backend hmx -- ./htp_v68_test --pipeline
python3 tools/device.py --backend hmx -- ./htp_v68_test --hmx-attention

python3 tools/device.py --backend hmx --trace -- ./llama-cli \
  -m qwen2.5-0.5b-f16-hmx.gguf -p 'The capital of France is' \
  -n 32 -c 128 -b 16 -ub 16 -t 4 -tb 4 -fa \
  --temp 0 --seed 1234 --no-warmup --no-display-prompt
```

测试应显示 `V68_RPC_TEST PASS failures=0`，案例数分别为 27、4、9。完整模型可以换成 `qwen2.5-0.5b-iq4-q8-hmx.gguf`。`--trace` 打开 `HTP_TRACE=1`，记录成功的请求计数及主机观测延迟；它不是 CPU/GPU/NPU 利用率计数器。GPU 不参与此后端。

HVX 使用 `--backend hvx`，其算子测试为 `./htp_v68_test`（不传 HMX 选项）。只有 `--dry-run` 会仅打印命令；正常 `device.py` 会执行，并保留子进程退出码。其 stdout 可直接用于日志或 JSON 重定向。

## 6. 性能与 CPU 对照

```sh
python3 tools/device.py --backend hmx -- ./llama-bench \
  -m qwen2.5-0.5b-f16-hmx.gguf -p 32 -n 8 -b 32 -ub 32 -t 4 -fa 1 -r 3 -o json

python3 tools/build.py --backend cpu
python3 tools/deploy.py --backend cpu --model models/qwen2.5-0.5b-f16-cpu.gguf
python3 tools/device.py --backend cpu -- ./llama-bench \
  -m qwen2.5-0.5b-f16-cpu.gguf -p 32 -n 8 -b 32 -ub 32 -t 4 -fa 1 -r 3 -o json
```

benchmark 保留默认 warmup，关闭 trace。比较时使用同一原始模型、线程数和序列长度，并记录温度、频率及重复次数；仓库现有短跑数据没有控温，不能解释为峰值硬件吞吐。

小规模数值对照可用普通 CPU F16 和对应 HTP F16 的 `llama-perplexity`：CPU 对固定文本以 `-c 128 -b 32 -ub 32 -t 4 -tb 4 -fa --chunks 1 --save-all-logits <手机绝对路径>` 保存基线，HMX 使用 `--kl-divergence --kl-divergence-base <同一路径>` 读取相同 token。输入文本需至少能编码成 256 token。该 fork 保存的是缩放的 uint16 log probability，不是原始 FP32 logits；1 个 128-token chunk 比较 63 个位置。仓库记录的 token IDs 与摘要见 [results](../results/README.md)。

## 7. 排错

- 初始化失败：先看 DSP session、power、VTCM、HMX lock 错误；检查固件 unsigned cDSP 支持以及 `libhtp_ops.so` / skeleton 是否配套。
- `aligned_alloc` 找不到：检查是否混入 SDK 的 DSP C++ runtime。
- 模型结果错误：确认使用本 fork 的 HTP converter；量化必须启用 `REPACK_FOR_HVX=1`，CPU 必须使用普通 GGUF。
- packed matmul 报错：代码会显式失败，不能把 packed 权重交给普通 CPU matmul 当作回退。
- 请求超时：`HTP_OP_TIMEOUT_MS` 默认为 120000 ms。可用 `tools/device.py --trace -- env HTP_OP_TIMEOUT_MS=... ./llama-cli ...` 诊断；调整超时不能修复 DSP 死锁或不支持的算子。

无需 SDK 的主机测试见 [TESTING.md](TESTING.md)。
