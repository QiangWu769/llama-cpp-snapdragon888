# llama.cpp on Snapdragon 888 / Hexagon v68

将 [haozixu/llama.cpp-npu](https://github.com/haozixu/llama.cpp-npu) 与 [haozixu/htp-ops-lib](https://github.com/haozixu/htp-ops-lib) 的研究实现移植到 **骁龙 888（SM8350 / Hexagon v68）**。已在 OnePlus 9、Android 14 上使用真实 HMX 矩阵指令运行 Qwen2.5-0.5B，覆盖 F16 与 IQ4_NL + Q8_0 混合量化模型。

这是 **CPU + cDSP 的混合 LLM 推理后端**：HMX 执行矩阵乘法及 attention 的 QK/PV，HVX 和 DSP 标量代码负责转换、打包及部分计算；embedding、最终 output projection 和普通算子仍由 CPU 执行。GPU 未使用。它可以作为端侧 Agent 的推理基础，但仓库本身不提供手机 GUI Agent 应用。

**功能已打通，当前短测尚未快过 CPU F16。** 本仓库的重点是提供可检查的 v68 适配源码、数值验证和复现方法。

## 相对上游，改了什么才能在 888 上运行？

上游 README 的硬件要求为 Snapdragon 8 Gen 2 或更新设备，推荐构建 `DSP_ARCH=v73`。本次移植不只是把编译目标改成 `v68`：

| 部分 | 上游 v73 路径 | 本次 v68 适配与源码 |
| --- | --- | --- |
| HMX 输出 | `cvt.hf = acc` 后 `mxmem = cvt` | 使用旧式 `mxmem(..., 2047):after.hf = acc`，显式清 accumulator；[hmx_utils.h](htp-ops-lib/include/dsp/hmx_utils.h) |
| HMX bias / 转换控制 | `bias = mxmem2(...)`，初始化 FP16 scale | 使用 `bias = mxmem(...)` 和实测有效的全零旧式控制区；数值缩放单独计算，不能把旧字段当作 FP16 scale；[hmx_utils.h](htp-ops-lib/include/dsp/hmx_utils.h) |
| HMX 所有权 | `HAP_compute_res_hmx_lock2` shared API | 在真正执行矩阵指令的 worker 上获取、释放旧版 exclusive lock；[hmx_mgr.c](htp-ops-lib/src/dsp/hmx_mgr.c) |
| VTCM | 多个 1 MiB 工作区 | 针对实测 4 MiB VTCM 使用 512 KiB 工作区、容量检查和同步屏障；[mat_mul.c](htp-ops-lib/src/dsp/ops/mat_mul.c)、[vtcm_mgr.cc](htp-ops-lib/src/dsp/vtcm_mgr.cc) |
| 转换与反量化 | 使用 qfloat 转换/乘法链 | 使用整数 HVX 保持 IEEE FP16/FP32 转换与量化乘积的舍入行为，修正 lane 配对；[hvx_convert.h](htp-ops-lib/include/dsp/hvx_convert.h)、[mat_mul.c](htp-ops-lib/src/dsp/ops/mat_mul.c) |
| FlashAttention | 原 v73 实现 | 新增 v68 QK/PV HMX 分支，显式缩放、FP32 online softmax、GQA、mask 和尾块处理；[flash_attn.c](htp-ops-lib/src/dsp/ops/flash_attn.c) |
| 主机与通信 | 部分失败可继续 CPU 路径，请求缺少有界等待 | 精确分流普通/重排权重，重排矩阵失败立即终止；检查初始化、RPC 状态与超时；[ggml-htp](llama.cpp-npu/ggml/src/ggml-htp)、[commu.c](htp-ops-lib/src/dsp/commu.c) |
| 构建和验证 | 原 HMX 构建与测试 | 增加 HMX/HVX 选择、保留 v68 HVX 参考后端，新增独立数值参考和真机 RPC 测试；[CMakeLists.txt](htp-ops-lib/CMakeLists.txt)、[tests/v68](htp-ops-lib/tests/v68) |

其中，主机错误传播、布局分流、线程清理等属于通用正确性与健壮性修复，不是 v68 独有的 ISA 要求。详细解释见 **[888 移植说明](docs/888-PORT.zh-CN.md)**。

**未修改模型结构、tokenizer、上游 HTP converter 或 quantizer。** 沿用上游已有的七类层权重 32×32 重排，以及 `REPACK_FOR_HVX=1` 的量化打包格式。量化权重先反量化为 FP16 再送入 HMX，不是原生 INT4 HMX 计算。普通 GGUF 与 HTP 重排 GGUF 不能混用。

## 快速开始

完整环境、构建、转换和部署步骤见 **[BUILD.md](docs/BUILD.md)**；参数化入口是 [tools/build.py](tools/build.py)、[tools/deploy.py](tools/deploy.py) 和 [tools/device.py](tools/device.py)。SDK、NDK、模型权重和设备运行库需自行准备。

已测试工具链：Linux x86-64、Android NDK r26d、Hexagon SDK 6.6.0.0、Hexagon Tools 19.0.07。已测试设备：OnePlus 9 LE2115、Snapdragon 888、Android 14；这些运行通过普通 ADB shell 与 unsigned cDSP FastRPC 完成，未调用 `su` 或修改系统/vendor 分区。

| 模式 | 关键构建选择 | 用途 |
| --- | --- | --- |
| v68 HMX | 脚本设置 `DSP_VERSION=v68`、`HTP_USE_HMX=ON`；llama `GGML_HTP=ON` | 本次 HMX 实现 |
| v68 HVX | 脚本设置 `DSP_VERSION=v68`、`HTP_USE_HMX=OFF`；llama `GGML_HTP=ON` | 保留的正确性参考后端 |
| CPU | llama `GGML_HTP=OFF`，独立二进制与库目录 | 配合普通布局 GGUF 做 CPU 对照 |

`HTP_USE_HMX` 在 v68 默认关闭，`--backend hmx` 构建脚本会显式开启。按构建文档设置 `HEXAGON_SDK_ROOT`、`HEXAGON_TOOLS_ROOT`、`ANDROID_NDK`，并准备转换后的模型后，可运行：

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

HTP 在此 fork 中作为 ACCEL 自动注册，`-ngl 0` 或 `--device none` 不足以建立 CPU-only 对照。必须使用独立 `GGML_HTP=OFF` 构建和普通 GGUF。`HTP_TRACE=1` 是主机观察到的 DSP 请求数量、耗时和状态，**不是 NPU 利用率**。

## 已验证结果

最终向量实现通过 **40 个真实 FastRPC 算子用例**：

| 验证 | 结果 |
| --- | --- |
| 27 个 F16 / Q8_0 / IQ4_NL GEMM + 4 个 Q8_0 / IQ4_NL pipeline 用例 | 对包含实际 FP16 舍入边界的独立参考，全部最大绝对误差为 0 |
| 9 个 HMX attention 用例 | 对完整精度 softmax 参考，最大绝对误差 0.000559777 |
| Qwen2.5-0.5B F16 完整模型生成 32 tokens | 正常退出；5,376 次 F16 矩阵请求、768 次 attention 请求 |
| Qwen2.5-0.5B IQ4_NL + Q8_0 完整模型生成 32 tokens | 正常退出；3,840 次 IQ4_NL、1,536 次 Q8_0 矩阵请求、768 次 attention 请求 |
| F16 固定 128-token 输入，与 CPU 比较 63 个位置 | 平均 KLD 0.00001；概率 RMS 差 0.099%；top-token 一致率 100% |

GEMM 覆盖长 K、多个输出 tile、M 尾块和量化流水线；attention 覆盖 GQA=7、additive/null mask、全屏蔽行及尾块。概率对照使用本 fork 保存的缩放 uint16 log-probabilities，不是原始 FP32 logits。请求计数是对应完整测试进程的记录，不能直接理解为每生成 token 的算子数。

这些是小模型的功能与数值验证，不是通用模型精度结论。host 测试另外覆盖内存布局、IEEE 转换与全部 16,777,216 个 INT8 × FP16 scale 位模式组合；host 模型不能替代真实 DSP 验证。公开记录见 [results/README.md](results/README.md)，方法和复现入口见 [TESTING.md](docs/TESTING.md) 与 [算子测试说明](htp-ops-lib/tests/v68/README.md)。

## 当前性能

同一 Qwen2.5-0.5B，短序列 `llama-bench`：`-p 32 -n 8 -b 32 -ub 32 -t 4 -fa 1`，保留默认 warmup，`HTP_TRACE=0`。CPU/HMX 各 3 次重复，较慢的 HVX 参考为 1 次。

| 后端 / 模型 | Prefill pp32，tokens/s | Decode tg8，tokens/s |
| --- | ---: | ---: |
| CPU / F16 | 42.58 | 8.88 |
| 保留的 HVX / F16 | 0.42 | 0.10 |
| HMX / F16 | 27.57 | 4.42 |
| HMX / IQ4_NL + Q8_0 | 14.75 | 1.01 |

HMX 快于本仓库用于正确性验证的 HVX 参考实现，但 **尚未快过 CPU F16**。量化行精度不同，尚无对应的 CPU 量化基线。这些是完整混合后端的耗时，包含 CPU、通信、打包与反量化，不代表 HMX 峰值吞吐。测试未控制温度，也未调优 CPU ARM 编译选项或线程亲和性；尚无功耗或硬件利用率结论。

## 来源、目录与边界

| 目录 | 固定上游版本 |
| --- | --- |
| [llama.cpp-npu/](llama.cpp-npu) | [haozixu/llama.cpp-npu @ 57e34a34](https://github.com/haozixu/llama.cpp-npu/tree/57e34a34e0293894bd026280875703561c0106ef) |
| [htp-ops-lib/](htp-ops-lib) | [haozixu/htp-ops-lib @ 85eb88ed](https://github.com/haozixu/htp-ops-lib/tree/85eb88edcafd35afff1a43606a4c47eec9a0ca0b) |

原实现对应论文 [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/abs/2509.23324)。本仓库是独立移植；初始提交保留固定上游源码，后续提交展示适配差异，便于审查。版本信息见 [UPSTREAM.json](UPSTREAM.json)，归属及许可说明见 [NOTICE.md](NOTICE.md)；不对上游无顶层许可证的子树另行赋予统一许可证。

**[查看相对原始快照的全部适配差异](https://github.com/QiangWu769/llama-cpp-snapdragon888/compare/13a0dbb...main)**（源码、测试、脚本与说明）。本地也可运行 `git diff 13a0dbb HEAD -- htp-ops-lib llama.cpp-npu`，只查看两个原项目中的改动。

目前仅验证上述设备、固件、模型和短上下文。其他 Snapdragon 888 手机是否开放 unsigned cDSP、其固件运行库是否兼容，以及其他模型/长上下文是否正确，均需独立验证。本仓库不分发 Qualcomm SDK、QNN/固件库、模型权重或编译后的设备二进制。
