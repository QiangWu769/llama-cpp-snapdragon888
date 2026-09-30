# 骁龙 888 / Hexagon v68 移植说明

本文说明相对固定的 haozixu 上游版本改了什么，以及这些改动解决了什么问题。上游来源见 [UPSTREAM.json](../UPSTREAM.json)；结果汇总与性能限制见 [仓库首页](../README.md)，构建步骤见 [BUILD.md](BUILD.md)。

本次实现保留上游 llama.cpp HTP 主机接口、FastRPC 通道、GGUF 转换和主要矩阵分块组织，增加 v68 HMX 路径，并保留独立 HVX 参考实现。已在 Snapdragon 888 的真实 cDSP 上完成算子及 Qwen2.5-0.5B 全模型验证。这里的 HMX 是矩阵单元；HVX 是向量单元，两者不能混称为相同的执行路径。

## 1. 硬件与固件适配

### 1.1 HMX 输出：从新式转换指令改为旧式 `after.hf`

上游 [hmx_utils.h](../htp-ops-lib/include/dsp/hmx_utils.h) 使用较新架构的 accumulator 转换、写回序列。v68 分支使用旧式写回，并明确清除 accumulator：

```text
原路径：
  cvt.hf = acc(2)
  mxmem(out, 0) = cvt

v68 路径：
  mxmem(out, 2047):after.hf = acc
  mxclracc.hf
```

这里的 `2047` 与旧式写回组合经过真机验证。多输出 tile、正负值、长 K 以及流水线测试用于检查布局、输出范围和跨 tile accumulator 残留。不能据此推导该控制参数所有位的完整 ISA 语义，也不能把新版 HMX 手册中的转换字段直接套到 v68。

输入继续使用上游 32×32 FP16 tile 布局。单个 tile 中逻辑 `(row, col)` 的半精度元素位置是 `(row / 2) * 64 + col * 2 + row % 2`。没有为 888 引入另一套模型文件布局。

### 1.2 `bias = mxmem` 的全零初始化不是数值 scale

上游通过 `bias = mxmem2(...)` 装载新格式控制/scale 数据；v68 改为 `bias = mxmem(...)`，将传入的旧式控制区清零。代码保留原辅助函数名 `hmx_init_column_scales`，但 v68 分支明确忽略其 `v_scale` 参数。

**不能把旧式 bias 区的低 16 位解释为普通 FP16 乘数。** 在被测 v68 上，沿用上游 `0x3c00` 初始化会改变输出行为；全零初始化通过了矩阵数值测试。此处记录的是已验证的初始化协议，不声称已经完整解析旧格式字段。

因此 attention 的 `1/sqrt(D)` 在 FP32 中显式应用，不再依赖 bias 区表达数值缩放。源码：[hmx_utils.h](../htp-ops-lib/include/dsp/hmx_utils.h)、[flash_attn.c](../htp-ops-lib/src/dsp/ops/flash_attn.c)。

### 1.3 使用固件提供的旧版 exclusive HAP 锁

原路径使用 `HAP_compute_res_hmx_lock2(..., HAP_COMPUTE_RES_HMX_SHARED)`。被测固件采用旧版 `HAP_compute_res_hmx_lock` / `HAP_compute_res_hmx_unlock`，因此 v68 在真正发出矩阵指令的 worker 上取得、释放 exclusive 所有权，而非仅在创建 worker pool 的线程上初始化锁。

锁失败会立即终止 DSP 执行；HMX 所有权切换前后加入编译器及 DSP 内存屏障。资源申请或 worker pool 初始化失败会向上传播并清理已申请资源。源码：[hmx_mgr.c](../htp-ops-lib/src/dsp/hmx_mgr.c)、[worker_pool.c](../htp-ops-lib/src/dsp/worker_pool.c)、[power.c](../htp-ops-lib/src/dsp/power.c)。

### 1.4 适配实测 4 MiB VTCM

上游矩阵路径以多个 1 MiB 区域组织 activation、weight、output 和双缓冲 scratch。被测 888 返回 4 MiB VTCM，部分原布局无法容纳。v68 把各基础工作区改为 **512 KiB**；最大的相应布局为六个区域加 identity tile 和控制区，约 3 MiB，且执行前按实际可用空间检查边界。

这会改变 M/N 分块大小，而不是把所有矩阵乘法的 K 都切成 512。普通路径仍在 accumulator 中完成完整 K；量化 output-stationary 分支另有原本就存在的 **512 个 K 元素**分块，两个“512”不是同一件事。

同时增加 DMA descriptor 清零和对齐、DMA 完成后的可见性屏障、worker 交接屏障、维度/字节数检查、奇数行补零。源码：[mat_mul.c](../htp-ops-lib/src/dsp/ops/mat_mul.c)、[vtcm_mgr.cc](../htp-ops-lib/src/dsp/vtcm_mgr.cc)。容量选择属于本次硬件适配，越界与同步检查也提升通用健壮性。

### 1.5 构建开关与固件运行库

[CMakeLists.txt](../htp-ops-lib/CMakeLists.txt) 增加 `HTP_USE_HMX`：v68 默认 OFF；显式 ON 时定义 `HTP_HMX_V68`、启用 HMX 编译选项并选择 v68 HMX 分支。OFF 选择新增的 [mat_mul_v68.c](../htp-ops-lib/src/dsp/ops/mat_mul_v68.c) 和 [flash_attn_v68.c](../htp-ops-lib/src/dsp/ops/flash_attn_v68.c)，作为独立 HVX 参考实现。非 v68 构建保留原架构分支，但本次未重新验证其他设备。

被测固件的 DSP C++ 运行库也影响加载：SDK 19 附带的 DSP `libc++.so.1` / `libc++abi.so.1` 引用了设备未提供的 `aligned_alloc`。部署使用设备固件的 DSP 运行库；Android ARM64 的 `libc++_shared.so` 是另一套运行库，仍需与主机端可执行文件一起部署。参见 [构建与部署说明](BUILD.md)，仓库不分发这些 SDK/固件二进制。

## 2. 数值正确性适配

### 2.1 FP16 / FP32 转换不经过有损 qfloat 中间值

原 HVX 转换链中存在通过 qf16/qf32 中间格式计算的路径。将 FP16 加零转换成 qf16，也可能丢失原 IEEE binary16 的有效位；随后转换回 FP32 无法恢复这些位。仅验证整数小矩阵不足以发现这类误差。

v68 的 [hvx_convert.h](../htp-ops-lib/include/dsp/hvx_convert.h) 使用整数 HVX 操作重建 IEEE 位模式：

- FP16 → FP32 保留符号、正规数、次正规数和无穷；NaN 安静化并保留相应 payload。
- FP32 → FP16 对普通有限数使用向量整数快速路径，按 round-to-nearest, ties-to-even 舍入；下溢、溢出和非有限值由整数标量路径补齐。
- 明确处理偶数/奇数 lane 的打包与解包顺序，避免把高低向量误当作连续的两半。

独立 host 模型检查全部 65,536 个 FP16 位模式，并覆盖 FP32 舍入边界和随机值。测试说明：[HVX_CONVERSION.md](../htp-ops-lib/tests/v68/HVX_CONVERSION.md)。这些 host 测试验证源代码算术和 lane 组织，真实 DSP 测试另行验证指令执行。

### 2.2 量化权重乘 scale：精确整数乘积后只舍入一次

量化路径沿用上游 Q8_0 / IQ4_NL 布局和码表，将量化值反量化为 FP16，再用 FP16 HMX。v68 的关键改动位于 [mat_mul.c](../htp-ops-lib/src/dsp/ops/mat_mul.c)：

1. 将有符号 INT8 值精确构造为 FP16 位模式。
2. 用整数 widening multiply 计算量化整数与 FP16 scale 有效数字的乘积，重建精确 FP32 乘积位模式。
3. 用上述 IEEE 转换按 nearest/ties-to-even 舍入到 FP16。
4. 显式配对 widening multiply 的偶/奇 lane 与对应 scale，修正不同 scale 相邻时暴露的错配。

这避免了 qfloat 乘法对恰好位于 FP16 中点附近的乘积引入额外低位扰动。默认实现使用向量整数路径，同时保留 `HTP_V68_DEQUANT_SCALAR_REFERENCE` 诊断分支。

独立测试遍历 **256 × 65,536 = 16,777,216** 个 INT8 × FP16 scale 位模式组合，并使用相邻不同 scale 检查 lane 顺序。见 [INTEGER_DEQUANT.md](../htp-ops-lib/tests/v68/INTEGER_DEQUANT.md)。完整模型已验证 IQ4_NL + Q8_0；没有完成 Q4_0 全模型验证，不能由公共反量化代码的存在推导出同等级覆盖。

### 2.3 独立 GEMM 参考必须反映 FP16 边界

[matmul_reference.h](../htp-ops-lib/tests/v68/matmul_reference.h) 不调用被测 DSP kernel，而是解码权重布局、构造输入并独立计算参考结果：activation FP32 先舍入到 FP16；量化值与存储的 FP16 scale 相乘后舍入到 FP16；点积使用高精度参考累加，输出再舍入 FP16。

量化 output-stationary 分支满足 `M >= 128 && K > N && N > 1024 && K < 16384` 时，每 512 个 K 元素输出一次 FP16，再通过 identity tile 载入旧部分和。因此该分支的参考也逐块执行 `half(previous + partial_sum)`，不能用完整 K 最后只舍入一次的参考替代。

真机测试阈值为 `0.01 + 0.001 * abs(reference)`，用于容纳舍入与消去附近的误差，同时检测布局错误；最终 31 个 GEMM/pipeline 用例实际最大绝对误差均为 **0**。这个结果针对声明的 FP16 参考，不等于与全 FP32 GEMM 逐位相同。

## 3. 为 v68 重写 HMX attention 路径

[flash_attn.c](../htp-ops-lib/src/dsp/ops/flash_attn.c) 的 `HTP_HMX_V68` 分支实现独立的分块路径，原 v73 分支保留在同文件另一条件分支中。

接口保持 Q/O 为 FP32、K/V 为 FP16，布局为 Q/O `[token][head][D]`、K/V `[token][kv_head][D]`；mask 为 FP16 additive mask，行跨度对齐到 64 个 KV 元素。公共函数历史上的 `__fp16 *` 参数声明不代表 Q/O 的实际存储类型。

每个任务处理一个 KV head 的最多 32 个 query/grouped-head 行：

1. Q 转 FP16，并把 Q/K 打包为 HMX tile；尾部补零。
2. HMX 执行 QK，结果在 FP32 中乘 `1/sqrt(D)` 并加入 mask。
3. 使用 FP32 running maximum、denominator 和输出合并状态进行稳定 online softmax。
4. P 先归一化，再转 FP16，与 FP16 V 执行 HMX PV；避免让 HMX 输出承载较长的未归一化和值。
5. 全部被 mask 的行输出零，尾部或完全不可见 key 不把无效 V 数据带入矩阵乘法。

每个 worker 的临时区按实际 D 和可用 VTCM 检查容量，HMX dot 在所属 worker 中取锁、装载零控制区并清 accumulator。已覆盖 Qwen 的 GQA=7；kernel 限制 `D <= 512`，当前主机 RPC 接口也不支持任意 stride、ALiBi 或 logit softcap。

真机 9 个用例对完整精度 softmax 参考的最大绝对误差为 **0.000559777**，测试阈值为 0.003。这反映 QK/PV 中的 FP16 边界，不要求与纯 FP32 attention 相同。直接算子测试支持 null mask；llama 主机分流当前要求符合其 RPC 映射条件的 mask，不能把直接 kernel 能力等同于所有 GGML 图都可下放。见 [ATTENTION.md](../htp-ops-lib/tests/v68/ATTENTION.md)。

## 4. 主机路由与 IPC：通用正确性修复

这些改动在移植中暴露并得到修复，但不是“骁龙 888 缺少某条指令”的问题。

### 4.1 保留上游模型格式，禁止错误的 CPU 回退

上游 [HTP converter](../llama.cpp-npu/extras/convert_hf_to_gguf_htp.py) 已对七类层权重做 32×32 permutation：attention Q/K/V/output 与 FFN up/down/gate。本次没有修改 converter、quantizer、模型结构或 tokenizer；`REPACK_FOR_HVX=1` 的量化 superblock ABI 也沿用上游。

[htp-ops.cc](../llama.cpp-npu/ggml/src/ggml-htp/htp-ops.cc) 精确识别这七类 `blk.<id>.*.weight`，包含 GGML scheduler 装饰后的副本名。它们已经重排，不能在 DSP 不支持或初始化失败时交给普通 CPU matmul 按行主序读取。因此不满足执行条件时立即报错，而不是静默产生错误结果。

`token_embd.weight` 与最终 `output.weight` 仍是普通布局。[ggml-htp.cc](../llama.cpp-npu/ggml/src/ggml-htp/ggml-htp.cc) 使这两类算子留在正常 CPU buffer 路径，其他合法普通算子仍可使用 CPU。没有在 GGUF 中新增自动格式标志，tensor 名称本身也不能证明文件确实经过 HTP 转换：用户必须遵循转换器契约。

主机还严格检查 matmul 的维度、类型、连续性、RPC buffer 和 attention 的固定布局/scale。RPC 未携带任意 stride/scale，不能接受超出其表达能力的 GGML 运算。RMSNorm 因 RPC 未传递 epsilon 保持 CPU 路径。`HTP_DISABLE_FLASH_ATTN=1` 可用于诊断 CPU attention。

### 4.2 错误必须从 DSP 传播回调用方

[ggml-htp.cc](../llama.cpp-npu/ggml/src/ggml-htp/ggml-htp.cc) 使用立即解析的动态库加载，检查符号、DSP session、backend 初始化和 message channel 结果。初始化直接检查生成的 FastRPC stub 返回值，避免原便捷包装函数丢弃错误。

[htp-ops.cc](../llama.cpp-npu/ggml/src/ggml-htp/htp-ops.cc) 串行保护共享 mapper/参数区/消息区，用 pending 状态初始化请求，并检查 map、unmap、compute 的完成状态。默认单请求超时为 120 秒，可用 `HTP_OP_TIMEOUT_MS` 调整；超时后终止，避免复用 DSP 可能仍在访问的缓冲区。

`HTP_TRACE=1` 输出成功请求的 opcode、累计次数、tensor 名和主机端耗时。它用于确认请求和定位问题，不是 HMX 指令计数、NPU 利用率或功耗测量。

### 4.3 生命周期、同步与 CPU 工作区

[commu.c](../htp-ops-lib/src/dsp/commu.c) 在启动接收线程前发布 channel 状态，使用 atomic 停止标志；空消息继续等待，关闭时先等待线程退出，再释放映射/栈。部分初始化失败路径也补充资源清理。[op_executor.cc](../htp-ops-lib/src/dsp/op_executor.cc) 检查未知 opcode，并按真实 mask 对齐跨度映射数据。

[htp-cpu-impl.c](../llama.cpp-npu/ggml/src/ggml-htp/htp-cpu-impl.c) 修正 CPU 工作区容量记录；后端注册线程设置接口，使 `-t` / `-tb` 可作用于混合后端 CPU 工作。这些修复有助于结果可重复和性能测量，但不能用它们宣称 HMX 硬件获得了某个加速比。

## 5. 验证覆盖及结论边界

真机入口为 [v68_test.c](../htp-ops-lib/src/host/v68_test.c)：

| 命令 | 覆盖 |
| --- | --- |
| `htp_v68_test --hmx --full` | 27 个 F16/Q8_0/IQ4_NL 用例，包括 M=1/5/32/33、K=896/4864、N=896，以及多个输出 tile 和非对称数据 |
| `htp_v68_test --pipeline` | Q8_0/IQ4_NL 各两例：M=128/K=896/N=896 与 M=129/K=1568/N=1056，覆盖四阶段 pipeline 和分 K 的 output-stationary 路径 |
| `htp_v68_test --hmx-attention` | 9 个 attention 用例，覆盖 GQA、additive/null mask、全屏蔽行和 KV 尾部 |
| `htp_v68_test` | 保留 HVX 路径的算子测试；应使用对应 HVX 构建执行 |

测试将独立参考计算放在普通 host 内存，避免在 uncached RPC buffer 上执行大规模 CPU 参考循环；输出使用哨兵和边界保护，RPC 有有界等待。host 自动化的范围和限制见 [TESTING.md](TESTING.md)。

最终 HMX 向量实现的 40 例真机算子测试、F16/混合量化各 32-token 完整模型运行，以及 F16 固定输入的概率对照均已通过，数值见 [首页结果表](../README.md#已验证结果)，脱敏记录见 [results/README.md](../results/README.md)。它们证明该实现可在已测 888 上执行与得到相符结果；尚不能推导更大模型、长上下文、所有固件或任意量化配置均可用。

当前 HMX F16 的短测 pp32/tg8 约为 27.57/4.42 tokens/s，低于同参数 CPU F16 的 42.58/8.88。该结果包含 CPU 算子、FastRPC、打包与反量化，且未控制温度或调优亲和性。因此当前交付应表述为 **完成 v68 HMX 功能移植与数值验证**，而非已经实现对 CPU 的端到端加速。
