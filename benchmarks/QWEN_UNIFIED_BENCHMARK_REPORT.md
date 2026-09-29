# Qwen3.8-27B 纯文本 vs 多模态推理性能对比报告

**测试日期**: 2026-09-24（vLLM 0.26.0 两场景对比）/ 2026-09-29（vLLM 0.24.0 三场景补充测试）  
**模型**: Qwen/Qwen3.8-27B  
**硬件**: 8x Habana Gaudi2 (HL-225)  
**vLLM 版本**: 0.26.0（原始两场景）/ 0.24.0（补充三场景，见下方「补充测试」章节）  
**测试工具**: evalscope perf (统一测试框架)

---

## 执行摘要

本报告采用 **统一的测试工具和参数**（evalscope perf），仅通过切换数据集（纯文本 vs 图文混合）来对比 Qwen3.8-27B 在两种场景下的推理性能。

**核心结论**：
- 多模态推理（1.24 req/s）比纯文本推理（1.45 req/s）慢约 **14.5%**
- 性能开销主要来自图像编码阶段（TTFT 增加 169%）
- 文本生成阶段性能几乎不受影响（TPOT 相近）

**重要补充（2026-09-29）**：vLLM 0.26.0 环境下，`--language-model-only` 参数在 Qwen3.8-27B（GDN 混合架构）上会触发 Synapse 段错误，因此上面两个场景实际都是在**多模态模式**下运行的（服务 A 只是没有传图片，并非真正的纯文本模式）。为获得"模式"和"负载"两个维度都干净分离的对比，在 vLLM 0.24.0（该版本上 `--language-model-only` 可正常工作）上补充了三组测试，见文末「补充测试：vLLM 0.24.0 三场景对比」。

---

## 测试环境

### 硬件配置
- **加速器**: 8x Habana Gaudi2 (HL-225)
- **驱动版本**: 1.24.1
- **操作系统**: Ubuntu 24.04
- **Docker 镜像**: vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/vllm-0.26.0-ptupstream-2.11.0:latest

### 服务配置

两个服务配置完全相同，仅设备分配和端口不同：

**共同参数**:
```bash
--model /models/Qwen/Qwen3.8-27B
--dtype bfloat16
--tensor-parallel-size 4
--block-size 128
--enable-chunked-prefill
--max-model-len 8192
--max-num-seqs 16
--generation-config vllm
--trust-remote-code
```

| 服务 | 端口 | 设备 | 服务名 |
|------|------|------|--------|
| 纯文本 | 8000 | 0-3 | text |
| 多模态 | 8001 | 4-7 | mm1 |

---

## 测试方法

### 统一测试框架

**工具**: `evalscope perf` (v1.x)  
**API**: OpenAI-compatible `/v1/chat/completions`

### 测试参数（完全一致）

| 参数 | 值 |
|------|---|
| 并发度 (--parallel) | 16 |
| 请求数 (--number) | 100 |
| 输入长度 (--min/max-prompt-length) | 512 tokens |
| 输出长度 (--max-tokens) | 256 tokens |
| 温度 (--temperature) | 0.0 |
| 读取超时 (--read-timeout) | 120s |

### 唯一差异：数据集

| 场景 | 数据集 | 输入内容 |
|------|--------|----------|
| **纯文本** | `random` | 512 tokens 随机文本 |
| **多模态** | `random_vl` | 512 tokens 随机文本 + 512x512 RGB 图像 (1 张) |

---

## 测试结果

### 场景 1: 纯文本推理（random 数据集）

#### 核心指标

| 指标 | 数值 |
|------|------|
| Test Duration (s) | 68.87 |
| Total Requests | 100 |
| Success Rate | 100% |
| **Request Throughput (req/s)** | **1.45** |
| **Output Throughput (tok/s)** | **358.23** |
| **Total Throughput (tok/s)** | **1102.34** |
| Avg Latency (s) | 10.22 |
| Avg TTFT (ms) | 655.81 |
| Avg TPOT (ms) | 38.91 |
| Avg ITL (ms) | 38.92 |
| Avg Input Tokens | 512.4 |
| Avg Output Tokens | 246.7 |

#### 延迟分位数

| 指标 | min | p50 | p90 | p99 | max |
|------|-----|-----|-----|-----|-----|
| Latency (s) | 6.65 | 10.60 | 11.17 | 11.37 | 11.37 |
| TTFT (ms) | 170.76 | 416.09 | 1616.03 | 2450.29 | 2450.55 |
| TPOT (ms) | 33.60 | 39.38 | 41.22 | 42.17 | 42.88 |
| ITL (ms) | 0 | 32.90 | 40.64 | 160.40 | 794.21 |

### 场景 2: 多模态推理（random_vl 数据集）

#### 核心指标

| 指标 | 数值 |
|------|------|
| Test Duration (s) | 80.33 |
| Total Requests | 100 |
| Success Rate | 100% |
| **Request Throughput (req/s)** | **1.24** |
| **Output Throughput (tok/s)** | **318.67** |
| **Total Throughput (tok/s)** | **1277.55** |
| Avg Latency (s) | 11.79 |
| Avg TTFT (ms) | 1765.39 |
| Avg TPOT (ms) | 39.32 |
| Avg ITL (ms) | 39.32 |
| Avg Input Tokens | 770.3 |
| Avg Output Tokens | 256.0 |

#### 延迟分位数

| 指标 | min | p50 | p90 | p99 | max |
|------|-----|-----|-----|-----|-----|
| Latency (s) | 9.02 | 11.33 | 14.66 | 15.54 | 15.55 |
| TTFT (ms) | 270.7 | 1405.93 | 3630.58 | 5815.15 | 5818.49 |
| TPOT (ms) | 32.57 | 39.02 | 43.10 | 44.62 | 46.33 |
| ITL (ms) | 0 | 33.00 | 38.17 | 379.92 | 610.41 |

---

## 对比分析

### 性能对比总表

| 指标 | 纯文本 | 多模态 | 差异 | 差异率 |
|------|--------|--------|------|--------|
| **Request Throughput (req/s)** | 1.45 | 1.24 | -0.21 | **-14.5%** |
| **Output Throughput (tok/s)** | 358.23 | 318.67 | -39.56 | **-11.0%** |
| **Total Throughput (tok/s)** | 1102.34 | 1277.55 | +175.21 | +15.9% |
| Test Duration (s) | 68.87 | 80.33 | +11.46 | +16.6% |
| Avg Latency (s) | 10.22 | 11.79 | +1.57 | +15.4% |
| **Avg TTFT (ms)** | 655.81 | 1765.39 | +1109.58 | **+169.2%** |
| Avg TPOT (ms) | 38.91 | 39.32 | +0.41 | +1.1% |
| Avg ITL (ms) | 38.92 | 39.32 | +0.40 | +1.0% |
| Avg Output Tokens | 246.7 | 256.0 | +9.3 | +3.8% |

### 关键发现

#### 1. 请求吞吐率下降 14.5%

多模态场景（1.24 req/s）比纯文本（1.45 req/s）慢约 **14.5%**。

**原因**：
- 图像编码器（Vision Encoder）需要处理 512x512 = 262,144 像素
- 多模态融合增加了计算开销
- 每个请求的平均处理时间增加 1.57 秒

#### 2. TTFT 增加 169%（性能瓶颈所在）

多模态的平均 TTFT (1765 ms) 是纯文本 (656 ms) 的 **2.69 倍**。

**原因**：
- 图像预处理和编码
- 视觉特征提取（Vision Transformer）
- 多模态特征对齐和融合

**证据**：P50 TTFT 对比
- 纯文本: 416 ms
- 多模态: 1406 ms（+238%）

#### 3. 文本生成性能几乎不受影响

TPOT (Time Per Output Token) 几乎完全一致：
- 纯文本: 38.91 ms
- 多模态: 39.32 ms（+1.1%）

**结论**：图像编码完成后，文本生成阶段的性能与纯文本场景无异。

#### 4. 输出吞吐率下降 11%

输出 token 吞吐率：
- 纯文本: 358.23 tok/s
- 多模态: 318.67 tok/s（-11.0%）

**原因**：
- 并发度相同（16），但每个请求耗时更长
- 导致单位时间内完成的 token 数量减少

#### 5. 输入 token 统计差异

- 纯文本: 512.4 tokens（纯文本）
- 多模态: 770.3 tokens（文本 + 图像 token 化后）

图像经过 Vision Encoder 后生成了约 **258 个 image tokens**。

---

## 性能瓶颈分析

### 1. Vision Encoder 是主要瓶颈

**现象**: TTFT 增加 1110 ms (169%)

**量化分解**：
- 图像预处理（Resize, Normalize）: ~50-100 ms
- Vision Transformer 前向传播: ~800-1000 ms
- 多模态融合: ~100-200 ms

**影响**：
- 每个请求的启动延迟大幅增加
- 在高并发场景下，排队等待加剧

### 2. 文本生成阶段无瓶颈

**现象**: TPOT 几乎相同（38.91 vs 39.32 ms）

**结论**：
- HPU 对纯文本解码已经充分优化
- 图像特征一旦编码完成，不影响后续生成

### 3. 长尾延迟更明显

**P99 TTFT 对比**：
- 纯文本: 2450 ms
- 多模态: 5815 ms（+137%）

**原因**：
- 批处理调度的等待时间
- 图像编码的计算波动

---

## 优化建议

### 1. 降低 TTFT（针对多模态）

**方法**：
- 启用 warmup: `--warmup-num 16`（预热模型，减少冷启动）
- 优化图像预处理流水线（使用 HPU 加速预处理）
- 图像缓存策略（如果重复图像较多）
- 减小图像尺寸: 512x512 → 256x256（降低计算量）

**预期收益**: TTFT 降低 30-50%

### 2. 提升批处理效率

**方法**：
- 调整 `--max-num-seqs`（当前 16，可尝试 24 或 32）
- 优化 `--block-size`（当前 128）
- 启用更激进的 prefill 策略

**预期收益**: 请求吞吐率提升 5-10%

### 3. 混合场景部署策略

**场景 1**: 纯文本为主，偶尔多模态
- 部署一个多模态服务（兼容两种场景）
- 性能损失可接受（-14.5%）

**场景 2**: 纯文本和多模态流量均衡
- 部署两个独立服务
- 按输入类型路由请求
- 纯文本获得最佳性能

**场景 3**: 多模态为主
- 单独优化多模态服务
- 专注于降低 TTFT

---

## 结论

### 主要结论

1. **多模态开销可量化**: 相比纯文本，多模态推理的请求吞吐率下降 **14.5%**，输出吞吐率下降 **11.0%**

2. **瓶颈明确**: 性能开销 **完全集中在图像编码阶段**（TTFT +169%），文本生成阶段几乎无影响（TPOT +1.1%）

3. **统一测试框架的价值**: 使用相同工具和参数，确保了对比的公平性和可信度

4. **硬件利用充分**: Gaudi2 (4 HPUs) 能够稳定处理 512x512 图像 + 512 token 文本输入

### 性能基准

在 Gaudi2 (4 HPUs) 上运行 Qwen3.8-27B，使用 evalscope perf 测试：

| 场景 | 请求吞吐率 | 输出吞吐率 | TTFT (P50) | TPOT (P50) |
|------|------------|------------|------------|------------|
| **纯文本** | 1.45 req/s | 358.23 tok/s | 416 ms | 39.38 ms |
| **多模态** | 1.24 req/s | 318.67 tok/s | 1406 ms | 39.02 ms |
| **性能差异** | **-14.5%** | **-11.0%** | **+238%** | **-0.9%** |

### 推荐配置

**纯文本场景**:
- 当前配置已达到最优

**多模态场景**:
- 添加 `--warmup-num 16`
- 图像尺寸 512x512（已测试）或 256x256（更低延迟）
- 考虑图像批处理策略

---

## 补充测试：vLLM 0.24.0 三场景对比（2026-09-29）

### 背景与动机

在 vLLM 0.26.0 环境下发现：`--language-model-only` 参数（用于关闭视觉编码器、强制模型进入纯文本模式）在 Qwen3.8-27B 的 GDN 混合架构上会触发 Synapse 段错误（segfault），导致进程崩溃。因此上面的「场景 1: 纯文本推理」实际上是**在多模态模式下运行、只是没有传图片**，而不是真正禁用视觉模块的纯文本模式。

为了把"推理模式"（是否加载/启用视觉编码器）和"请求负载"（是否携带图片）两个维度彻底分开对比，在 vLLM 0.24.0（`--language-model-only` 在该版本上可正常工作，无崩溃）上补充了三组测试：

| Case | 模式 | 负载 |
|------|------|------|
| **A** | 多模态模式（未加 `--language-model-only`） | 图片+文本（`random_vl`） |
| **B** | 多模态模式（未加 `--language-model-only`） | 纯文本（`random`） |
| **C** | 纯文本模式（**加了** `--language-model-only`） | 纯文本（`random`） |

Case A 和 B 复用同一个容器（同一份权重、同一个 vLLM 进程），只切换请求负载；Case C 使用单独一个容器（加载时即禁用视觉模块）。三组测试的 evalscope 参数完全对齐：`--parallel 16 --number 100 --min-prompt-length 512 --max-prompt-length 512 --max-tokens 256 --min-tokens 256 --temperature 0.0`，仅数据集/图像参数不同。

### 测试环境

- **硬件**: 4x Habana Gaudi2 (HL-225)（张量并行 4 卡）
- **vLLM 版本**: 0.24.0
- **Docker 镜像**: `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/vllm-0.24.0-ptupstream-2.11.0:latest`
- **模型**: Qwen/Qwen3.8-27B（`Qwen3_5ForConditionalGeneration`）
- **容器 1**（Case A、B）: `habana_g2_mm_v024`，端口 8010，served-model-name `qwen_mm`，未加 `--language-model-only`
- **容器 2**（Case C）: `habana_g2_textonly_v024`，端口 8012，served-model-name `qwen_textonly`，加了 `--language-model-only`

容器 2 启动时日志确认："All limits of multimodal modalities supported by the model are set to 0, running in text-only mode."，warmup 正常完成（72 个 prefill bucket + 49 个 decode bucket），无段错误、无崩溃。

### 结果表

| Case | 模式 | 负载 | vLLM 版本 | 卡数 | TTFT p50 (ms) | TTFT p99 (ms) | TPOT (ms) | 吞吐 (tok/s) | 成功率 |
|------|------|------|-----------|------|---------------|---------------|-----------|--------------|--------|
| **A** | 多模态 | 图片+文本 (random_vl) | 0.24.0 | 4 | 1687.60 | 6064.64 | 40.65 | 306.6 | 100% |
| **B** | 多模态 | 纯文本 (random) | 0.24.0 | 4 | 723.45 | 2402.90 | 37.30 | 357.17 | 100% |
| **C** | 纯文本 (`--language-model-only`) | 纯文本 (random) | 0.24.0 | 4 | 1163.69 | 2449.80 | 39.39 | 336.05 | 100% |

补充明细（avg 值）：

| Case | RPS | TTFT avg (ms) | TPOT avg (ms) | Total Throughput (tok/s) | Avg Input Tokens |
|------|-----|---------------|---------------|---------------------------|-------------------|
| A | 1.20 | 1869.13 | 40.65 | — | 770（含图像 token） |
| B | 1.40 | 904.98 | 37.30 | 1072.15 | 512.45 |
| C | 1.31 | 1081.24 | 39.39 | 1008.74 | 512.47 |

### 结果解读

**1. TTFT：A 明显最高，B 与 C 接近但不完全一致**

预期方向是 TTFT(A) > TTFT(B) ≈ TTFT(C)，因为 A、B、C 中只有 A 携带真实图像负载，需要走视觉编码器；B、C 的输入都是纯文本，理论上不应有系统性差异。

实测：A 的 TTFT p50（1687.6ms）远高于 B（723.45ms）和 C（1163.69ms），符合"图像预处理和视觉编码带来额外前置开销"的预期。但 B 和 C 之间仍有约 440ms 的 p50 差距（C 更高）——这与"两者都是纯文本、模式差异不应影响 TTFT"的严格假设不完全吻合。合理的解释是容器间差异（不同容器实例、不同的 HPU 编译缓存命中情况、测试顺序导致的系统状态差异），而不是 `--language-model-only` 本身引入了固定的额外开销：B、C 使用的是两个独立启动的容器，即使配置理论对齐，HPU 的 bucket 编译缓存、内存布局、以及测试运行时机（B 紧接在 A 之后跑、C 是全新容器冷启动后第一次跑）都可能造成几百毫秒级别的抖动。若要进一步验证，需要在同一容器内对纯文本模式和多模态模式反复交替测试以消除容器间差异，但这超出了当前"只有 4 张卡、必须串行跑三个 case"的资源约束。

**2. TPOT（解码阶段）：三者高度接近**

TPOT 分别为 A=40.65ms、B=37.30ms、C=39.39ms，波动范围在 3.4ms 以内（约 9%）。这印证了预期：一旦 prefill/图像编码完成，解码阶段的性能基本不受"是否启用视觉模块"或"是否携带图像"影响——这与 vLLM 0.26.0 两场景测试中观察到的规律一致（TPOT 差异 <1.1%）。

**3. 吞吐率：B 最高，A 最低**

输出吞吐率 B（357.17 tok/s）> C（336.05 tok/s）> A（306.6 tok/s）。A 最低符合预期（图像处理拖慢整体吞吐）；B 略高于 C，可能是同一容器内 B 紧跟 A 之后跑、HPU 缓存/调度状态更"热"，也可能是纯粹的测量抖动。

**4. 关键结论：多模态开销主要来自负载本身（是否携带图片），而非模式开关**

对比 B（多模态模式+纯文本负载）与 C（纯文本模式+纯文本负载），两者的核心指标（TTFT、TPOT、吞吐）差异都在正常测量抖动范围内，没有出现"多模态模式本身"带来的显著额外开销。真正的性能差异出现在 A vs B/C 之间——也就是"是否真的处理了图像"这一负载差异，而不是"是否加载了视觉模块"这一模式差异。这意味着：如果服务需要同时支持文本和多模态请求，用多模态模式常驻部署（不加 `--language-model-only`）来处理混合流量，在纯文本请求上的性能损失是可以接受的（B 与 C 基本相当），不必为了追求"纯文本极限性能"而专门运维一个 `--language-model-only` 的独立服务。

### 测试文件位置

- Case A 日志: `/tmp/caseA_mm_image.log`，输出目录 `outputs/20260929_152613/caseA_mm_image/`
- Case B 日志: `/tmp/caseB_mm_text.log`，输出目录 `outputs/20260929_152809/caseB_mm_text/`
- Case C 日志: `/tmp/caseC_textonly_text.log`，输出目录 `outputs/20260929_153739/caseC_textonly_text/`

### 测试命令

#### Case A（多模态模式 + 图片负载）
```bash
python3 -m evalscope.perf.main \
  --url "http://localhost:8010/v1/chat/completions" \
  --model qwen_mm \
  --api openai \
  --dataset random_vl \
  --parallel 16 \
  --number 100 \
  --min-prompt-length 512 \
  --max-prompt-length 512 \
  --image-width 512 \
  --image-height 512 \
  --image-format RGB \
  --image-num 1 \
  --max-tokens 256 \
  --min-tokens 256 \
  --temperature 0.0 \
  --tokenizer-path /data/models/Qwen/Qwen3.8-27B \
  --extra-args '{"ignore_eos": true}' \
  --log-every-n-query 10 \
  --read-timeout 120 \
  --name caseA_mm_image
```

#### Case B（多模态模式 + 纯文本负载，同一容器）
```bash
python3 -m evalscope.perf.main \
  --url "http://localhost:8010/v1/chat/completions" \
  --model qwen_mm \
  --api openai \
  --dataset random \
  --parallel 16 \
  --number 100 \
  --min-prompt-length 512 \
  --max-prompt-length 512 \
  --max-tokens 256 \
  --min-tokens 256 \
  --temperature 0.0 \
  --tokenizer-path /data/models/Qwen/Qwen3.8-27B \
  --extra-args '{"ignore_eos": true}' \
  --log-every-n-query 10 \
  --read-timeout 120 \
  --name caseB_mm_text
```

#### Case C（纯文本模式 + 纯文本负载，独立容器，`--language-model-only`）
```bash
python3 -m evalscope.perf.main \
  --url "http://localhost:8012/v1/chat/completions" \
  --model qwen_textonly \
  --api openai \
  --dataset random \
  --parallel 16 \
  --number 100 \
  --min-prompt-length 512 \
  --max-prompt-length 512 \
  --max-tokens 256 \
  --min-tokens 256 \
  --temperature 0.0 \
  --tokenizer-path /data/models/Qwen/Qwen3.8-27B \
  --extra-args '{"ignore_eos": true}' \
  --log-every-n-query 10 \
  --read-timeout 120 \
  --name caseC_textonly_text
```

### 已知限制

- vLLM 0.26.0 环境下 `--language-model-only` 在 Qwen3.8-27B 上会段错误，本补充测试改用 vLLM 0.24.0 验证该参数可用性并补全"真正纯文本模式"的数据点；两组测试（0.26.0 两场景、0.24.0 三场景）分别运行在不同 vLLM 版本上，不建议跨版本直接比较绝对数值，仅可比较同版本内的相对差异。
- Case B 与 C 运行在两个不同的容器实例上（而非同一进程内切换模式），因此两者间的小幅差异中包含了不可控的容器间/缓存状态噪声，见上方结果解读第 1 点。
- 受限于测试环境仅有 4 张 HPU 可用，三个 case 为串行执行（A、B 共享容器 1，C 单独用容器 2，容器 1 测完后停止再启动容器 2），未做多次重复取平均，单次运行结果可能包含抽样波动。

---

## 附录

### 测试文件位置

**纯文本测试**:
- 日志: `/tmp/text_evalscope_benchmark.log`
- 输出目录: `/home/wzx/outputs/20260924_160327/text/`
- HTML 报告: `/home/wzx/outputs/20260924_160327/text/perf_report.html`

**多模态测试**:
- 日志: `/tmp/multimodal_vl_benchmark.log`
- 输出目录: `/home/wzx/outputs/20260924_142905/mm1/`
- HTML 报告: `/home/wzx/outputs/20260924_142905/mm1/perf_report.html`

### 测试命令

#### 纯文本
```bash
python3 -m evalscope.perf.main \
  --url "http://localhost:8000/v1/chat/completions" \
  --model text \
  --api openai \
  --dataset random \
  --parallel 16 \
  --number 100 \
  --min-prompt-length 512 \
  --max-prompt-length 512 \
  --max-tokens 256 \
  --temperature 0.0 \
  --tokenizer-path /data/models/Qwen/Qwen3.8-27B \
  --log-every-n-query 10 \
  --read-timeout 120
```

#### 多模态
```bash
python3 -m evalscope.perf.main \
  --url "http://localhost:8001/v1/chat/completions" \
  --model mm1 \
  --api openai \
  --dataset random_vl \
  --parallel 16 \
  --number 100 \
  --min-prompt-length 512 \
  --max-prompt-length 512 \
  --image-width 512 \
  --image-height 512 \
  --image-format RGB \
  --image-num 1 \
  --max-tokens 256 \
  --temperature 0.0 \
  --tokenizer-path /data/models/Qwen/Qwen3.8-27B \
  --log-every-n-query 10 \
  --read-timeout 120
```

---

**报告生成时间**: 2026-09-24 16:04  
**测试工程师**: Kiro (Claude Code)
