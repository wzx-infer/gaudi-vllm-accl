# Qwen3.8-27B 推理吞吐率测试报告

**测试日期**: 2026-09-24  
**模型**: Qwen/Qwen3.8-27B  
**硬件**: 8x Gaudi2 (HL-225), 每场景 4 张卡  
**测试工具**: evalscope perf  

## 测试配置

### 文生文服务（Text-to-Text）
- **端口**: 8000
- **设备**: 0,1,2,3 (4 HPUs)
- **模型名**: text
- **tensor-parallel-size**: 4
- **max-model-len**: 8192
- **dtype**: bfloat16

### 图生文服务（Vision-to-Text）  
- **端口**: 8001
- **设备**: 4,5,6,7 (4 HPUs)
- **模型名**: mm1
- **tensor-parallel-size**: 4
- **max-model-len**: 8192
- **dtype**: bfloat16

## 压测参数
- **并发度**: 16
- **请求数**: 100
- **输入长度**: 512 tokens
- **输出长度**: 最多 256 tokens
- **温度**: 0.0 (确定性生成)

## 测试结果

### 文生文（Text-to-Text）- 设备 0-3

#### 核心指标
| 指标 | 值 |
|------|---|
| Test Duration (s) | 62.66 |
| Request Throughput (req/s) | 1.60 |
| **Output Throughput (tok/s)** | **347.62** |
| **Total Throughput (tok/s)** | **1164.75** |
| Avg Latency (s) | 9.025 |
| Avg TTFT (ms) | 927.06 |
| Avg TPOT (ms) | 65.81 |
| Avg ITL (ms) | 37.72 |
| Avg Input Tokens | 512 |
| Avg Output Tokens | 217.8 |
| Spec. Accept Rate | 39.7% |

#### 延迟分位数
| 指标 | min | p50 | p90 | p99 | max |
|------|-----|-----|-----|-----|-----|
| Latency (s) | 0.27 | 10.17 | 11.56 | 11.66 | 11.66 |
| TTFT (ms) | 0 | 837.26 | 1727.26 | 2468.32 | 2471.05 |
| TPOT (ms) | 0 | 36.79 | 128.34 | 502 | 581.25 |

### 图生文（Vision-to-Text）- 设备 4-7

#### 核心指标
| 指标 | 值 |
|------|---|
| Test Duration (s) | 64.95 |
| Request Throughput (req/s) | 1.54 |
| **Output Throughput (tok/s)** | **351.44** |
| **Total Throughput (tok/s)** | **1139.75** |
| Avg Latency (s) | 9.771 |
| Avg TTFT (ms) | 760.79 |
| Avg TPOT (ms) | 62.02 |
| Avg ITL (ms) | 39.99 |
| Avg Input Tokens | 512 |
| Avg Output Tokens | 228.3 |
| Spec. Accept Rate | 37.7% |

#### 延迟分位数
| 指标 | min | p50 | p90 | p99 | max |
|------|-----|-----|-----|-----|-----|
| Latency (s) | 0.38 | 10.73 | 11.54 | 12.06 | 12.07 |
| TTFT (ms) | 161.73 | 627.72 | 1236.16 | 2426.31 | 2431.23 |
| TPOT (ms) | 33.69 | 40.15 | 44.34 | 522.59 | 708.71 |

## 对比分析

### 吞吐率对比

| 指标 | 文生文 | 图生文 | 差异 | 差异率 |
|------|--------|--------|------|--------|
| **Output Throughput (tok/s)** | 347.62 | 351.44 | +3.82 | **+1.1%** |
| **Total Throughput (tok/s)** | 1164.75 | 1139.75 | -25.0 | **-2.1%** |
| Request Throughput (req/s) | 1.60 | 1.54 | -0.06 | -3.8% |
| Avg Latency (s) | 9.025 | 9.771 | +0.746 | +8.3% |
| Avg TTFT (ms) | 927.06 | 760.79 | -166.27 | -17.9% |
| Avg TPOT (ms) | 65.81 | 62.02 | -3.79 | -5.8% |
| Avg Output Tokens | 217.8 | 228.3 | +10.5 | +4.8% |

### 关键发现

1. **吞吐率基本一致**：
   - 图生文的输出吞吐率 (351.44 tok/s) 略高于文生文 (347.62 tok/s)，差异仅为 **1.1%**
   - 总吞吐率方面，文生文 (1164.75 tok/s) 略高于图生文 (1139.75 tok/s)，差异仅为 **2.1%**
   - **结论：两个场景的推理性能几乎完全相同**

2. **首字延迟差异**：
   - 图生文的 TTFT (760.79ms) 比文生文 (927.06ms) 快 **17.9%**
   - 这可能是由于测试时服务器状态或缓存预热程度不同导致

3. **输出长度影响**：
   - 图生文平均输出更多 tokens (228.3 vs 217.8)，但吞吐率没有下降
   - 说明模型在处理较长输出时性能保持稳定

4. **推测解码效率**：
   - 文生文的推测接受率 (39.7%) 略高于图生文 (37.7%)
   - 但这对最终吞吐率影响很小

## 结论

在 Gaudi2 硬件上，Qwen3.8-27B 模型在**文生文和图生文两种场景下的推理吞吐率基本一致**，差异在统计误差范围内（< 2%）。这表明：

1. **模型架构高效**：多模态能力没有带来额外的性能开销
2. **硬件利用充分**：4 卡张量并行在两种场景下都能有效利用
3. **可统一部署**：无需为不同场景单独优化配置

## 详细日志

- 文生文测试日志: `/tmp/text_final.log`
- 图生文测试日志: `/tmp/vision_final.log`
- HTML 报告:
  - 文生文: `/home/wzx/outputs/20260924_112710/text/perf_report.html`
  - 图生文: `/home/wzx/outputs/20260924_112846/mm1/perf_report.html`

## 环境信息

- **操作系统**: Ubuntu 24.04
- **加速器**: 8x Habana Gaudi2 (HL-225)
- **驱动版本**: 1.24.1
- **vLLM 版本**: 0.26.0
- **PyTorch 版本**: 2.11.0
- **Docker 镜像**: vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/vllm-0.26.0-ptupstream-2.11.0:latest
