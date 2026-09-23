# 补充测试说明

本文档说明补充压测的设计思路和分析重点。

## 测试背景

在完成基础性能测试后，发现以下待验证问题：

1. **T1配置长输出场景缺失**: 之前测试主要关注短输出（100-256 tokens），缺少生产环境常见的长输出场景（2048 tokens）
2. **49K vs 47K TTFT异常**: 观察到相近输入长度下TTFT（首字延迟）存在显著差异，疑似存在性能瓶颈

## 测试设计

### Phase 1: T1配置长输出测试

**测试用例：**
- Case 1: 32并发 + 16K输入 + 2048输出
- Case 2: 32并发 + 32K输入 + 2048输出

**测试目的：**
1. 验证中长上下文场景下的长输出生成能力
2. 评估32并发高负载下的吞吐量和稳定性
3. 对比16K vs 32K输入对2K输出性能的影响

**关键指标：**
- **Throughput**: 期望 > 0.15 req/s (32并发下)
- **Decode TPS**: 期望 ≥ 12 tok/s/req (考虑长输出的调度开销)
- **TTFT**: 16K场景 < 2s, 32K场景 < 4s
- **端到端延迟**: P99 < 30s

**异常判断：**
- 如果32K场景下TPS显著低于16K (下降>30%)，可能存在：
  - KV cache管理效率问题
  - Batch调度策略在长上下文下的退化
  - 内存带宽瓶颈

### Phase 2: 49K vs 47K TTFT对比

**测试用例：**
- 并发度: 1, 4, 8
- 输入长度: 47104 (46K), 49152 (48K)
- 输出长度: 100 (短输出，聚焦prefill性能)

**测试目的：**
1. 验证TTFT异常是否在不同并发度下都复现
2. 定位异常的触发条件（长度阈值 vs 并发相关）
3. 排查是否与特定硬件配置或内存对齐有关

**分析维度：**

| 并发度 | 分析重点 |
|--------|----------|
| 1 | 纯prefill性能，排除调度干扰 |
| 4 | 低并发batch效率 |
| 8 | 中等并发下的资源竞争 |

**预期模式：**

**正常情况** (性能差异<10%):
```
并发1: 47K TTFT=2.1s, 49K TTFT=2.3s  (差异9.5%)
并发4: 47K TTFT=2.5s, 49K TTFT=2.7s  (差异8.0%)
并发8: 47K TTFT=3.2s, 49K TTFT=3.5s  (差异9.4%)
```

**异常情况** (49K显著劣化):
```
并发1: 47K TTFT=2.1s, 49K TTFT=4.5s  (差异114% ❌)
并发4: 47K TTFT=2.5s, 49K TTFT=5.2s  (差异108% ❌)
并发8: 47K TTFT=3.2s, 49K TTFT=6.8s  (差异113% ❌)
```

**可能原因分析：**

1. **内存对齐问题**
   - 49K可能触发某个不对齐的内存分配边界
   - Gaudi HPU内存对齐要求: 128字节或更大
   - **验证方法**: 查看vLLM日志中的内存分配警告

2. **KV Cache分块策略**
   - vLLM可能使用固定大小的KV cache块（如16K块）
   - 49K = 3块 + 1K碎片, 47K = 2块 + 15K
   - 跨块访问可能引入额外开销
   - **验证方法**: 检查 `block_size` 配置

3. **Recipe Cache缺失**
   - Gaudi使用recipe cache优化重复计算图
   - 49K可能超出某个缓存阈值，导致重新编译
   - **验证方法**: 对比第一次vs后续请求的TTFT

4. **批处理调度策略**
   - 某些长度可能触发不同的调度路径
   - **验证方法**: 添加详细的调度日志

## 结果验证清单

运行 `supplement_perf_test.sh` 后，逐项检查：

### ✅ T1配置长输出验证

- [ ] 16K+2048场景下 TPS ≥ 12 tok/s/req
- [ ] 32K+2048场景下 TPS ≥ 10 tok/s/req
- [ ] 32并发下无OOM或超时错误
- [ ] P99延迟在可接受范围内

### ✅ TTFT异常分析

- [ ] 47K vs 49K 在所有并发度下的TTFT差异 < 20%
- [ ] 如果差异>20%, 记录：
  - [ ] 差异幅度是否随并发度增加
  - [ ] vLLM日志中是否有异常警告
  - [ ] recipe cache命中情况

### 📊 数据提取命令

```bash
# 提取所有TTFT数据
grep -A 10 "TTFT_compare" perf_logs/supplement_perf_*.log | grep "TTFT"

# 对比47K vs 49K的中位数
grep "parallel=1.*in=47104" -A 20 perf_logs/supplement_perf_*.log | grep "TTFT (median)"
grep "parallel=1.*in=49152" -A 20 perf_logs/supplement_perf_*.log | grep "TTFT (median)"

# 提取T1配置的TPS
grep "T1_long_output" -A 20 perf_logs/supplement_perf_*.log | grep "TPS"
```

## 后续优化方向

根据测试结果，可能的优化路径：

1. **如果T1长输出性能不达标**:
   - 调整 `max_num_seqs` 参数
   - 优化decode阶段的batch调度策略
   - 考虑使用continuous batching

2. **如果49K TTFT异常**:
   - 调整KV cache块大小 (`block_size`)
   - 优化内存分配对齐
   - 检查是否需要增加recipe cache容量

3. **通用优化**:
   - 启用FlashAttention (如支持)
   - 调整 `gpu_memory_utilization`
   - 使用更激进的并行策略 (tensor/pipeline parallel)

## 参考配置

vLLM启动参数建议（根据测试结果调整）：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen/Qwen3.5-122B-A10B-FP8 \
    --tensor-parallel-size 8 \
    --max-model-len 65536 \
    --block-size 16 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --trust-remote-code
```

## 联系与反馈

如发现其他性能异常或需要进一步测试场景，请提交Issue。
