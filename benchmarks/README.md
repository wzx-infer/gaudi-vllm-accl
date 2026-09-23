# Performance Benchmarks

本目录包含针对 Intel Gaudi 加速器上 vLLM 推理服务的性能测试脚本。

## 测试脚本

### 1. `full_perf_test.sh` - 完整基准测试

全面的性能基准测试，涵盖以下场景：

- **Decode Only**: 纯生成能力测试（短输入，长输出）
- **Prefill Only**: 首字延迟测试（不同长度输入，最小输出）
- **Mixed**: 混合负载测试（中等输入输出）
- **Long Context**: 长上下文测试（最高 50K tokens）

**使用方法：**

```bash
cd benchmarks
chmod +x full_perf_test.sh

# 修改脚本配置区的参数
# - MODEL_NAME: 模型名称
# - MODEL_PATH: 模型路径
# - API_URL: vLLM API 地址
# - API_KEY: API密钥（如需要）

# 运行完整测试（预计 30-60 分钟）
./full_perf_test.sh

# 查看日志
tail -f perf_logs/batch_perf_*.log
```

### 2. `supplement_perf_test.sh` - 补充测试

针对特定问题的补充压测：

- **T1配置长输出**: 32并发下 16K/32K 输入 + 2048 token 输出
- **TTFT异常分析**: 47K vs 49K 输入长度的首字延迟对比

**使用场景：**
- 已完成基础测试，需要针对特定问题深入分析
- 发现性能异常，需要对照实验验证

**使用方法：**

```bash
cd benchmarks
chmod +x supplement_perf_test.sh
./supplement_perf_test.sh

# 查看日志
tail -f perf_logs/supplement_perf_*.log
```

## 配置说明

所有脚本需要在运行前配置以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_NAME` | 模型标识符 | `Qwen3.5-122B-A10B-FP8` |
| `MODEL_PATH` | 模型文件路径 | `/models/Qwen/...` |
| `TOKENIZER_PATH` | Tokenizer路径 | 同 MODEL_PATH |
| `API_URL` | vLLM服务地址 | `http://172.17.0.1:8006/v1/completions` |
| `API_KEY` | API密钥 | `dscct2026` |
| `LOG_DIR` | 日志输出目录 | `./perf_logs` |

## 性能指标

测试脚本会输出以下关键指标：

- **TTFT (Time To First Token)**: 首字延迟，衡量prefill性能
- **TPS (Tokens Per Second)**: 每秒生成token数，衡量decode吞吐
- **Throughput**: 总吞吐量 (requests/sec 或 tokens/sec)
- **Latency (P50/P90/P99)**: 端到端延迟分位数

## 测试环境要求

- **依赖**: `evalscope` (ModelScope评测工具)
- **网络**: 需要访问 vLLM API 服务
- **时间**: 
  - 完整测试: 30-60 分钟
  - 补充测试: 10-15 分钟

## 结果分析

### 正常性能预期

- **Decode TPS**: ≥ 15 tok/s/req (单请求)
- **TTFT (3K输入)**: < 500ms (低并发)
- **TTFT (50K输入)**: < 5s (低并发)

### 异常模式识别

1. **TTFT突增**: 可能的KV cache分配瓶颈
2. **TPS下降**: batch调度效率问题
3. **高并发下崩溃**: 显存不足或配置问题

## 日志示例

```
===== [long_ctx] parallel=32 number=64 in=49152 out=100 warmup=8 =====
Model: Qwen3.5-122B-A10B-FP8
Requests: 64 (warmup: 8)
Duration: 245.32s
Throughput: 0.261 req/s
TTFT (median): 2.341s
TTFT (P99): 4.782s
TPS (median): 14.23 tok/s
Latency (P50): 8.234s
```

## 故障排查

### 常见问题

1. **连接失败**: 检查 `API_URL` 和网络连通性
2. **超时**: 增加 `--total-timeout` 参数
3. **内存不足**: 降低并发度或输入长度

### 调试建议

```bash
# 单独运行单个case测试连通性
evalscope perf \
    --parallel 1 \
    --number 1 \
    --model "Qwen3.5-122B-A10B-FP8" \
    --url "http://172.17.0.1:8006/v1/completions" \
    --api openai \
    --dataset random \
    --min-prompt-length 100 \
    --max-prompt-length 100 \
    --max-tokens 10 \
    --tokenizer-path "/models/Qwen/Qwen3.5-122B-A10B-FP8"
```

## 扩展测试

如需自定义测试场景，可修改脚本中的 `CASES` 数组：

```bash
CASES=(
    "type concurrency input_length output_length"
    "long_ctx 16  32768  512"  # 自定义: 16并发, 32K输入, 512输出
)
```

## 贡献

欢迎提交 Issue 报告性能异常或改进建议。
