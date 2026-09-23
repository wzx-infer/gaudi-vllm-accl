#!/bin/bash
set -euo pipefail

# ==================== 配置区 ====================
MODEL_NAME="Qwen3.5-122B-A10B-FP8"
MODEL_PATH="/models/Qwen/Qwen3.5-122B-A10B-FP8"
TOKENIZER_PATH="/models/Qwen/Qwen3.5-122B-A10B-FP8"
API_URL="http://172.17.0.1:8006/v1/completions"
API_KEY="dscct2026"
LOG_DIR="./perf_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/supplement_perf_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] Supplement perf test started. Log: $LOG" | tee "$LOG"

# ==================== 补充测试用例 ====================
# 本脚本专注于补充之前遗留的测试场景

# === Case 1-2: T1配置 32并发 长输出测试 ===
# 目的：验证16K/32K输入下，2048 token长输出的性能表现
echo -e "\n========================================" | tee -a "$LOG"
echo "=== Phase 1: T1配置 长输出测试 ===" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for IN_LEN in 16384 32768; do
    PARALLEL=32
    OUT_LEN=2048
    NUMBER=64  # 32并发建议用64个请求保证统计显著性
    WARMUP=8   # 长上下文需要更多预热

    HEADER="\n===== [T1_long_output] parallel=$PARALLEL in=$IN_LEN out=$OUT_LEN number=$NUMBER warmup=$WARMUP ====="
    echo -e "$HEADER" | tee -a "$LOG"

    evalscope perf \
        --parallel "$PARALLEL" \
        --number "$NUMBER" \
        --warmup-num "$WARMUP" \
        --model "$MODEL_NAME" \
        --url "$API_URL" \
        --api openai \
        --dataset random \
        --min-prompt-length "$IN_LEN" \
        --max-prompt-length "$IN_LEN" \
        --max-tokens "$OUT_LEN" \
        --min-tokens "$OUT_LEN" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --api-key "$API_KEY" \
        --temperature 0.0 \
        --total-timeout 3600 \
        2>&1 | tee -a "$LOG"

    sleep 5
done

# === Case 3-5: 49K vs 47K TTFT对比测试 ===
# 目的：对比相近输入长度下TTFT的异常差异，定位性能瓶颈
echo -e "\n========================================" | tee -a "$LOG"
echo "=== Phase 2: 49K vs 47K TTFT 对比 ===" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# 使用多个并发度观察TTFT差异的一致性
for PARALLEL in 1 4 8; do
    for IN_LEN in 47104 49152; do  # 47K (47104 = 46*1024) vs 49K (49152 = 48*1024)
        OUT_LEN=100  # 短输出，聚焦TTFT指标

        if [ "$PARALLEL" -le 4 ]; then
            NUMBER=16
        else
            NUMBER=32
        fi
        WARMUP=8

        HEADER="\n===== [TTFT_compare] parallel=$PARALLEL in=$IN_LEN out=$OUT_LEN number=$NUMBER warmup=$WARMUP ====="
        echo -e "$HEADER" | tee -a "$LOG"

        evalscope perf \
            --parallel "$PARALLEL" \
            --number "$NUMBER" \
            --warmup-num "$WARMUP" \
            --model "$MODEL_NAME" \
            --url "$API_URL" \
            --api openai \
            --dataset random \
            --min-prompt-length "$IN_LEN" \
            --max-prompt-length "$IN_LEN" \
            --max-tokens "$OUT_LEN" \
            --min-tokens "$OUT_LEN" \
            --tokenizer-path "$TOKENIZER_PATH" \
            --api-key "$API_KEY" \
            --temperature 0.0 \
            --total-timeout 3600 \
            2>&1 | tee -a "$LOG"

        sleep 5
    done

    # 每组并发度对比完后，输出分隔便于日志分析
    echo -e "\n--- 并发度 $PARALLEL 对比完成 ---\n" | tee -a "$LOG"
done

echo -e "\n[$(date '+%F %T')] Supplement test completed. Log: $LOG" | tee -a "$LOG"
echo -e "\n建议分析重点：" | tee -a "$LOG"
echo "1. 16K vs 32K 在 2048 token输出下的吞吐量和延迟" | tee -a "$LOG"
echo "2. 47K vs 49K 的TTFT差异是否在所有并发度下都显著" | tee -a "$LOG"
echo "3. 如果49K TTFT异常，可能与内存对齐、KV cache分配策略相关" | tee -a "$LOG"
