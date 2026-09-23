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
LOG="$LOG_DIR/batch_perf_$(date +%Y%m%d_%H%M%S).log"

# 动态请求数：低并发达标统计显著性，高并发控制总时长
get_number() {
    local parallel=$1
    if [ "$parallel" -le 4 ]; then echo 16;
    elif [ "$parallel" -le 16 ]; then echo 32;
    else echo 64; fi
}

# warmup：长上下文需要更多预热以填充recipe cache
get_warmup() {
    local in_len=$1
    if [ "$in_len" -ge 32768 ]; then echo 8;
    elif [ "$in_len" -ge 8192 ]; then echo 6;
    else echo 4; fi
}

# ==================== 测试用例定义 ====================
# 格式: "type concurrency in_len out_len"
# type: decode_only | prefill_only | mixed | long_ctx
CASES=(
    # === Phase 1: Decode Only (验证纯生成能力是否达15 tok/s/req) ===
    "decode_only  1   32   256"
    "decode_only  8   32   256"
    "decode_only  16  32   256"
    "decode_only  32  32   256"

    # === Phase 2: Prefill Only (验证不同输入长度的首字处理能力) ===
    "prefill_only 1   3000   1"
    "prefill_only 4   3000   1"
    "prefill_only 1   8192   1"
    "prefill_only 4   8192   1"
    "prefill_only 1   49152  1"
    "prefill_only 4   49152  1"

    # === Phase 3: Mixed Baseline (3K输入，修正输出长度问题) ===
    "mixed 8   3000 100"
    "mixed 16  3000 100"
    "mixed 32  3000 100"

    # === Phase 4: Long Context 50K (核心验收场景) ===
    "long_ctx 1   49152 100"
    "long_ctx 2   49152 100"
    "long_ctx 4   49152 100"
    "long_ctx 8   49152 100"
    "long_ctx 16  49152 100"
    "long_ctx 32  49152 100"
)

# ==================== 执行循环 ====================
echo "[$(date '+%F %T')] Batch perf test started. Log: $LOG" | tee "$LOG"

for item in "${CASES[@]}"; do
    read -r TYPE PARALLEL IN_LEN OUT_LEN <<< "$item"
    NUMBER=$(get_number "$PARALLEL")
    WARMUP=$(get_warmup "$IN_LEN")

    # Prefill-only 模式特殊处理：输出固定1 token，不需要min-tokens
    MIN_TOKENS_ARG=""
    if [ "$TYPE" != "prefill_only" ]; then
        MIN_TOKENS_ARG="--min-tokens $OUT_LEN"
    fi

    HEADER="\n===== [$TYPE] parallel=$PARALLEL number=$NUMBER in=$IN_LEN out=$OUT_LEN warmup=$WARMUP ====="
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
        $MIN_TOKENS_ARG \
        --tokenizer-path "$TOKENIZER_PATH" \
        --api-key "$API_KEY" \
        --temperature 0.0 \
        --total-timeout 3600 \
        2>&1 | tee -a "$LOG"

    # 每个case之间短暂冷却，避免热累积影响下一个case
    sleep 5
done

echo -e "\n[$(date '+%F %T')] All cases completed. Log: $LOG" | tee -a "$LOG"
