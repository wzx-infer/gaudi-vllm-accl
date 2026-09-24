#!/bin/bash
# Start vLLM server for vision-to-text inference on devices 4-7

export HABANA_VISIBLE_DEVICES="4,5,6,7"

MODEL_PATH="/data/models/Qwen/Qwen3.8-27B"
PORT=8001

echo "Starting Qwen3.8-27B vision server on devices 4-7..."
echo "Port: $PORT"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port $PORT \
    --host 0.0.0.0
