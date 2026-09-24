#!/bin/bash
# Start vLLM server for text-to-text inference on devices 0-3

export HABANA_VISIBLE_DEVICES="0,1,2,3"

MODEL_PATH="/data/models/Qwen/Qwen3.8-27B"
PORT=8000

echo "Starting Qwen3.8-27B text server on devices 0-3..."
echo "Port: $PORT"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port $PORT \
    --host 0.0.0.0
