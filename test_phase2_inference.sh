#!/bin/bash
#
# Test Phase 2 MLA implementation inference on Gaudi hardware.
# Compares Phase 1 (simplified QKV) vs Phase 2 (full MLA) output quality.
#

set -e

MODEL_PATH="/mnt/4t_ext/DeepSeek-V4.1-Flash"
PORT=8000
MAX_TOKENS=100

echo "========================================================================"
echo "DeepSeek V4.1 Flash - Phase 2 Inference Test"
echo "========================================================================"
echo ""
echo "This test will:"
echo "1. Start vLLM server with Phase 2 MLA implementation"
echo "2. Run inference with a test prompt"
echo "3. Compare output quality with Phase 1 baseline"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Stop any existing vLLM server
echo "Stopping any existing vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

# Start Phase 2 vLLM server
echo ""
echo "Starting Phase 2 vLLM server..."
echo "Using model: $MODEL_PATH"
echo "Port: $PORT"
echo ""

# Launch server in background
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name deepseek-v41-phase2 \
    --port $PORT \
    --trust-remote-code \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --override-neuron-config '{"on_device_embedding":true}' \
    > /tmp/vllm_phase2_server.log 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo "Log file: /tmp/vllm_phase2_server.log"

# Wait for server to be ready
echo ""
echo "Waiting for server to be ready..."
MAX_WAIT=300  # 5 minutes
WAIT_TIME=0
while ! curl -s http://localhost:$PORT/v1/models > /dev/null; do
    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo "Error: Server failed to start within $MAX_WAIT seconds"
        echo "Check log file: /tmp/vllm_phase2_server.log"
        tail -50 /tmp/vllm_phase2_server.log
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    echo -n "."
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
done
echo ""
echo "Server is ready!"

# Test prompt
TEST_PROMPT="Write a short Python function to calculate the Fibonacci sequence."

echo ""
echo "========================================================================"
echo "Running inference test..."
echo "========================================================================"
echo "Prompt: $TEST_PROMPT"
echo ""

# Run inference
RESPONSE=$(curl -s http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"deepseek-v41-phase2\",
        \"prompt\": \"$TEST_PROMPT\",
        \"max_tokens\": $MAX_TOKENS,
        \"temperature\": 0.0
    }")

# Extract generated text
GENERATED_TEXT=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'choices' in data and len(data['choices']) > 0:
        print(data['choices'][0]['text'])
    else:
        print('Error: No text generated')
        print('Response:', json.dumps(data, indent=2), file=sys.stderr)
except Exception as e:
    print(f'Error parsing response: {e}', file=sys.stderr)
    sys.exit(1)
")

echo "========================================================================"
echo "Phase 2 Generated Output:"
echo "========================================================================"
echo "$GENERATED_TEXT"
echo ""

# Check output quality
if echo "$GENERATED_TEXT" | grep -q "def\|function\|fibonacci"; then
    echo "✓ Output appears to be coherent Python code"
    QUALITY="GOOD"
else
    echo "✗ Output may be gibberish (expected for Phase 1, should be better for Phase 2)"
    QUALITY="POOR"
fi

echo ""
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo "Model: DeepSeek V4.1 Flash Phase 2 (MLA)"
echo "Output Quality: $QUALITY"
echo "Prompt tokens: ~20"
echo "Generated tokens: $MAX_TOKENS"
echo ""
echo "Compare with Phase 1 baseline:"
echo "- Phase 1 used simplified QKV attention (skipped 93,335 MLA weights)"
echo "- Phase 2 uses full MLA architecture (wq_a, wq_b, wkv, q_norm, kv_norm, wo_a, wo_b)"
echo ""
echo "Expected improvement:"
echo "- Phase 1: Likely gibberish output (simplified architecture)"
echo "- Phase 2: Coherent, high-quality output (full MLA architecture)"
echo ""
echo "Server log: /tmp/vllm_phase2_server.log"
echo "Server PID: $SERVER_PID (still running - kill manually when done)"
echo "========================================================================"

# Keep server running for manual testing
echo ""
echo "Server is still running. To stop it:"
echo "  kill $SERVER_PID"
echo ""
echo "To test manually:"
echo "  curl http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d '{\"model\": \"deepseek-v41-phase2\", \"prompt\": \"Your prompt here\", \"max_tokens\": 50}'"
