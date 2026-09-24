#!/bin/bash
# Parallel benchmark runner for Qwen3.8-27B
# Runs text-to-text on devices 0-3 and vision-to-text on devices 4-7 simultaneously

echo "=========================================="
echo "Qwen3.8-27B Parallel Benchmark Suite"
echo "=========================================="
echo "Text benchmark: devices 0-3 (4 HPUs)"
echo "Vision benchmark: devices 4-7 (4 HPUs)"
echo ""

# Create results directory
mkdir -p /data/benchmark_results

# Start text benchmark in background
echo "[$(date +%H:%M:%S)] Starting text-to-text benchmark..."
python3 /workspace/benchmarks/qwen_text_benchmark.py > /data/benchmark_results/text_benchmark.log 2>&1 &
TEXT_PID=$!

# Start vision benchmark in background
echo "[$(date +%H:%M:%S)] Starting vision-to-text benchmark..."
python3 /workspace/benchmarks/qwen_vision_benchmark.py > /data/benchmark_results/vision_benchmark.log 2>&1 &
VISION_PID=$!

# Monitor both processes
echo ""
echo "Both benchmarks running in parallel..."
echo "Text PID: $TEXT_PID"
echo "Vision PID: $VISION_PID"
echo ""

# Wait for text benchmark
echo "Waiting for text benchmark to complete..."
wait $TEXT_PID
TEXT_EXIT=$?

# Wait for vision benchmark
echo "Waiting for vision benchmark to complete..."
wait $VISION_PID
VISION_EXIT=$?

echo ""
echo "=========================================="
echo "Benchmark Results Summary"
echo "=========================================="
echo "Text benchmark exit code: $TEXT_EXIT"
echo "Vision benchmark exit code: $VISION_EXIT"
echo ""

# Display logs
if [ $TEXT_EXIT -eq 0 ]; then
    echo "--- Text Benchmark Log ---"
    tail -20 /data/benchmark_results/text_benchmark.log
    echo ""
fi

if [ $VISION_EXIT -eq 0 ]; then
    echo "--- Vision Benchmark Log ---"
    tail -20 /data/benchmark_results/vision_benchmark.log
    echo ""
fi

# List result files
echo "Result files:"
ls -lh /data/benchmark_results/qwen_*_benchmark_*.json 2>/dev/null || echo "No JSON results found"

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="

exit 0
