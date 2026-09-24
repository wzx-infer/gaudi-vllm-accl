#!/usr/bin/env python3
"""
Text-to-Text benchmark for Qwen3.8-27B on Gaudi2 (devices 0-3)
"""
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Set Gaudi devices 0-3 for text generation
os.environ["HABANA_VISIBLE_DEVICES"] = "0,1,2,3"

def run_text_benchmark():
    """Run text-to-text throughput benchmark"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not available, using placeholder mode")
        return generate_placeholder_results("text")

    model_path = "/data/models/Qwen/Qwen3.8-27B"

    print("=" * 80)
    print("Qwen3.8-27B Text-to-Text Benchmark")
    print(f"Devices: 0-3 (4 HPUs)")
    print(f"Model: {model_path}")
    print("=" * 80)

    # Initialize vLLM
    print("\n[1/4] Loading model...")
    start_load = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.2f}s")

    # Prepare test prompts
    print("\n[2/4] Preparing test prompts...")
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate Fibonacci numbers.",
        "What are the key differences between TCP and UDP?",
        "Describe the process of photosynthesis.",
    ] * 25  # 100 prompts total

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=256,
    )

    # Warmup
    print("\n[3/4] Running warmup...")
    _ = llm.generate(prompts[:5], sampling_params)

    # Benchmark
    print("\n[4/4] Running benchmark...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    num_requests = len(outputs)

    results = {
        "benchmark_type": "text-to-text",
        "model": "Qwen3.8-27B",
        "devices": "0-3",
        "num_hpus": 4,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "tensor_parallel_size": 4,
            "dtype": "bfloat16",
            "max_model_len": 4096,
        },
        "metrics": {
            "num_requests": num_requests,
            "total_tokens_generated": total_tokens,
            "total_time_seconds": round(total_time, 2),
            "load_time_seconds": round(load_time, 2),
            "throughput_tokens_per_sec": round(total_tokens / total_time, 2),
            "throughput_requests_per_sec": round(num_requests / total_time, 2),
            "avg_tokens_per_request": round(total_tokens / num_requests, 2),
            "latency_seconds_per_request": round(total_time / num_requests, 2),
        }
    }

    return results


def generate_placeholder_results(benchmark_type):
    """Generate placeholder results if vLLM is not available"""
    return {
        "benchmark_type": benchmark_type,
        "model": "Qwen3.8-27B",
        "status": "placeholder",
        "message": "vLLM not available, this is a placeholder result"
    }


def save_results(results):
    """Save benchmark results to JSON"""
    output_dir = Path("/data/benchmark_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"qwen_text_benchmark_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Results saved to:", output_file)
    print(f"{'=' * 80}\n")
    print(json.dumps(results, indent=2))

    return output_file


if __name__ == "__main__":
    results = run_text_benchmark()
    save_results(results)
