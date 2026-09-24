#!/usr/bin/env python3
"""
Vision-to-Text (image generation) benchmark for Qwen3.8-27B on Gaudi2 (devices 4-7)
"""
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Set Gaudi devices 4-7 for vision tasks
os.environ["HABANA_VISIBLE_DEVICES"] = "4,5,6,7"

def run_vision_benchmark():
    """Run vision-to-text throughput benchmark"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not available, using placeholder mode")
        return generate_placeholder_results("vision")

    model_path = "/data/models/Qwen/Qwen3.8-27B"

    print("=" * 80)
    print("Qwen3.8-27B Vision-to-Text Benchmark")
    print(f"Devices: 4-7 (4 HPUs)")
    print(f"Model: {model_path}")
    print("=" * 80)

    # Initialize vLLM with multimodal support
    print("\n[1/4] Loading model with vision support...")
    start_load = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},  # Support one image per prompt
    )
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.2f}s")

    # Prepare test prompts with image placeholders
    print("\n[2/4] Preparing vision test prompts...")

    # For vision tasks, we'll use text prompts that would typically accompany images
    # In actual usage, these would include image data
    vision_prompts = [
        "<image>Describe what you see in this image in detail.",
        "<image>What is the main subject of this image?",
        "<image>Analyze the composition and colors in this image.",
        "<image>What emotions or mood does this image convey?",
    ] * 25  # 100 prompts total

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=256,
    )

    # Note: In production, you would load actual images here
    # For benchmark purposes, we're measuring the text generation capacity
    # when the model is configured for multimodal inputs

    # Warmup
    print("\n[3/4] Running warmup...")
    _ = llm.generate(vision_prompts[:5], sampling_params)

    # Benchmark
    print("\n[4/4] Running benchmark...")
    start_time = time.time()
    outputs = llm.generate(vision_prompts, sampling_params)
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    num_requests = len(outputs)

    results = {
        "benchmark_type": "vision-to-text",
        "model": "Qwen3.8-27B",
        "devices": "4-7",
        "num_hpus": 4,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "tensor_parallel_size": 4,
            "dtype": "bfloat16",
            "max_model_len": 4096,
            "multimodal": True,
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
        },
        "note": "Vision benchmark currently tests multimodal model capacity without actual image inputs"
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
    output_file = output_dir / f"qwen_vision_benchmark_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Results saved to:", output_file)
    print(f"{'=' * 80}\n")
    print(json.dumps(results, indent=2))

    return output_file


if __name__ == "__main__":
    results = run_vision_benchmark()
    save_results(results)
