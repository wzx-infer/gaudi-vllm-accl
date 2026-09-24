#!/usr/bin/env python3
"""Benchmark text-to-text throughput using vLLM OpenAI API"""
import time
import json
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8000/v1/completions"
NUM_REQUESTS = 100
MAX_TOKENS = 256

# Test prompts
PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
    "What are the benefits of microservices?",
    "Describe how neural networks work.",
] * 25

def send_request(prompt, request_id):
    """Send a single completion request"""
    start = time.time()
    try:
        response = requests.post(
            API_URL,
            json={
                "model": "Qwen3.8-27B",
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.8,
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        tokens = len(data["choices"][0]["text"].split())
        latency = time.time() - start

        return {
            "success": True,
            "request_id": request_id,
            "tokens": tokens,
            "latency": latency,
        }
    except Exception as e:
        return {
            "success": False,
            "request_id": request_id,
            "error": str(e),
        }

def main():
    print("=" * 80)
    print("Qwen3.8-27B Text-to-Text Benchmark")
    print(f"API: {API_URL}")
    print(f"Requests: {NUM_REQUESTS}")
    print("=" * 80)

    # Warmup
    print("\n[1/3] Warmup...")
    for i in range(5):
        send_request(PROMPTS[i], -1)

    # Benchmark
    print("\n[2/3] Running benchmark...")
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(send_request, PROMPTS[i], i)
            for i in range(NUM_REQUESTS)
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 10 == 0:
                print(f"  Completed: {len(results)}/{NUM_REQUESTS}")

    end_time = time.time()

    # Calculate metrics
    print("\n[3/3] Calculating metrics...")
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    total_time = end_time - start_time
    total_tokens = sum(r["tokens"] for r in successful)
    avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0

    metrics = {
        "benchmark_type": "text-to-text",
        "model": "Qwen3.8-27B",
        "devices": "0-3",
        "timestamp": datetime.now().isoformat(),
        "results": {
            "total_requests": NUM_REQUESTS,
            "successful": len(successful),
            "failed": len(failed),
            "total_time_sec": round(total_time, 2),
            "total_tokens": total_tokens,
            "throughput_tokens_per_sec": round(total_tokens / total_time, 2),
            "throughput_requests_per_sec": round(len(successful) / total_time, 2),
            "avg_latency_sec": round(avg_latency, 2),
        }
    }

    # Save results
    output_file = f"/data/benchmark_results/text_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    main()
