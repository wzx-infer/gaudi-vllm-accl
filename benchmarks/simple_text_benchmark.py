#!/usr/bin/env python3
"""Simple text-to-text benchmark for Qwen3.8-27B"""
import os
import time
import json
from datetime import datetime

os.environ["HABANA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams

MODEL_PATH = "/data/models/Qwen/Qwen3.8-27B"
NUM_PROMPTS = 100

prompts = [
    "Explain machine learning briefly.",
    "Write a Python sorting function.",
    "Compare TCP and UDP protocols.",
    "Describe photosynthesis process.",
] * 25

print("="*80)
print("Qwen3.8-27B Text Benchmark (Device 0)")
print("="*80)

print("\n[1/3] Loading model...")
start_load = time.time()
llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=2048)
load_time = time.time() - start_load
print(f"✓ Loaded in {load_time:.1f}s")

sampling_params = SamplingParams(temperature=0.8, max_tokens=256)

print("\n[2/3] Warmup (5 requests)...")
llm.generate(prompts[:5], sampling_params)

print("\n[3/3] Benchmarking...")
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
metrics = {
    "type": "text-to-text",
    "device": "0",
    "requests": NUM_PROMPTS,
    "time_sec": round(elapsed, 2),
    "total_tokens": total_tokens,
    "throughput_tok_s": round(total_tokens / elapsed, 2),
    "throughput_req_s": round(NUM_PROMPTS / elapsed, 2),
}

output_file = f"/data/benchmark_results/text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
os.makedirs("/data/benchmark_results", exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(json.dumps(metrics, indent=2))
print(f"\nSaved: {output_file}")
