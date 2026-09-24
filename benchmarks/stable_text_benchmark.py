#!/usr/bin/env python3
"""Stable text benchmark using vLLM offline inference"""
import os
import time
import json
from datetime import datetime

os.environ["HABANA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_USE_V1"] = "0"  # Disable v1 engine

from vllm import LLM, SamplingParams

MODEL = "/data/models/Qwen/Qwen3.8-27B"
prompts = ["Explain machine learning.", "Write Python code.", "Compare TCP and UDP.", "Describe photosynthesis."] * 25

print("="*80)
print("Text Benchmark (Devices 0-3)")
print("="*80)

print("\nLoading model...")
llm = LLM(model=MODEL, tensor_parallel_size=4, dtype="bfloat16", max_model_len=2048, trust_remote_code=True)

print("Warmup...")
llm.generate(prompts[:5], SamplingParams(max_tokens=256))

print("Benchmarking 100 requests...")
start = time.time()
outputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=256))
elapsed = time.time() - start

tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
result = {
    "type": "text-to-text",
    "devices": "0-3",
    "requests": 100,
    "time_sec": round(elapsed, 2),
    "tokens": tokens,
    "throughput_tok_s": round(tokens/elapsed, 2),
    "throughput_req_s": round(100/elapsed, 2)
}

os.makedirs("/data/benchmark_results", exist_ok=True)
path = f"/data/benchmark_results/text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(path, 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*80)
print(json.dumps(result, indent=2))
print(f"\nSaved: {path}")
