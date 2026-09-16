# gaudi-vllm-accl

Lightweight vLLM plugin for Intel Gaudi accelerators.

## Overview

`gaudi-vllm-accl` is a **plugin package** for vLLM that provides:

1. **New Model Support**: Integration for models not yet in upstream vLLM-Gaudi (e.g., DeepSeek V4.1 Flash)
2. **Acceleration Features**: Reusable optimizations applicable across models

This package **does not** fork or copy upstream vLLM-Gaudi. Instead, it uses vLLM's plugin mechanism to inject models and features cleanly.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Upstream vLLM-Gaudi (pip dependency)       │
│  - Core vLLM framework                      │
│  - Gaudi backend integration                │
└─────────────────────────────────────────────┘
                    ▲
                    │ Plugin Entry Point
                    │
┌─────────────────────────────────────────────┐
│  gaudi-vllm-accl (this package)             │
│  ├── New Models (DeepSeek V4.1 Flash, ...)  │
│  └── Acceleration Features                  │
│      ├── Attention optimizations            │
│      └── MoE optimizations                  │
└─────────────────────────────────────────────┘
```

## Installation

```bash
# Install from source (development)
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Usage

Once installed, the plugin is automatically registered with vLLM via entry points. No code changes needed in your vLLM scripts.

```python
from vllm import LLM

# The plugin automatically registers DeepSeekV41FlashForCausalLM
llm = LLM(model="deepseek-ai/DeepSeek-V4.1-Flash", ...)
```

## Project Structure

```
gaudi-vllm-accl/
├── src/gaudi_vllm_accl/
│   ├── register.py          # Plugin entry point
│   ├── models/              # New model implementations
│   │   └── deepseek_v41_flash.py
│   └── accel/               # Acceleration features
│       ├── attention/
│       └── moe/
├── benchmarks/              # Performance benchmarks
├── tests/                   # Unit tests
└── pyproject.toml
```

## Development Status

🚧 **Early Development** - Core structure is in place, model implementations are in progress.

## Contributing

This is currently a personal project. Contributions welcome after initial release.

## License

Apache-2.0
