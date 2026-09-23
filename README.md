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

### Prerequisites

- **vllm-gaudi 0.26.0** must be pre-installed (typically on Intel Gaudi clusters)
- Python >= 3.9

### Quick Start

```bash
# Clone the repository
git clone https://github.com/wzx-infer/gaudi-vllm-accl.git
cd gaudi-vllm-accl

# Install plugin (no dependency downloads)
pip install -e . --no-deps

# Verify installation
python -c "from gaudi_vllm_accl.register import register; register()"
```

**Why `--no-deps`?** This plugin relies on the pre-installed `vllm-gaudi==0.26.0` in your environment. The `--no-deps` flag prevents pip from attempting to download incompatible vLLM versions.

For detailed installation instructions and troubleshooting, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

### Version Requirements

- ✅ **Supported**: vllm-gaudi 0.26.0
- ❌ **Not compatible**: vllm >= 0.6.0 (API differences)

The plugin performs strict version checking at runtime and will raise a clear error if the version doesn't match.

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
