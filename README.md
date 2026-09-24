# gaudi-vllm-accl

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

`gaudi-vllm-accl` is a lightweight **plugin package** for vLLM that adds support for new models and acceleration features on Intel Gaudi accelerators.

**Key Features**:
- 🔌 **Plugin Architecture**: Clean integration via vLLM's plugin mechanism (no forking)
- 🚀 **New Model Support**: DeepSeek V4.1 Flash with Sparse MLA + MegaMoE
- ⚡ **Gaudi Optimizations**: Attention and MoE optimizations for Gaudi2/Gaudi3

**Design Philosophy**: This package does not fork or copy upstream vLLM-Gaudi. Instead, it uses vLLM's entry point mechanism to inject models and features cleanly.

### Architecture

```
┌─────────────────────────────────────────────┐
│  Upstream vLLM-Gaudi (pip dependency)       │
│  - Core vLLM framework                      │
│  - Gaudi backend integration                │
└─────────────────────────────────────────────┘
                    ▲
                    │ Plugin Entry Point
                    │ (vllm.general_plugins)
┌─────────────────────────────────────────────┐
│  gaudi-vllm-accl (this package)             │
│  ├── Custom Transformers Configs            │
│  ├── New Models (DeepSeek V4.1 Flash, ...)  │
│  └── Acceleration Features                  │
│      ├── Attention optimizations            │
│      └── MoE optimizations                  │
└─────────────────────────────────────────────┘
```

### Installation

#### Prerequisites

- **vLLM-Gaudi 0.26.0** (pre-installed on Gaudi clusters)
- Python >= 3.9
- Intel Gaudi2/Gaudi3 accelerator

#### Quick Start

```bash
# Clone repository
git clone https://github.com/wzx-infer/gaudi-vllm-accl.git
cd gaudi-vllm-accl

# Install plugin (no dependency downloads)
pip install -e . --no-deps

# Verify installation
python -c "from gaudi_vllm_accl.register import register; register(); print('✓ Plugin loaded')"
```

**Why `--no-deps`?** This plugin relies on pre-installed `vllm-gaudi==0.26.0`. The `--no-deps` flag prevents pip from downloading incompatible vLLM versions.

#### Version Requirements

- ✅ **Supported**: vllm-gaudi 0.26.0
- ❌ **Not compatible**: vllm >= 0.6.0 (API breaking changes)

### Usage

Once installed, the plugin automatically registers with vLLM via entry points.

```python
from vllm import LLM

# Plugin automatically registers DeepseekV41ForCausalLM
llm = LLM(
    model="/path/to/DeepSeek-V4.1-Flash",
    dtype="bfloat16",
    tensor_parallel_size=8,
    max_model_len=8192,
)

# Generate
outputs = llm.generate(["Hello, how are you?"])
for output in outputs:
    print(output.outputs[0].text)
```

### Project Structure

```
gaudi-vllm-accl/
├── src/gaudi_vllm_accl/
│   ├── register.py              # Plugin entry point
│   ├── models/
│   │   ├── deepseek_v41_flash.py   # Config + model class
│   │   └── gaudi/
│   │       └── model.py            # Gaudi backend implementation
│   └── accel/                   # Acceleration features (TBD)
├── benchmarks/                  # Performance benchmarks
├── tests/                       # Unit tests
└── pyproject.toml
```

### Development Status

**Phase 1: BF16 Baseline** (In Progress)
- ✅ Project structure
- ✅ DeepSeekV41Config with Transformers AutoConfig registration
- ✅ Simplified model implementation (339 lines)
  - Standard QKV attention (CSA2 sparse MLA pending)
  - Dense FFN (full MoE with 384 experts pending)
  - DecoderLayer + CausalLM entry point
- ⚠️ **Current Blocker**: Plugin loading order issue
  - vLLM calls `AutoConfig.from_pretrained()` before plugin `register()` executes
  - Need to ensure config registration happens before model config validation

**Phase 2: Full Architecture** (Planned)
- [ ] CSA2 sparse MLA attention
- [ ] MegaMoE with 384 routed experts + 1 shared expert
- [ ] Quantization handling (MXFP4 weights, FP8 e4m0 activations → Gaudi2 FP8 e4m3/BF16)
- [ ] CED (Compressed Embedding Dictionary)
- [ ] Engram attention (sparse long-term memory)

### Known Issues

1. **Plugin Loading Order**: vLLM initializes `ModelConfig` before loading `vllm.general_plugins`, causing Transformers to fail recognizing `deepseek_v41` model type
   - **Workaround**: Investigating vLLM's plugin loading mechanism
   
2. **Missing `embed_input_ids` Method**: vLLM warns about missing method (non-blocking)

3. **Version Check**: Currently using warning instead of strict error for version mismatch

### Contributing

This is currently a personal research project. Contributions welcome after initial stable release.

### License

Apache-2.0

---

<a name="chinese"></a>
## 中文

### 概述

`gaudi-vllm-accl` 是一个轻量级的 vLLM **插件包**，用于在 Intel Gaudi 加速器上添加新模型支持和加速特性。

**核心特性**：
- 🔌 **插件架构**：通过 vLLM 的插件机制实现干净的集成（无需 fork）
- 🚀 **新模型支持**：DeepSeek V4.1 Flash（Sparse MLA + MegaMoE）
- ⚡ **Gaudi 优化**：针对 Gaudi2/Gaudi3 的 Attention 和 MoE 优化

**设计理念**：本项目不 fork 或复制上游 vLLM-Gaudi 代码，而是通过 vLLM 的入口点机制实现模型和特性的干净注入。

### 架构设计

```
┌─────────────────────────────────────────────┐
│  上游 vLLM-Gaudi (pip 依赖)                  │
│  - vLLM 核心框架                             │
│  - Gaudi 后端集成                            │
└─────────────────────────────────────────────┘
                    ▲
                    │ 插件入口点
                    │ (vllm.general_plugins)
┌─────────────────────────────────────────────┐
│  gaudi-vllm-accl (本项目)                    │
│  ├── 自定义 Transformers 配置                │
│  ├── 新模型实现 (DeepSeek V4.1 Flash 等)     │
│  └── 加速特性                                │
│      ├── Attention 优化                      │
│      └── MoE 优化                            │
└─────────────────────────────────────────────┘
```

### 安装

#### 前置要求

- **vLLM-Gaudi 0.26.0**（Gaudi 集群通常已预装）
- Python >= 3.9
- Intel Gaudi2/Gaudi3 加速器

#### 快速开始

```bash
# 克隆仓库
git clone https://github.com/wzx-infer/gaudi-vllm-accl.git
cd gaudi-vllm-accl

# 安装插件（不下载依赖）
pip install -e . --no-deps

# 验证安装
python -c "from gaudi_vllm_accl.register import register; register(); print('✓ 插件加载成功')"
```

**为什么使用 `--no-deps`？** 本插件依赖环境中预装的 `vllm-gaudi==0.26.0`。`--no-deps` 标志可防止 pip 下载不兼容的 vLLM 版本。

#### 版本要求

- ✅ **支持**：vllm-gaudi 0.26.0
- ❌ **不兼容**：vllm >= 0.6.0（API 破坏性变更）

### 使用方法

安装后，插件会通过入口点机制自动注册到 vLLM。

```python
from vllm import LLM

# 插件自动注册 DeepseekV41ForCausalLM
llm = LLM(
    model="/path/to/DeepSeek-V4.1-Flash",
    dtype="bfloat16",
    tensor_parallel_size=8,
    max_model_len=8192,
)

# 生成
outputs = llm.generate(["你好，最近怎么样？"])
for output in outputs:
    print(output.outputs[0].text)
```

### 项目结构

```
gaudi-vllm-accl/
├── src/gaudi_vllm_accl/
│   ├── register.py              # 插件入口点
│   ├── models/
│   │   ├── deepseek_v41_flash.py   # Config + 模型类
│   │   └── gaudi/
│   │       └── model.py            # Gaudi 后端实现
│   └── accel/                   # 加速特性（待实现）
├── benchmarks/                  # 性能基准测试
├── tests/                       # 单元测试
└── pyproject.toml
```

### 开发状态

**阶段 1：BF16 基线** (进行中)
- ✅ 项目结构搭建
- ✅ DeepSeekV41Config 及 Transformers AutoConfig 注册
- ✅ 简化模型实现（339 行）
  - 标准 QKV attention（CSA2 sparse MLA 待实现）
  - Dense FFN（完整 384 expert MoE 待实现）
  - DecoderLayer + CausalLM 主入口
- ⚠️ **当前卡点**：插件加载顺序问题
  - vLLM 在插件 `register()` 执行前调用 `AutoConfig.from_pretrained()`
  - 需要确保 config 注册在模型配置验证之前完成

**阶段 2：完整架构** (计划中)
- [ ] CSA2 sparse MLA attention
- [ ] MegaMoE（384 个路由 expert + 1 个共享 expert）
- [ ] 量化处理（MXFP4 权重，FP8 e4m0 激活值 → Gaudi2 FP8 e4m3/BF16）
- [ ] CED (压缩嵌入字典)
- [ ] Engram attention（稀疏长期记忆）

### 已知问题

1. **插件加载顺序问题**：vLLM 在加载 `vllm.general_plugins` 之前初始化 `ModelConfig`，导致 Transformers 无法识别 `deepseek_v41` 模型类型
   - **解决方案**：正在研究 vLLM 的插件加载机制
   
2. **缺少 `embed_input_ids` 方法**：vLLM 发出警告（不阻塞运行）

3. **版本检查**：当前使用 warning 而非严格错误处理版本不匹配

### 参与贡献

本项目目前是个人研究项目。首次稳定版本发布后欢迎贡献。

### 开源协议

Apache-2.0

---

## Quick Links

- **Repository**: https://github.com/wzx-infer/gaudi-vllm-accl
- **Issues**: https://github.com/wzx-infer/gaudi-vllm-accl/issues
- **vLLM**: https://github.com/vllm-project/vllm
- **vLLM-Gaudi**: https://github.com/HabanaAI/vllm-fork

## Acknowledgments

This project is built on top of:
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference engine
- [vLLM-Gaudi](https://github.com/HabanaAI/vllm-fork) - Gaudi backend integration
- DeepSeek V4.1 Flash architecture by DeepSeek AI
