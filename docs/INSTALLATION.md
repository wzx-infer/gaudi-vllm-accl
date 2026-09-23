# 安装指南

## 环境要求

- **Python**: >= 3.9
- **vLLM**: 0.26.0 (通过 vllm-gaudi 提供)
- **vllm-gaudi**: 0.26.0 (预装在 Gaudi 集群)
- **硬件**: Intel Gaudi 2 加速器

## 方案 A：无依赖安装（推荐，适用于 Gaudi 集群）

### 特点
- ✅ 不会触发任何 vLLM 相关的 pip 下载
- ✅ 适配离线环境
- ✅ 依赖已预装的 `vllm-gaudi==0.26.0`

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/wzx-infer/gaudi-vllm-accl.git
cd gaudi-vllm-accl

# 2. 可编辑模式安装（不安装任何依赖）
pip install -e . --no-deps

# 3. 验证插件注册成功
python -c "from importlib.metadata import entry_points; \
eps = list(entry_points(group='vllm.general_plugins')); \
print('Plugin registered:', any(ep.name == 'gaudi_vllm_accl' for ep in eps))"

# 4. 验证版本检查
python -c "from gaudi_vllm_accl.register import register; register()"
```

### 预期输出

```
Plugin registered: True
INFO:gaudi_vllm_accl.register:Registering gaudi-vllm-accl plugin...
INFO:gaudi_vllm_accl.register:vLLM version check passed: 0.26.0
...
```

---

## 方案 B：显式依赖声明

### 特点
- ✅ 依赖关系在 `pyproject.toml` 中显式声明
- ⚠️ 需要 `vllm-gaudi` 可从 pip 安装源访问

### pyproject.toml 配置

```toml
dependencies = [
    "vllm-gaudi==0.26.0",
]
```

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/wzx-infer/gaudi-vllm-accl.git
cd gaudi-vllm-accl

# 2. 如果 vllm-gaudi 在 PyPI 或私有源
pip install -e .

# 3. 如果 vllm-gaudi wheel 在本地
pip install -e . --find-links /path/to/wheels
```

---

## 开发模式安装

如果需要安装开发工具（pytest, black, ruff 等）：

```bash
# 方案 A：手动安装 dev 依赖
pip install -e . --no-deps
pip install pytest pytest-cov black ruff isort mypy

# 方案 B：使用 optional-dependencies
pip install -e .[dev] --no-deps
```

---

## 验证安装

### 1. 检查 vLLM 版本

```bash
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

预期输出：`vLLM version: 0.26.0`

### 2. 检查插件加载

```bash
python -c "
from importlib.metadata import entry_points
eps = list(entry_points(group='vllm.general_plugins'))
for ep in eps:
    print(f'Plugin: {ep.name} -> {ep.value}')
"
```

预期输出：
```
Plugin: gaudi_vllm_accl -> gaudi_vllm_accl.register:register
```

### 3. 手动触发插件注册

```bash
python -c "from gaudi_vllm_accl.register import register; register()"
```

预期输出：
```
INFO:gaudi_vllm_accl.register:Registering gaudi-vllm-accl plugin...
INFO:gaudi_vllm_accl.register:vLLM version check passed: 0.26.0
INFO:gaudi_vllm_accl.register:Registered model: DeepSeekV41FlashForCausalLM
INFO:gaudi_vllm_accl.register:Acceleration features registration: not yet implemented
INFO:gaudi_vllm_accl.register:gaudi-vllm-accl plugin registered successfully.
```

---

## 常见问题

### Q1: 安装时提示 "vllm>=0.6.0 not found"

**原因**: 使用了旧版本的 `pyproject.toml`，其中 `dependencies` 包含 `vllm>=0.6.0`

**解决**:
```bash
# 拉取最新代码
git pull origin main

# 确认 dependencies = [] 为空
cat pyproject.toml | grep -A 2 "dependencies"

# 重新安装
pip uninstall gaudi-vllm-accl -y
pip install -e . --no-deps
```

### Q2: 运行时提示 "vLLM version mismatch"

**原因**: 环境中安装的 vLLM 版本不是 0.26.0

**解决**:
```bash
# 检查当前版本
python -c "import vllm; print(vllm.__version__)"

# 如果不是 0.26.0，重新安装 vllm-gaudi
pip uninstall vllm vllm-gaudi -y
pip install vllm-gaudi==0.26.0
```

### Q3: Entry point 没有注册成功

**原因**: 可能是 setuptools 缓存问题

**解决**:
```bash
# 清理缓存并重新安装
pip uninstall gaudi-vllm-accl -y
rm -rf src/gaudi_vllm_accl.egg-info
pip install -e . --no-deps

# 重新验证
python -c "from importlib.metadata import entry_points; \
print(list(entry_points(group='vllm.general_plugins')))"
```

### Q4: Windows 环境编码错误

**原因**: Windows 默认使用 GBK 编码

**解决**: 本项目代码已自动处理，但如果仍有问题：
```bash
# 设置环境变量
set PYTHONIOENCODING=utf-8
python your_script.py
```

---

## 卸载

```bash
pip uninstall gaudi-vllm-accl -y
```

插件的 entry point 会自动移除。
