# ✅ 依赖冲突修复完成

## 📋 问题回顾

**原问题**: `pip install -e .` 会尝试下载 `vllm>=0.6.0`，与环境中已安装的 `vllm-gaudi==0.26.0` 冲突。

**根本原因**:
1. `vllm-gaudi` 是独立包名，pip 不认为它满足 `vllm` 依赖
2. pip 依赖解析会强制尝试从 PyPI 下载 `vllm`
3. 版本不兼容（0.26.0 vs >=0.6.0）会破坏 Gaudi 环境

---

## ✅ 已完成的修改

### 1. 更新 `pyproject.toml`（方案 A）

```toml
# 之前
dependencies = ["vllm>=0.6.0"]

# 之后
dependencies = []

[project.optional-dependencies]
gaudi = ["vllm-gaudi==0.26.0"]  # 仅用于文档说明
```

### 2. 增强 `register.py` 运行时检查

新增版本验证函数：

```python
REQUIRED_VLLM_VERSION = "0.26.0"

def _check_vllm_version() -> None:
    """严格检查 vLLM 版本，不匹配时抛出清晰错误"""
    import vllm
    if vllm.__version__ != REQUIRED_VLLM_VERSION:
        raise RuntimeError(
            f"vLLM version mismatch!\n"
            f"  Required: {REQUIRED_VLLM_VERSION}\n"
            f"  Installed: {vllm.__version__}"
        )
```

在 `register()` 函数开头调用检查。

### 3. 文档更新

创建文档：
- ✅ `docs/INSTALLATION.md` - 详细安装指南
- ✅ `docs/DEPENDENCY_STRATEGY.md` - 方案对比和技术细节
- ✅ `README.md` - 更新安装说明

---

## 🚀 新的安装流程

### 在 Gaudi 测试机上

```bash
# 1. 确认环境版本
python -c "import vllm; print('vLLM version:', vllm.__version__)"
# 预期输出: vLLM version: 0.26.0

# 2. 克隆仓库
git clone https://github.com/wzx-infer/gaudi-vllm-accl.git
cd gaudi-vllm-accl

# 3. 安装插件（不会触发任何下载）
pip install -e . --no-deps

# 4. 验证插件注册
python -c "from gaudi_vllm_accl.register import register; register()"
```

### 预期输出

```
INFO:gaudi_vllm_accl.register:Registering gaudi-vllm-accl plugin...
INFO:gaudi_vllm_accl.register:vLLM version check passed: 0.26.0
INFO:gaudi_vllm_accl.register:Registered model: DeepSeekV41FlashForCausalLM
INFO:gaudi_vllm_accl.register:Acceleration features registration: not yet implemented
INFO:gaudi_vllm_accl.register:gaudi-vllm-accl plugin registered successfully.
```

---

## 📊 方案对比

| 特性 | 方案 A（✅ 已采用） | 方案 B（备选） |
|------|-------------------|---------------|
| pip install 行为 | 完全无网络请求 | 查询 pip 源但不重装 |
| 离线环境 | ✅ 完美支持 | ⚠️ 需配置 |
| 依赖可见性 | ❌ 运行时检查 | ✅ pip metadata |
| 错误提示 | 运行时清晰报错 | 安装时可能报错 |
| 灵活性 | ✅ 支持任何 fork | ⚠️ 硬编码包名 |

---

## 🔍 版本检查机制

### 触发时机
- vLLM 加载插件时（自动）
- 手动调用 `register()` 时

### 错误示例

如果环境版本错误：

```
RuntimeError: vLLM version mismatch!
  Required: 0.26.0
  Installed: 0.28.1
This plugin is specifically designed for vLLM 0.26.0 (or vllm-gaudi 0.26.0).
Using a different version may cause API incompatibilities or runtime errors.
Please install the correct version:
  pip install vllm-gaudi==0.26.0
```

---

## 📁 修改的文件

1. ✅ `pyproject.toml` - 移除 vllm 依赖，添加 optional-dependencies
2. ✅ `src/gaudi_vllm_accl/register.py` - 增加版本检查函数
3. ✅ `README.md` - 更新安装说明
4. ✅ `docs/INSTALLATION.md` - 新建详细安装指南
5. ✅ `docs/DEPENDENCY_STRATEGY.md` - 新建方案对比文档

---

## ⚠️ 重要约束

1. **禁止手动编辑 `requires.txt`** - 这是自动生成的文件
2. **本地 Windows 环境不运行 vLLM** - 仅用于代码编写
3. **所有测试在远程 Gaudi 机器进行** - 由用户手动上传执行

---

## 🎯 下一步

等待用户从 Gaudi 测试机返回的验证结果：

1. **M0 API 检查**: 运行 `scripts/m0_api_check.py`，返回 `api_check.json`
2. **插件安装验证**: 确认 `pip install -e . --no-deps` 成功
3. **版本检查验证**: 确认运行时版本检查生效

收到结果后，开始 M1 里程碑实现。

---

## 📚 参考文档

- [安装指南](docs/INSTALLATION.md)
- [依赖策略](docs/DEPENDENCY_STRATEGY.md)
- [里程碑规划](MILESTONES.md)
