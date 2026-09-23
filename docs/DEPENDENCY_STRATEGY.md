# 依赖管理方案总结

## 问题背景

原始 `pyproject.toml` 声明 `vllm>=0.6.0` 会导致：

1. **包名不匹配**: `vllm-gaudi` 是独立包名，pip 认为 `vllm` 依赖未满足
2. **版本冲突**: 环境锁定 `vllm-gaudi==0.26.0`，但 pip 会尝试下载 `vllm>=0.6.0`
3. **环境破坏**: 强制安装会覆盖已预装的 `vllm-gaudi`，导致 Gaudi 特定功能丢失

---

## 方案 A：无依赖 + 运行时检查（✅ 已采用）

### 配置

**pyproject.toml**:
```toml
dependencies = []

[project.optional-dependencies]
gaudi = ["vllm-gaudi==0.26.0"]  # 仅用于文档说明
dev = ["pytest>=7.0", "black>=23.0", "ruff>=0.1.0"]
```

**register.py**:
```python
REQUIRED_VLLM_VERSION = "0.26.0"

def _check_vllm_version() -> None:
    import vllm
    if vllm.__version__ != REQUIRED_VLLM_VERSION:
        raise RuntimeError(f"vLLM version mismatch! Required: {REQUIRED_VLLM_VERSION}")
```

### 安装命令

```bash
# 不会触发任何 vLLM 下载
pip install -e . --no-deps
```

### 优点

✅ 完全避免 pip 依赖冲突  
✅ 适配离线 Gaudi 集群  
✅ 灵活支持 `vllm-gaudi` 或其他 fork  
✅ 清晰的运行时错误提示  

### 缺点

⚠️ 依赖关系不在 pip metadata 中可见  
⚠️ 用户需要手动确保 `vllm-gaudi==0.26.0` 已安装  

---

## 方案 B：显式依赖声明（备选）

### 配置

**pyproject.toml**:
```toml
dependencies = ["vllm-gaudi==0.26.0"]

[project.optional-dependencies]
dev = ["pytest>=7.0", "black>=23.0", "ruff>=0.1.0"]
```

### 安装命令

```bash
# pip 会检查 vllm-gaudi==0.26.0，已安装则跳过
pip install -e .

# 如果 vllm-gaudi 不在 PyPI，需指定 wheel 位置
pip install -e . --find-links /path/to/wheels
```

### 优点

✅ 依赖关系显式声明在 `pyproject.toml`  
✅ `pip show gaudi-vllm-accl` 可看到依赖  
✅ 已预装时不会重装  

### 缺点

⚠️ 如果 `vllm-gaudi` 不在 pip 可访问源，会报错  
⚠️ 需要配置私有 pip index 或本地 wheel 路径  
⚠️ 依然有触发网络请求的风险（pip 会尝试查找包）  

---

## 技术细节对比

| 维度 | 方案 A | 方案 B |
|------|--------|--------|
| **pip install 行为** | 不触发任何网络请求 | 会查询 pip 源，但不会重装已有版本 |
| **离线环境** | ✅ 完美支持 | ⚠️ 需要 `--no-index` 或预配置源 |
| **依赖可见性** | ❌ 不在 pip metadata | ✅ `pip show` 可见 |
| **错误提示时机** | 运行时（调用 `register()`） | 安装时（如果包不可访问） |
| **灵活性** | ✅ 支持任何 vLLM fork | ⚠️ 硬编码 `vllm-gaudi` 包名 |
| **用户体验** | 需要阅读文档确认依赖 | pip 自动提示依赖需求 |

---

## 为什么选择方案 A？

1. **Gaudi 集群场景**: `vllm-gaudi` 通常由集群管理员预装，用户无需自己安装
2. **离线环境友好**: Gaudi 测试机可能无法访问公网 PyPI
3. **避免误操作**: 防止用户误安装 `vllm>=0.6.0` 破坏环境
4. **运行时验证**: `register()` 在插件加载时明确检查版本，错误提示清晰

---

## 版本检查机制

### 实现位置

`src/gaudi_vllm_accl/register.py:_check_vllm_version()`

### 检查逻辑

```python
import vllm

# 1. 检查 vLLM 是否安装
if not hasattr(vllm, '__version__'):
    logger.warning("Could not determine vLLM version")
    return

# 2. 严格版本匹配
if vllm.__version__ != "0.26.0":
    raise RuntimeError(
        f"vLLM version mismatch!\n"
        f"  Required: 0.26.0\n"
        f"  Installed: {vllm.__version__}\n"
        f"Please install: pip install vllm-gaudi==0.26.0"
    )
```

### 触发时机

- vLLM 通过 entry point 调用 `register()` 时
- 用户手动调用 `from gaudi_vllm_accl.register import register; register()` 时

### 错误示例

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

## 验证步骤

### 1. 检查 dependencies 为空

```bash
$ cat pyproject.toml | grep -A 5 "^dependencies"
dependencies = []

[project.optional-dependencies]
gaudi = ["vllm-gaudi==0.26.0"]
```

### 2. 安装不触发下载

```bash
$ pip install -e . --no-deps
Obtaining file:///path/to/gaudi-vllm-accl
  Preparing metadata (pyproject.toml) ... done
Installing collected packages: gaudi-vllm-accl
Successfully installed gaudi-vllm-accl-0.1.0
```

（无任何 "Collecting vllm..." 或网络请求）

### 3. 版本检查生效

```bash
# 正确版本
$ python -c "from gaudi_vllm_accl.register import register; register()"
INFO:gaudi_vllm_accl.register:vLLM version check passed: 0.26.0

# 错误版本（假设安装了 0.28.0）
$ python -c "from gaudi_vllm_accl.register import register; register()"
RuntimeError: vLLM version mismatch!
  Required: 0.26.0
  Installed: 0.28.0
```

---

## 相关文件

- **配置**: `pyproject.toml`
- **版本检查**: `src/gaudi_vllm_accl/register.py`
- **安装文档**: `docs/INSTALLATION.md`
- **验证脚本**: `scripts/m0_api_check.py` (用于检查 vLLM API)

---

## 下一步

等待从 Gaudi 测试机返回的 M0 验证结果（`api_check.json`），以确认：
1. vLLM 0.26.0 的实际 API 结构
2. Plugin entry point 机制是否可用
3. ModelRegistry 的注册方法签名
