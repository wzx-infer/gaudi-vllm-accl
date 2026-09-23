#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M0: vLLM 0.26.0 API Verification Script

This script inspects the vLLM 0.26.0 API to determine:
1. ModelRegistry location and API
2. Plugin system support
3. Attention layer signatures
4. Parallel layer APIs
5. MoE support
6. KV cache configuration

Output: Prints findings to console and saves to api_check.json
"""

import inspect
import json
import sys
import io
from typing import Any, Dict, List, Optional

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def safe_import(module_path: str, obj_name: Optional[str] = None):
    """Safely import a module or object, returning None on failure."""
    try:
        parts = module_path.split('.')
        module = __import__(module_path)
        for part in parts[1:]:
            module = getattr(module, part)

        if obj_name:
            return getattr(module, obj_name, None)
        return module
    except (ImportError, AttributeError) as e:
        return None


def get_signature_info(obj) -> Dict[str, Any]:
    """Extract signature information from a callable."""
    if obj is None:
        return {"error": "Object not found"}

    try:
        sig = inspect.signature(obj)
        return {
            "parameters": [
                {
                    "name": name,
                    "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    "default": str(param.default) if param.default != inspect.Parameter.empty else "Required"
                }
                for name, param in sig.parameters.items()
            ],
            "return_annotation": str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else "Any"
        }
    except (ValueError, TypeError) as e:
        return {"error": str(e)}


def get_class_fields(cls) -> List[str]:
    """Extract field names from a class."""
    if cls is None:
        return []

    fields = []
    # Check for dataclass fields
    if hasattr(cls, '__dataclass_fields__'):
        fields = list(cls.__dataclass_fields__.keys())
    # Check for __annotations__
    elif hasattr(cls, '__annotations__'):
        fields = list(cls.__annotations__.keys())
    # Fallback to __dict__
    else:
        fields = [k for k in dir(cls) if not k.startswith('_')]

    return fields


def check_modelregistry() -> Dict[str, Any]:
    """Check 1: ModelRegistry API"""
    print("=" * 80)
    print("1. Checking ModelRegistry...")
    print("=" * 80)

    result = {
        "module_path": None,
        "register_model_signature": None,
        "supported_archs": [],
        "additional_methods": []
    }

    # Try different possible locations
    possible_paths = [
        ("vllm.model_executor.models", "ModelRegistry"),
        ("vllm.model_executor.models.registry", "ModelRegistry"),
        ("vllm.transformers_utils.models", "ModelRegistry"),
    ]

    registry_class = None
    found_path = None

    for module_path, class_name in possible_paths:
        registry_class = safe_import(module_path, class_name)
        if registry_class:
            found_path = f"{module_path}.{class_name}"
            result["module_path"] = found_path
            print(f"✓ Found ModelRegistry at: {found_path}")
            break

    if not registry_class:
        print("✗ ModelRegistry not found in any expected location")
        return result

    # Check register_model method
    if hasattr(registry_class, 'register_model'):
        result["register_model_signature"] = get_signature_info(registry_class.register_model)
        print(f"✓ register_model method exists")
        print(f"  Signature: {result['register_model_signature']}")
    else:
        print("✗ register_model method not found")

    # Get supported architectures
    if hasattr(registry_class, 'get_supported_archs'):
        try:
            archs = registry_class.get_supported_archs()
            result["supported_archs"] = list(archs) if archs else []
            print(f"✓ Supported architectures: {len(result['supported_archs'])} total")
            # Show a few examples
            if result["supported_archs"]:
                print(f"  Examples: {result['supported_archs'][:5]}")
        except Exception as e:
            print(f"✗ Error getting supported archs: {e}")

    # List other useful methods
    methods = [m for m in dir(registry_class) if not m.startswith('_') and callable(getattr(registry_class, m, None))]
    result["additional_methods"] = methods
    print(f"✓ Other methods: {', '.join(methods[:10])}")

    return result


def check_plugin_system() -> Dict[str, Any]:
    """Check 2: Plugin entry point support"""
    print("\n" + "=" * 80)
    print("2. Checking Plugin System...")
    print("=" * 80)

    result = {
        "has_plugin_loader": False,
        "entry_points_supported": [],
        "plugin_module_path": None
    }

    # Check for plugin loading code
    possible_paths = [
        "vllm.plugins",
        "vllm.entrypoints.plugins",
        "vllm.utils.plugins",
    ]

    for path in possible_paths:
        module = safe_import(path)
        if module:
            result["plugin_module_path"] = path
            result["has_plugin_loader"] = True
            print(f"✓ Plugin module found at: {path}")

            # Check for load_plugins or similar functions
            funcs = [f for f in dir(module) if 'plugin' in f.lower() and not f.startswith('_')]
            print(f"  Functions: {funcs}")
            break

    if not result["has_plugin_loader"]:
        print("✗ No plugin system found")

    # Try to check entry points using importlib.metadata
    try:
        from importlib.metadata import entry_points

        # Try different possible entry point groups
        possible_groups = [
            "vllm.general_plugins",
            "vllm.plugins",
            "vllm_plugins",
        ]

        for group in possible_groups:
            try:
                eps = entry_points(group=group)
                if eps:
                    result["entry_points_supported"].append(group)
                    print(f"✓ Entry point group '{group}' is recognized")
            except Exception:
                pass

        if not result["entry_points_supported"]:
            print("⚠ No vLLM entry point groups found (may not be registered yet)")
    except ImportError:
        print("⚠ importlib.metadata not available")

    return result


def check_attention() -> Dict[str, Any]:
    """Check 3: Attention layer API"""
    print("\n" + "=" * 80)
    print("3. Checking Attention Layer...")
    print("=" * 80)

    result = {
        "module_path": None,
        "init_signature": None,
        "forward_signature": None,
        "attention_backend": None
    }

    # Try to find Attention class
    possible_paths = [
        ("vllm.model_executor.layers.attention", "Attention"),
        ("vllm.attention", "Attention"),
        ("vllm.model_executor.layers.attention.attention", "Attention"),
    ]

    attention_class = None
    for module_path, class_name in possible_paths:
        attention_class = safe_import(module_path, class_name)
        if attention_class:
            result["module_path"] = f"{module_path}.{class_name}"
            print(f"✓ Found Attention at: {result['module_path']}")
            break

    if not attention_class:
        print("✗ Attention class not found")
        return result

    # Get __init__ signature
    result["init_signature"] = get_signature_info(attention_class.__init__)
    print(f"✓ __init__ signature:")
    for param in result["init_signature"].get("parameters", []):
        print(f"    {param['name']}: {param['annotation']} = {param['default']}")

    # Get forward signature
    if hasattr(attention_class, 'forward'):
        result["forward_signature"] = get_signature_info(attention_class.forward)
        print(f"✓ forward signature:")
        for param in result["forward_signature"].get("parameters", []):
            print(f"    {param['name']}: {param['annotation']} = {param['default']}")

    # Check for attention backend system
    backend_paths = [
        "vllm.attention.backends",
        "vllm.model_executor.layers.attention.backends",
    ]

    for path in backend_paths:
        module = safe_import(path)
        if module:
            result["attention_backend"] = path
            backends = [b for b in dir(module) if not b.startswith('_')]
            print(f"✓ Attention backends available at {path}:")
            print(f"  {', '.join(backends[:10])}")
            break

    return result


def check_attention_metadata() -> Dict[str, Any]:
    """Check 4: AttentionMetadata"""
    print("\n" + "=" * 80)
    print("4. Checking AttentionMetadata...")
    print("=" * 80)

    result = {
        "module_path": None,
        "fields": []
    }

    possible_paths = [
        ("vllm.attention", "AttentionMetadata"),
        ("vllm.attention.backends.abstract", "AttentionMetadata"),
        ("vllm.model_executor.layers.attention", "AttentionMetadata"),
    ]

    metadata_class = None
    for module_path, class_name in possible_paths:
        metadata_class = safe_import(module_path, class_name)
        if metadata_class:
            result["module_path"] = f"{module_path}.{class_name}"
            print(f"✓ Found AttentionMetadata at: {result['module_path']}")
            break

    if not metadata_class:
        print("✗ AttentionMetadata not found")
        return result

    # Get fields
    result["fields"] = get_class_fields(metadata_class)
    print(f"✓ Fields ({len(result['fields'])}):")
    for field in result["fields"][:20]:  # Show first 20
        print(f"    - {field}")

    return result


def check_parallel_layers() -> Dict[str, Any]:
    """Check 5: Parallel linear layers"""
    print("\n" + "=" * 80)
    print("5. Checking Parallel Layers...")
    print("=" * 80)

    result = {}

    layers_to_check = [
        "ColumnParallelLinear",
        "RowParallelLinear",
        "VocabParallelEmbedding",
        "QKVParallelLinear",
    ]

    possible_module_paths = [
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.layers.vocab_parallel_embedding",
        "vllm.model_executor.parallel_utils.parallel_state",
    ]

    for layer_name in layers_to_check:
        layer_class = None
        found_path = None

        for module_path in possible_module_paths:
            layer_class = safe_import(module_path, layer_name)
            if layer_class:
                found_path = f"{module_path}.{layer_name}"
                break

        if layer_class:
            print(f"✓ {layer_name} found at: {found_path}")
            result[layer_name] = {
                "module_path": found_path,
                "init_signature": get_signature_info(layer_class.__init__)
            }
            # Show key parameters
            params = result[layer_name]["init_signature"].get("parameters", [])
            key_params = [p["name"] for p in params if p["name"] != "self"][:5]
            print(f"    Key params: {', '.join(key_params)}")
        else:
            print(f"✗ {layer_name} not found")
            result[layer_name] = None

    return result


def check_moe_support() -> Dict[str, Any]:
    """Check 6: MoE layer support"""
    print("\n" + "=" * 80)
    print("6. Checking MoE Support...")
    print("=" * 80)

    result = {
        "fused_moe": None,
        "moe_layer": None,
        "moe_related_modules": []
    }

    # Check for FusedMoE
    possible_paths = [
        ("vllm.model_executor.layers.fused_moe", "FusedMoE"),
        ("vllm.model_executor.layers.moe", "FusedMoE"),
    ]

    for module_path, class_name in possible_paths:
        fused_moe = safe_import(module_path, class_name)
        if fused_moe:
            result["fused_moe"] = {
                "module_path": f"{module_path}.{class_name}",
                "init_signature": get_signature_info(fused_moe.__init__)
            }
            print(f"✓ FusedMoE found at: {result['fused_moe']['module_path']}")
            break

    if not result["fused_moe"]:
        print("✗ FusedMoE not found")

    # Check for generic MoE modules
    moe_module_paths = [
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.moe",
        "vllm.moe",
    ]

    for path in moe_module_paths:
        module = safe_import(path)
        if module:
            result["moe_related_modules"].append(path)
            items = [item for item in dir(module) if not item.startswith('_')]
            print(f"✓ MoE module found: {path}")
            print(f"    Contents: {', '.join(items[:10])}")

    if not result["moe_related_modules"]:
        print("✗ No MoE-related modules found")

    return result


def check_kv_cache_config() -> Dict[str, Any]:
    """Check 7: KV Cache configuration"""
    print("\n" + "=" * 80)
    print("7. Checking KV Cache Config...")
    print("=" * 80)

    result = {
        "module_path": None,
        "fields": []
    }

    possible_paths = [
        ("vllm.config", "CacheConfig"),
        ("vllm.engine.arg_utils", "CacheConfig"),
        ("vllm.worker.cache_engine", "CacheConfig"),
    ]

    cache_config = None
    for module_path, class_name in possible_paths:
        cache_config = safe_import(module_path, class_name)
        if cache_config:
            result["module_path"] = f"{module_path}.{class_name}"
            print(f"✓ Found CacheConfig at: {result['module_path']}")
            break

    if not cache_config:
        print("✗ CacheConfig not found")
        return result

    # Get fields
    result["fields"] = get_class_fields(cache_config)
    print(f"✓ Fields ({len(result['fields'])}):")
    for field in result["fields"]:
        print(f"    - {field}")

    return result


def main():
    print("=" * 80)
    print("M0: vLLM 0.26.0 API Verification")
    print("=" * 80)
    print()

    # Run all checks
    results = {
        "vllm_version": None,
        "modelregistry": check_modelregistry(),
        "plugin_system": check_plugin_system(),
        "attention": check_attention(),
        "attention_metadata": check_attention_metadata(),
        "parallel_layers": check_parallel_layers(),
        "moe_support": check_moe_support(),
        "kv_cache_config": check_kv_cache_config(),
    }

    # Get vLLM version
    try:
        import vllm
        results["vllm_version"] = vllm.__version__
        print(f"\n✓ vLLM version: {vllm.__version__}")
    except Exception as e:
        print(f"\n✗ Could not determine vLLM version: {e}")

    # Save to JSON
    output_path = "api_check.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"✓ Results saved to: {output_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
