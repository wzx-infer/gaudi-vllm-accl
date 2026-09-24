"""
Plugin registration entry point for vLLM.

This module is invoked by vLLM through the entry point mechanism defined in pyproject.toml.
It registers:
1. Transformers config (so vLLM can load the config)
2. New model architectures (e.g., DeepSeek V4.1 Flash)
3. Acceleration features (attention, MoE optimizations)
"""

from typing import Optional
import logging
import sys

logger = logging.getLogger(__name__)

# Required vLLM version
REQUIRED_VLLM_VERSION = "0.26.0"


def _check_vllm_version() -> None:
    """Verify that the correct vLLM version is installed."""
    try:
        import vllm
    except ImportError as e:
        raise ImportError(
            f"vLLM is not installed. This plugin requires vLLM {REQUIRED_VLLM_VERSION} "
            f"or vllm-gaudi {REQUIRED_VLLM_VERSION}. "
            f"Please install the correct version before loading this plugin."
        ) from e

    installed_version = getattr(vllm, "__version__", None)

    if installed_version is None:
        logger.warning(
            "Could not determine vLLM version. "
            f"This plugin is designed for vLLM {REQUIRED_VLLM_VERSION}. "
            "Compatibility is not guaranteed."
        )
        return

    if installed_version != REQUIRED_VLLM_VERSION:
        logger.warning(
            f"vLLM version mismatch: installed={installed_version}, "
            f"required={REQUIRED_VLLM_VERSION}. Proceeding anyway."
        )

    logger.info(f"vLLM version check passed: {installed_version}")


def register() -> None:
    """Main plugin registration function called by vLLM."""
    logger.info("Registering gaudi-vllm-accl plugin...")

    # Step 0: Verify vLLM version before any registration
    _check_vllm_version()

    # Part A: Register Transformers config (MUST be first)
    _register_transformers_config()

    # Part B: Register new models
    _register_models()

    # Part C: Register acceleration features
    _register_accel()

    logger.info("gaudi-vllm-accl plugin registered successfully.")


def _register_transformers_config() -> None:
    """
    Register custom Transformers configs with AutoConfig.

    This MUST happen before vLLM tries to load model configs, otherwise
    Transformers will fail to recognize our custom model types.
    """
    try:
        from transformers import AutoConfig
        from gaudi_vllm_accl.models.deepseek_v41_flash import DeepSeekV41Config

        AutoConfig.register("deepseek_v41", DeepSeekV41Config)
        AutoConfig.register("deepseek_v41_text", DeepSeekV41Config)

        logger.info("Registered DeepSeek V4.1 config with Transformers AutoConfig")
    except Exception as e:
        logger.warning(f"Failed to register Transformers config: {e}")


def _register_models() -> None:
    """Register new model architectures with vLLM's ModelRegistry."""
    try:
        from vllm.model_executor.models import ModelRegistry
    except ImportError:
        logger.warning(
            "Could not import ModelRegistry from vLLM. "
            "Model registration skipped. Is vLLM installed?"
        )
        return

    # Register DeepSeek V4.1 Flash model
    try:
        # Use string format to avoid CUDA initialization on import
        ModelRegistry.register_model(
            "DeepseekV41ForCausalLM",
            "gaudi_vllm_accl.models.gaudi.model:DeepseekV41ForCausalLM"
        )
        logger.info("Registered model: DeepseekV41ForCausalLM")
    except Exception as e:
        logger.error(f"Failed to register DeepseekV41ForCausalLM: {e}")


def _register_accel() -> None:
    """Register acceleration features (placeholder for future)."""
    logger.info("Acceleration features registration: not yet implemented")
