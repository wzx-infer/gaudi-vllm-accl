"""
Plugin registration entry point for vLLM.

This module is invoked by vLLM through the entry point mechanism defined in pyproject.toml.
It registers:
1. New model architectures (e.g., DeepSeek V4.1 Flash)
2. Acceleration features (attention, MoE optimizations)
"""

from typing import Optional
import logging
import sys

logger = logging.getLogger(__name__)

# Required vLLM version
REQUIRED_VLLM_VERSION = "0.26.0"


def _check_vllm_version() -> None:
    """
    Verify that the correct vLLM version is installed.

    Raises:
        ImportError: If vLLM is not installed
        RuntimeError: If vLLM version doesn't match required version
    """
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
        raise RuntimeError(
            f"vLLM version mismatch!\n"
            f"  Required: {REQUIRED_VLLM_VERSION}\n"
            f"  Installed: {installed_version}\n"
            f"This plugin is specifically designed for vLLM {REQUIRED_VLLM_VERSION} "
            f"(or vllm-gaudi {REQUIRED_VLLM_VERSION}). "
            f"Using a different version may cause API incompatibilities or runtime errors.\n"
            f"Please install the correct version:\n"
            f"  pip install vllm-gaudi=={REQUIRED_VLLM_VERSION}"
        )

    logger.info(f"vLLM version check passed: {installed_version}")


def register() -> None:
    """
    Main plugin registration function called by vLLM.

    This function is invoked automatically when vLLM loads general plugins.
    It registers all models and acceleration features provided by this package.

    Raises:
        ImportError: If vLLM is not installed
        RuntimeError: If vLLM version doesn't match required version
    """
    logger.info("Registering gaudi-vllm-accl plugin...")

    # Step 0: Verify vLLM version before any registration
    _check_vllm_version()

    # Part A: Register new models
    _register_models()

    # Part B: Register acceleration features
    _register_accel()

    logger.info("gaudi-vllm-accl plugin registered successfully.")


def _register_models() -> None:
    """
    Register new model architectures with vLLM's ModelRegistry.

    Models registered here become available for use in vLLM without modifying
    upstream code. The registration maps model class names (from config.json)
    to implementation classes in this package.
    """
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
        from gaudi_vllm_accl.models.deepseek_v41_flash import DeepSeekV41FlashForCausalLM

        ModelRegistry.register_model(
            "DeepSeekV41FlashForCausalLM",
            DeepSeekV41FlashForCausalLM
        )
        logger.info("Registered model: DeepSeekV41FlashForCausalLM")
    except Exception as e:
        logger.error(f"Failed to register DeepSeekV41FlashForCausalLM: {e}")

    # TODO: Add more model registrations here as needed
    # Example:
    # ModelRegistry.register_model(
    #     "AnotherModelForCausalLM",
    #     AnotherModelForCausalLM
    # )


def _register_accel() -> None:
    """
    Register acceleration features (attention, MoE optimizations).

    This is a placeholder for future acceleration feature injection.
    Acceleration features will be registered through layer replacement or
    other injection mechanisms provided by vLLM, NOT by monkey-patching
    upstream code.

    Future implementation might include:
    - Custom attention implementations for Gaudi
    - Optimized MoE routing and expert parallelism
    - Kernel fusion optimizations
    """
    # TODO: Implement acceleration feature registration
    # This will depend on vLLM's extension points for custom kernels/layers

    logger.info("Acceleration features registration: not yet implemented")

    # Placeholder for future implementation:
    # - Register custom attention via vLLM's attention backend system
    # - Register MoE optimizations via layer replacement hooks
    # - Do NOT monkey-patch upstream vLLM-Gaudi code
