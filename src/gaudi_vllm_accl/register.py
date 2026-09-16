"""
Plugin registration entry point for vLLM.

This module is invoked by vLLM through the entry point mechanism defined in pyproject.toml.
It registers:
1. New model architectures (e.g., DeepSeek V4.1 Flash)
2. Acceleration features (attention, MoE optimizations)
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """
    Main plugin registration function called by vLLM.

    This function is invoked automatically when vLLM loads general plugins.
    It registers all models and acceleration features provided by this package.
    """
    logger.info("Registering gaudi-vllm-accl plugin...")

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
