"""
Tests for plugin registration.
"""

import pytest


def test_register_function_exists():
    """Test that the register function can be imported."""
    from gaudi_vllm_accl.register import register
    assert callable(register)


def test_package_version():
    """Test that package version is defined."""
    from gaudi_vllm_accl import __version__
    assert __version__ == "0.1.0"


# TODO: Add more tests once model implementations are complete
# - Test model registration with vLLM
# - Test model loading and inference
# - Test acceleration features
