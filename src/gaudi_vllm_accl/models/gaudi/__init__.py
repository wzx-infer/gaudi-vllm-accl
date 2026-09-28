"""
Gaudi-specific implementations for DeepSeek V4.1 Flash.
"""

# Import Phase 2 MLA implementation as the default
from .deepseek_v41_phase2 import DeepseekV41ForCausalLM

__all__ = ['DeepseekV41ForCausalLM']
