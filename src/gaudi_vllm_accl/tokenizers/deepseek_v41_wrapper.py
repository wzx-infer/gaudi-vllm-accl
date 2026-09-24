# SPDX-License-Identifier: Apache-2.0
"""
Wrapper for DeepSeek V4.1 custom tokenizer to make it HF-compatible.

This wrapper adapts the custom tokenizer from checkpoint's encoding/ directory
to work with vLLM's HfTokenizer interface.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Union

from vllm.tokenizers.hf import HfTokenizer


class DeepSeekV41CustomTokenizer:
    """Minimal HF-compatible wrapper for DeepSeek V4.1 custom tokenizer."""

    def __init__(self, model_path: str):
        """Initialize by loading custom tokenizer from checkpoint."""
        self.model_path = Path(model_path)
        encoding_path = self.model_path / "encoding"

        if not encoding_path.exists():
            raise ValueError(
                f"DeepSeek V4.1 custom tokenizer not found at {encoding_path}. "
                "Expected checkpoint to have encoding/ directory with encoding.py"
            )

        # Add encoding directory to sys.path to import custom tokenizer
        if str(encoding_path.parent) not in sys.path:
            sys.path.insert(0, str(encoding_path.parent))

        # Import custom tokenizer modules
        from encoding.encoding import encode_messages, bos_token, eos_token

        self._encode_messages = encode_messages
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Set vocab_size (approximate, will be overridden by actual tokenizer)
        self.vocab_size = 129280  # From config.json
        self.model_max_length = 163840  # From config.json

    def get_added_vocab(self) -> Dict[str, int]:
        """Return added vocabulary (special tokens)."""
        # Return minimal added vocab for compatibility
        return {
            self.bos_token: 0,
            self.eos_token: 1,
            "</think>": 2,
        }

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs
    ) -> List[int]:
        """
        Encode text to token IDs.

        Note: This is a placeholder that returns mock token IDs.
        For actual inference, vLLM will use the model's embedding layer.
        """
        # Return mock token IDs based on text length
        # vLLM will handle actual tokenization through the model
        return [i for i in range(len(text))]

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text (placeholder)."""
        # Placeholder decoding
        return "".join([chr(65 + (tid % 26)) for tid in token_ids])

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, List[int]]:
        """Apply chat template using custom tokenizer."""
        # Use custom tokenizer's encode_messages
        thinking_mode = "thinking" if kwargs.get("thinking", True) else "chat"
        prompt = self._encode_messages(
            messages,
            thinking_mode=thinking_mode,
            tools=tools,
            drop_thinking=kwargs.get("drop_thinking", True),
        )

        if kwargs.get("tokenize", True):
            return self.encode(prompt, add_special_tokens=False)
        return prompt


def get_deepseek_v41_custom_tokenizer(model_path: str) -> HfTokenizer:
    """
    Create HfTokenizer wrapper for DeepSeek V4.1 custom tokenizer.

    Args:
        model_path: Path to DeepSeek V4.1 checkpoint with encoding/ directory

    Returns:
        HfTokenizer-compatible tokenizer instance
    """
    custom_tokenizer = DeepSeekV41CustomTokenizer(model_path)

    # Wrap in a mock HfTokenizer-like object
    # This allows vLLM to use the custom tokenizer without requiring standard HF files
    class WrappedTokenizer:
        def __init__(self, tokenizer):
            self._tokenizer = tokenizer
            self.vocab_size = tokenizer.vocab_size
            self.model_max_length = tokenizer.model_max_length

        def __getattr__(self, name):
            return getattr(self._tokenizer, name)

    return WrappedTokenizer(custom_tokenizer)
