# SPDX-License-Identifier: Apache-2.0
"""Minimal fake tokenizer for DeepSeek V4.1 when checkpoint lacks standard HF tokenizer files."""


class FakeBaseTokenizer:
    """Minimal HF-compatible tokenizer that provides basic interface for DeepSeek V4.1."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.vocab_size = 129280  # From config.json
        self.model_max_length = 163840  # From config.json

    def get_added_vocab(self) -> dict[str, int]:
        """Return minimal added vocabulary."""
        return {
            "</think>": 129280,
            "<｜begin▁of▁sentence｜>": 129281,
            "<｜end▁of▁sentence｜>": 129282,
        }

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs
    ) -> list[int]:
        """Mock encoding - returns placeholder token IDs."""
        # Store last call for compatibility with vLLM's wrapper
        self.last_encode = (text, add_special_tokens, kwargs)
        # Return mock token IDs (actual tokenization happens in model's embedding layer)
        return list(range(min(len(text), 100)))

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Mock decoding."""
        return "".join([chr(65 + (tid % 26)) for tid in token_ids])


def create_fake_base_tokenizer(model_path: str) -> FakeBaseTokenizer:
    """
    Create a minimal fake tokenizer for DeepSeek V4.1.

    This is used when the checkpoint lacks standard HuggingFace tokenizer files.
    The fake tokenizer provides the basic interface needed by vLLM's tokenizer wrapper,
    while the actual prompt encoding is handled by the custom encoding logic.

    Args:
        model_path: Path to DeepSeek V4.1 checkpoint

    Returns:
        FakeBaseTokenizer instance
    """
    return FakeBaseTokenizer(model_path)
