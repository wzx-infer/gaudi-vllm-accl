"""
DeepSeek V4.1 KV Compressor for Gaudi - Phase 2.1

Simplified compressor implementation using standard PyTorch operations.
Based on PR #56201, adapted for Intel Gaudi hardware.

TODO: Optimize with Gaudi-specific kernels in future iterations.
"""

from typing import Optional
import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear


class DeepseekCompressor(nn.Module):
    """
    DeepSeek V4.1 KV compressor for Gaudi.

    Pools consecutive tokens into compressed KV latents:
    - compress_ratio=1: Full-length (no pooling, single-token compression)
    - compress_ratio=2: 2:1 pooling (2 tokens -> 1 compressed latent)

    Architecture (from PR #56201):
    - hidden -> fused_wkv_wgate -> [kv_latent, gate_score] (if ratio > 1)
    - Pooling: softmax(gate) weighted sum over compress_ratio tokens
    - Output: normalized kv_latent in BF16
    """

    def __init__(
        self,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        prefix: str = "",
    ):
        super().__init__()

        if compress_ratio not in (1, 2):
            raise NotImplementedError(
                f"DeepSeek V4.1 compressor supports compress_ratio 1 and 2, "
                f"got {compress_ratio}. Ratio 4/128 are v4.0-specific."
            )

        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.prefix = prefix

        # Ratio 1: no gate (single token compression)
        # Ratio > 1: has gate for weighted pooling
        self.has_gate = compress_ratio > 1

        # Fused KV and gate projection
        wkv_wgate_sizes = (
            [self.head_dim, self.head_dim] if self.has_gate else [self.head_dim]
        )
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            wkv_wgate_sizes,
            bias=False,
            quant_config=None,
            disable_tp=True,  # Replicated across TP ranks
            prefix=f"{prefix}.fused_wkv_wgate",
        )

        # KV normalization
        self.norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch * seq_len, hidden_size]
        positions: Optional[torch.Tensor] = None,  # [batch * seq_len]
    ) -> torch.Tensor:
        """
        Compress hidden states into KV latents.

        Args:
            hidden_states: Input tokens [num_tokens, hidden_size]
            positions: Token positions [num_tokens] (optional, for group boundary detection)

        Returns:
            Compressed KV latents [num_compressed_tokens, head_dim] in BF16
        """
        num_tokens = hidden_states.shape[0]

        # Project to KV (and optionally gate)
        kv_score, _ = self.fused_wkv_wgate(hidden_states)

        if not self.has_gate:
            # Ratio 1: no pooling, just normalize
            kv = kv_score  # [num_tokens, head_dim]
            compressed_kv = self.norm(kv)
        else:
            # Ratio > 1: split KV and gate, then pool
            kv, gate = kv_score.split([self.head_dim, self.head_dim], dim=-1)

            # Simplified pooling for Phase 2.1 (Gaudi-compatible)
            # TODO: Replace with fused kernel in Phase 2.2
            compressed_kv = self._pool_with_gate(kv, gate, positions)

        # Convert to BF16 for cache storage
        return compressed_kv.to(torch.bfloat16)

    def _pool_with_gate(
        self,
        kv: torch.Tensor,  # [num_tokens, head_dim]
        gate: torch.Tensor,  # [num_tokens, head_dim]
        positions: Optional[torch.Tensor],  # [num_tokens]
    ) -> torch.Tensor:
        """
        Pool tokens using gated weighted sum.

        Simplified implementation for Phase 2.1:
        - Groups consecutive tokens by compress_ratio
        - Softmax gate over group
        - Weighted sum KV
        - RMSNorm output

        TODO Phase 2.2: Replace with Gaudi-optimized fused kernel
        """
        num_tokens = kv.shape[0]

        # Pad to multiple of compress_ratio
        pad_size = (self.compress_ratio - num_tokens % self.compress_ratio) % self.compress_ratio
        if pad_size > 0:
            kv = torch.cat([kv, torch.zeros(pad_size, self.head_dim, device=kv.device, dtype=kv.dtype)], dim=0)
            gate = torch.cat([gate, torch.zeros(pad_size, self.head_dim, device=gate.device, dtype=gate.dtype)], dim=0)

        # Reshape to [num_groups, compress_ratio, head_dim]
        num_groups = kv.shape[0] // self.compress_ratio
        kv = kv.view(num_groups, self.compress_ratio, self.head_dim)
        gate = gate.view(num_groups, self.compress_ratio, self.head_dim)

        # Softmax gate over compress_ratio dimension
        gate_weights = torch.softmax(gate, dim=1)  # [num_groups, compress_ratio, head_dim]

        # Weighted sum
        compressed_kv = (kv * gate_weights).sum(dim=1)  # [num_groups, head_dim]

        # Normalize
        compressed_kv = self.norm(compressed_kv)

        return compressed_kv


def get_compress_ratio(layer_idx: int, config) -> int:
    """
    Get compression ratio for a given layer.

    From config.yaml:
    - First 2 layers (0, 1): ratio=0 (disabled, use full-length)
    - Rest: ratio=2 (2:1 compression)

    Note: ratio=0 is treated as ratio=1 (full-length but compressed)
    """
    if hasattr(config, 'compress_ratio'):
        if isinstance(config.compress_ratio, list):
            # Per-layer ratios
            ratio = config.compress_ratio[layer_idx] if layer_idx < len(config.compress_ratio) else 1
        else:
            # Global ratio
            ratio = config.compress_ratio
    else:
        # Default: first 2 layers ratio=1, rest ratio=2
        ratio = 1 if layer_idx < 2 else 2

    # ratio=0 means disabled (treat as ratio=1 for full-length)
    return 1 if ratio == 0 else ratio
