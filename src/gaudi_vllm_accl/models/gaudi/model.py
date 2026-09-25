"""
DeepSeek V4.1 Flash model implementation for Intel Gaudi.

Phase 1: BF16 baseline with simplified architecture.
Based on PR #56201, adapted for Gaudi hardware.
"""

from typing import Optional, List, Tuple, Iterable
import torch
import torch.nn as nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    make_layers,
    PPMissingLayer,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class DeepseekV41Attention(nn.Module):
    """Simplified MLA attention for Phase 1.

    TODO Phase 2: Replace with CSA2 sparse MLA from PR #56201.
    """

    def __init__(self, config, vllm_config: VllmConfig, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.q_lora_rank = config.q_lora_rank
        self.layer_idx = layer_idx

        # Phase 1: Use standard QKV projection
        # TODO: Replace with MLA's compressed KV + LoRA Q
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            config.num_key_value_heads,
            bias=False,
        )

        # Standard attention
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim ** -0.5,
            num_kv_heads=config.num_key_value_heads,
            prefix=prefix,
        )

        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # RoPE
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": config.rope_theta,
            },
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Attention
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV41MoE(nn.Module):
    """Simplified MoE for Phase 1.

    Phase 1: Dense FFN to get model running.
    Phase 2: Full MoE with 384 routed experts + 1 shared expert.
    """

    def __init__(self, config, vllm_config: VllmConfig, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.layer_idx = layer_idx

        # Phase 1: Simple dense FFN (SwiGLU activation)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )

        if layer_idx == 0:
            logger.warning(
                f"Phase 1: Using dense FFN. Full MoE with {self.n_routed_experts} "
                f"experts will be added in Phase 2."
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate(x) * silu(up(x))
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        intermediate = torch.nn.functional.silu(gate) * up
        output, _ = self.down_proj(intermediate)
        return output


class DeepseekV41DecoderLayer(nn.Module):
    """Decoder layer with simplified architecture for Phase 1.

    Phase 1: Standard pre-norm transformer (no CED hyperconnections).
    Phase 2: Add CED multi-stream hyperconnections from PR #56201.
    """

    def __init__(self, config, vllm_config: VllmConfig, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = DeepseekV41Attention(
            config, vllm_config, layer_idx, prefix=f"{prefix}.attn"
        )

        # FFN (MoE)
        self.mlp = DeepseekV41MoE(
            config, vllm_config, layer_idx, prefix=f"{prefix}.mlp"
        )

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # Pre-attention norm + attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states

        # Pre-FFN norm + FFN + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DeepseekV41Model(nn.Module):
    """Main transformer model for Phase 1."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.vocab_size = config.vocab_size

        # Embeddings
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Decoder layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, idx=None: DeepseekV41DecoderLayer(
                config,
                vllm_config,
                int(prefix.split(".")[-1]) if idx is None else idx,
                prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        # Final norm
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        # Embedding
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # Transformer layers
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
            )

        # Final norm
        if get_pp_group().is_last_rank:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states": torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=dtype,
                device=device,
            )
        })


class DeepseekV41ForCausalLM(nn.Module, SupportsPP):
    """
    DeepSeek V4.1 Flash model for causal LM.

    Phase 1: BF16 baseline with standard components.
    Phase 2: Add CSA2 sparse attention, full MoE, CED, Engram.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # Phase 1: Text-only model, but checkpoint contains multimodal components
        # Add placeholder modules to receive weights (not used in forward pass)
        if hasattr(config, "vision_n_heads"):
            # Multimodal checkpoint: add placeholders for vision/aligner weights
            self.vision = nn.ModuleDict()  # Placeholder for vision tower weights
            self.aligner = nn.ModuleDict()  # Placeholder for aligner weights
            self.image_start = nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))
            logger.info("Added placeholders for multimodal weights (Phase 1: text-only)")

        # Model
        self.model = DeepseekV41Model(vllm_config, prefix=maybe_prefix(prefix, "model"))

        # LM head
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        logger.info(
            f"Initialized DeepseekV41ForCausalLM (Phase 1 - BF16 baseline): "
            f"{config.num_hidden_layers} layers, "
            f"hidden_size={config.hidden_size}, "
            f"vocab_size={config.vocab_size}"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs."""
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            sampling_metadata,
        )
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states": torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=dtype,
                device=device,
            )
        })
