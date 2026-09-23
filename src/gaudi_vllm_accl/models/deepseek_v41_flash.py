"""
DeepSeek V4.1 Flash model implementation for vLLM on Intel Gaudi.

Based on the official DeepSeek V4.1 architecture with:
- Sparse MLA (Multi-head Latent Attention)
- MegaMoE with 384 routed experts + 1 shared expert
- FP8/FP4 quantization
- Special components: Engram, DSpark, KV compression
"""

from typing import Optional, List, Tuple, Iterable
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sequence import IntermediateTensors
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class DeepSeekV41Config(PretrainedConfig):
    """
    Configuration for DeepSeek V4.1 Flash model.
    This wraps the nested text_config from the original config.json.
    """

    model_type = "deepseek_v41"

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=5120,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        moe_intermediate_size=2304,
        n_routed_experts=384,
        n_shared_experts=1,
        num_experts_per_tok=6,
        max_position_embeddings=1048576,
        rope_theta=10000,
        rms_norm_eps=1e-20,
        initializer_range=0.02,
        use_cache=True,
        tie_word_embeddings=False,
        # MLA specific
        q_lora_rank=1280,
        qk_rope_head_dim=64,
        # KV compression
        compress_ratios=None,
        kv_source_layer_ids=None,
        # Engram
        engram_layer_ids=None,
        # DSpark
        dspark_target_layer_ids=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        # MLA
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # Advanced features
        self.compress_ratios = compress_ratios or []
        self.kv_source_layer_ids = kv_source_layer_ids or []
        self.engram_layer_ids = engram_layer_ids or []
        self.dspark_target_layer_ids = dspark_target_layer_ids or []

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained_config(cls, config: PretrainedConfig):
        """Create DeepSeekV41Config from HuggingFace config with nested text_config."""
        if hasattr(config, 'text_config'):
            text_config = config.text_config
            return cls(
                vocab_size=text_config.get('vocab_size', 129280),
                hidden_size=text_config.get('hidden_size', 5120),
                num_hidden_layers=text_config.get('num_hidden_layers', 40),
                num_attention_heads=text_config.get('num_attention_heads', 64),
                num_key_value_heads=text_config.get('num_key_value_heads', 1),
                head_dim=text_config.get('head_dim', 512),
                moe_intermediate_size=text_config.get('moe_intermediate_size', 2304),
                n_routed_experts=text_config.get('n_routed_experts', 384),
                n_shared_experts=text_config.get('n_shared_experts', 1),
                num_experts_per_tok=text_config.get('num_experts_per_tok', 6),
                max_position_embeddings=text_config.get('max_position_embeddings', 1048576),
                rope_theta=text_config.get('rope_theta', 10000),
                rms_norm_eps=text_config.get('rms_norm_eps', 1e-20),
                q_lora_rank=text_config.get('q_lora_rank', 1280),
                qk_rope_head_dim=text_config.get('qk_rope_head_dim', 64),
                compress_ratios=text_config.get('compress_ratios', []),
                kv_source_layer_ids=text_config.get('kv_source_layer_ids', []),
                engram_layer_ids=text_config.get('engram_layer_ids', []),
                dspark_target_layer_ids=text_config.get('dspark_target_layer_ids', []),
            )
        return cls(**config.to_dict())


class DeepSeekV41Attention(nn.Module):
    """
    Sparse MLA (Multi-head Latent Attention) for DeepSeek V4.1.

    TODO: Full MLA implementation requires:
    - Query LoRA projection (q_lora_rank)
    - Compressed KV cache
    - Sparse attention indexing
    - RoPE on partial dimensions (qk_rope_head_dim)

    For now, this is a placeholder that will raise NotImplementedError.
    """

    def __init__(self, config: DeepSeekV41Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        logger.warning(
            f"DeepSeekV41Attention (layer {layer_idx}): "
            "Sparse MLA is not yet fully implemented. "
            "This requires custom Triton kernels and sparse attention backend."
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Sparse MLA attention for DeepSeek V4.1 requires custom kernels. "
            "Full implementation pending."
        )


class DeepSeekV41MoE(nn.Module):
    """
    MegaMoE layer with 384 routed experts + 1 shared expert.

    TODO: Full MoE implementation requires:
    - Expert parallelism support
    - Top-K routing with proper normalization
    - FP4 expert weights
    - Load balancing

    For now, this is a placeholder.
    """

    def __init__(self, config: DeepSeekV41Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        logger.warning(
            f"DeepSeekV41MoE (layer {layer_idx}): "
            f"MegaMoE with {self.n_routed_experts} experts is not yet fully implemented."
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"MegaMoE with {self.n_routed_experts} experts requires FusedMoE implementation. "
            "Full implementation pending."
        )


class DeepSeekV41DecoderLayer(nn.Module):
    """Single transformer decoder layer for DeepSeek V4.1."""

    def __init__(self, config: DeepSeekV41Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = DeepSeekV41Attention(config, layer_idx)

        # MoE FFN
        self.mlp = DeepSeekV41MoE(config, layer_idx)

        # Layer norms (using RMSNorm in real implementation)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata = None,
    ) -> torch.Tensor:
        # Pre-attention norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention (will raise NotImplementedError for now)
        hidden_states = self.self_attn(
            hidden_states, position_ids, kv_cache, attn_metadata
        )
        hidden_states = residual + hidden_states

        # Pre-MLP norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MoE FFN (will raise NotImplementedError for now)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DeepSeekV41Model(nn.Module):
    """DeepSeek V4.1 transformer model."""

    def __init__(self, config: DeepSeekV41Config, vllm_config: Optional[VllmConfig] = None):
        super().__init__()
        self.config = config
        self.vllm_config = vllm_config

        # Embeddings
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            DeepSeekV41DecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                positions,
                kv_caches[i] if i < len(kv_caches) else None,
                attn_metadata,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states


class DeepSeekV41FlashForCausalLM(nn.Module):
    """
    DeepSeek V4.1 Flash model for causal language modeling.

    This is a PROGRESSIVE implementation:
    - Stage 1 (current): Basic structure, config parsing, model registration
    - Stage 2 (next): Implement core attention and MoE with vLLM primitives
    - Stage 3 (future): Add advanced features (Engram, DSpark, full FP8)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config=None,
        lora_config=None,
        vllm_config: Optional[VllmConfig] = None,
    ):
        super().__init__()

        # Convert to DeepSeekV41Config if needed
        if not isinstance(config, DeepSeekV41Config):
            config = DeepSeekV41Config.from_pretrained_config(config)

        self.config = config
        self.vllm_config = vllm_config

        # Model
        self.model = DeepSeekV41Model(config, vllm_config)

        # LM head
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
        )

        # Logits processor
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
        )

        logger.info(
            f"Initialized DeepSeekV41FlashForCausalLM: "
            f"{config.num_hidden_layers} layers, "
            f"{config.n_routed_experts} experts, "
            f"{config.num_experts_per_tok} active experts per token"
        )

        logger.warning(
            "This is a work-in-progress implementation. "
            "Core components (Sparse MLA, MegaMoE) are not yet functional."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the model."""
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
        sampling_metadata,
    ) -> torch.Tensor:
        """Compute logits from hidden states."""
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            sampling_metadata,
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        """Sample next tokens from logits."""
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        """Load model weights."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


# Register the model architecture name
DeepseekV41ForCausalLM = DeepSeekV41FlashForCausalLM
