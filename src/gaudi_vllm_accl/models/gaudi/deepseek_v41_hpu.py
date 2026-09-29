"""DeepSeek V4.1 Flash - Phase 0 HPU implementation with checkpoint-aligned weight naming."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from vllm.config import VllmConfig, CacheConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm_gaudi.attention.backends.hpu_attn import HPUAttentionMetadata

from gaudi_vllm_accl.models.deepseek_v41_flash import DeepSeekV41Config


class DeepseekV41Attention(nn.Module):
    """Phase 0: Pure PyTorch attention matching checkpoint weight names."""
    
    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size
        
        self.scaling = self.head_dim ** -0.5
        
        # Checkpoint naming: fused_wqa_wkv = [wq_a; wkv]
        fused_dim = self.q_lora_rank + self.kv_lora_rank + self.rope_head_dim
        self.fused_wqa_wkv = nn.Linear(self.hidden_size, fused_dim, bias=False)
        
        # Norms
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        
        # Q projection: q_lora_rank -> num_heads * head_dim (TP sharded)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_local_heads * self.head_dim, bias=False)
        
        # O projection (LoRA style)
        self.o_lora_rank = getattr(config, 'o_lora_rank', self.kv_lora_rank)
        self.wo_a = nn.Linear(self.num_local_heads * self.v_head_dim, self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.o_lora_rank, self.hidden_size, bias=False)
        
        # Attention sink
        self.attn_sink = nn.Parameter(torch.zeros(1, self.num_local_heads, 1, self.v_head_dim))
        
        # RoPE
        rope_params = getattr(config, "rope_parameters", None) or {}
        if not isinstance(rope_params, dict):
            rope_params = {}
        rope_params.setdefault("rope_type", "default")
        self.rotary_emb = get_rope(
            self.rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_parameters=rope_params,
            is_neox_style=False,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
    ) -> torch.Tensor:
        # Fused projection
        fused = self.fused_wqa_wkv(hidden_states)
        
        # Split: [wq_a, kv_a, rope_k]
        wq_a = fused[:, :self.q_lora_rank]
        kv_a = fused[:, self.q_lora_rank:self.q_lora_rank + self.kv_lora_rank]
        rope_k = fused[:, self.q_lora_rank + self.kv_lora_rank:]
        
        # Q path
        wq_a = self.q_norm(wq_a)
        q = self.wq_b(wq_a)
        q = q.view(-1, self.num_local_heads, self.head_dim)
        
        # Split Q into nope and rope parts
        q_nope = q[:, :, :self.nope_head_dim]
        q_rope = q[:, :, self.nope_head_dim:]
        q_rope = self.rotary_emb(positions, q_rope)
        q = torch.cat([q_nope, q_rope], dim=-1)
        
        # KV path
        kv_a = self.kv_norm(kv_a)
        # Phase 0: Simplified - replicate KV across heads
        kv_expanded = kv_a.unsqueeze(1).expand(-1, self.num_local_heads, -1)
        
        # K: nope + rope
        k_nope = kv_expanded[:, :, :self.nope_head_dim]
        k_rope = rope_k.view(-1, 1, self.rope_head_dim).expand(-1, self.num_local_heads, -1)
        k_rope = self.rotary_emb(positions, k_rope)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
        # V
        v = kv_expanded[:, :, self.nope_head_dim:self.nope_head_dim + self.v_head_dim]
        
        # Attention (Phase 0: simple scaled dot-product)
        attn_output = self._simple_attention(q, k, v, kv_cache, attn_metadata)
        
        # Output projection
        attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
        out = self.wo_a(attn_output)
        out = tensor_model_parallel_all_reduce(out)
        out = self.wo_b(out)
        
        return out
    
    def _simple_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
    ) -> torch.Tensor:
        """Phase 0: Simple PyTorch attention without PagedAttention."""
        # q, k, v: [tokens, num_local_heads, dim]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output


class DeepseekV41MoE(nn.Module):
    """Phase 0: Only shared experts."""
    
    def __init__(self, config: DeepSeekV41Config, layer_idx: int):
        super().__init__()
        from vllm.model_executor.layers.activation import SiluAndMul
        
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        
        # Shared experts only
        self.shared_gate_up = nn.Linear(self.hidden_size, 2 * self.moe_intermediate_size, bias=False)
        self.shared_down = nn.Linear(self.moe_intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiluAndMul()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.shared_gate_up(hidden_states)
        x = self.act_fn(gate_up)
        x = self.shared_down(x)
        return x


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV41Attention(
            config, layer_idx, cache_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = DeepseekV41MoE(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states, kv_cache, attn_metadata)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class DeepseekV41Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: DeepSeekV41Config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.layers = nn.ModuleList([
            DeepseekV41DecoderLayer(
                config, i, cache_config, prefix=f"{prefix}.layers.{i}"
            )
            for i in range(self.start_layer, self.end_layer)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: HPUAttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions, hidden_states, kv_caches[i], attn_metadata, residual
            )
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV41ForCausalLM(nn.Module, SupportsPP):
    is_text_generation_model = True
    is_pooling_model = False
    supports_multimodal = False
    supports_pp = True
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: DeepSeekV41Config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.model = DeepseekV41Model(vllm_config=vllm_config, prefix=f"{prefix}.model")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: HPUAttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, inputs_embeds
        )
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
    
    def load_weights(self, weights):
        """Load weights from checkpoint with FP8 dequantization."""
        params_dict = dict(self.named_parameters())
        
        # Build weight mapping: checkpoint name -> model parameter name
        weight_map = {}
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            # Attention weights
            weight_map[f"layers.{layer_idx}.attn.wq_a.weight"] = f"{prefix}.self_attn.fused_wqa_wkv.weight"
            weight_map[f"layers.{layer_idx}.attn.wkv.weight"] = f"{prefix}.self_attn.fused_wqa_wkv.weight"
            weight_map[f"layers.{layer_idx}.attn.wq_b.weight"] = f"{prefix}.self_attn.wq_b.weight"
            weight_map[f"layers.{layer_idx}.attn.wo_a.weight"] = f"{prefix}.self_attn.wo_a.weight"
            weight_map[f"layers.{layer_idx}.attn.wo_b.weight"] = f"{prefix}.self_attn.wo_b.weight"
            weight_map[f"layers.{layer_idx}.attn.q_norm.weight"] = f"{prefix}.self_attn.q_norm.weight"
            weight_map[f"layers.{layer_idx}.attn.kv_norm.weight"] = f"{prefix}.self_attn.kv_norm.weight"
            weight_map[f"layers.{layer_idx}.attn.attn_sink"] = f"{prefix}.self_attn.attn_sink"
            # MLP weights
            weight_map[f"layers.{layer_idx}.ffn.shared_experts.w1.weight"] = f"{prefix}.mlp.shared_gate_up.weight"
            weight_map[f"layers.{layer_idx}.ffn.shared_experts.w2.weight"] = f"{prefix}.mlp.shared_down.weight"
            # Norms
            weight_map[f"layers.{layer_idx}.attn_norm.weight"] = f"{prefix}.input_layernorm.weight"
            weight_map[f"layers.{layer_idx}.ffn_norm.weight"] = f"{prefix}.post_attention_layernorm.weight"
        
        # Global weights
        weight_map["embed.weight"] = "model.embed_tokens.weight"
        weight_map["norm.weight"] = "model.norm.weight"
        weight_map["head.weight"] = "lm_head.weight"
        
        for name, loaded_weight in weights:
            # Skip multimodal and unused components
            if any(skip in name for skip in ["vision.", "aligner.", "mtp.", "image_", ".scale"]):
                continue
            
            # Map checkpoint name to model parameter
            if name in weight_map:
                param_name = weight_map[name]
                if param_name in params_dict:
                    param = params_dict[param_name]
                    # Handle FP8 dequantization
                    if loaded_weight.dtype == torch.float8_e4m3fn:
                        loaded_weight = loaded_weight.to(torch.bfloat16)
                    param.data.copy_(loaded_weight)
