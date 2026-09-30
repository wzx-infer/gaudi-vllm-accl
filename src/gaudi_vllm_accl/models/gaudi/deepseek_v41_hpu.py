"""DeepSeek V4.1 Flash - Phase 1 HPU implementation with correct architecture."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from vllm.config import VllmConfig, CacheConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mla import MLAAttention
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm_gaudi.attention.backends.hpu_attn import HPUAttentionMetadata

from gaudi_vllm_accl.models.deepseek_v41_flash import DeepSeekV41Config


class DeepseekV41Attention(MLAAttention):
    """Phase 1: Correct MLA architecture - wq_a and wkv are separate."""
    
    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        vllm_config: Optional[VllmConfig] = None,
    ):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.prefix = prefix
        # 注册到 static_forward_context，让 vLLM 能遍历到
        if vllm_config is not None and prefix:
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim  # 512 = nope(448) + rope(64)
        self.head_size = self.head_dim  # alias expected by vllm_gaudi MLAAttention isinstance branch
        self.rope_head_dim = config.qk_rope_head_dim  # 64
        self.nope_head_dim = self.head_dim - self.rope_head_dim  # 448
        self.v_head_dim = config.head_dim  # MLA: v_head_dim == head_dim == 512
        self.q_lora_rank = config.q_lora_rank  # 1280
        self.num_kv_heads = config.num_key_value_heads  # 1
        
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size
        
        self.scaling = self.head_dim ** -0.5
        
        # Correct: wq_a and wkv are TWO SEPARATE weights
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        
        # Norms
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # Q projection: q_lora_rank -> num_heads * head_dim (TP sharded)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_local_heads * self.head_dim, bias=False)
        
        # O projection (LoRA style with groups)
        self.o_lora_rank = getattr(config, 'o_lora_rank', 1024)
        self.o_groups = getattr(config, 'o_groups', 8)
        assert self.o_groups % tp_size == 0
        self.num_local_groups = self.o_groups // tp_size
        
        # wo_a: 8 groups, each group processes [num_heads/8 * v_head_dim] -> o_lora_rank
        # Input: [num_local_heads * v_head_dim] = [64/8 * 512] = [4096] when TP=8
        # Output: [num_local_groups * o_lora_rank] = [1 * 1024] = [1024] when TP=8
        #         or [8 * 1024] = [8192] when TP=1
        self.wo_a = nn.Linear(
            self.num_local_heads * self.v_head_dim, 
            self.num_local_groups * self.o_lora_rank, 
            bias=False
        )
        self.wo_b = nn.Linear(self.num_local_groups * self.o_lora_rank, self.hidden_size, bias=False)
        
        # Attention sink: [num_local_heads] per-head scalars
        self.attn_sink = nn.Parameter(torch.zeros(self.num_local_heads))
        
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
    
    def get_attn_backend(self):
        """AttentionLayerBase abstract method."""
        from vllm_gaudi.attention.backends.hpu_attn import HPUMLAAttentionBackend
        return HPUMLAAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        """Phase 1.1: minimal spec; attention maintains its own KV cache."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        try:
            with open("/tmp/dsv41_debug.log", "a") as _f:
                _f.write("[DSV41] get_kv_cache_spec -> FullAttentionSpec" + chr(10))
        except Exception:
            pass
        return FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=vllm_config.model_config.dtype,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Phase 1: our layer manages its own weights; skip MLAAttention real post-load step."""
        pass

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(self.prefix)

        # Phase 1: flatten any leading batch/seq dims to handle both [T,H] and [B,S,H] inputs
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # -> [num_tokens, hidden_size]
        positions = positions.view(-1)  # -> [num_tokens]

        # Separate projections
        qr = self.wq_a(hidden_states)  # [T, 1280]
        kv = self.wkv(hidden_states)   # [T, 512]
        
        # Normalize
        qr = self.q_norm(qr)  # [T, 1280]
        kv = self.kv_norm(kv)  # [T, 512]
        
        # Q projection
        q = self.wq_b(qr)  # [T, num_local_heads * head_dim]
        q = q.view(-1, self.num_local_heads, self.head_dim)  # [T, 8, 512]
        
        # KV: replicate to all heads (num_kv_heads=1)
        kv = kv.unsqueeze(1).expand(-1, self.num_local_heads, -1)  # [T, 64/TP, 512]
        
        # RoPE: apply to the LAST 64 dims [448:512]
        q_nope = q[:, :, :self.nope_head_dim]  # [T, 64/TP, 448]
        q_rope = q[:, :, self.nope_head_dim:]  # [T, 64/TP, 64]
        
        k_nope = kv[:, :, :self.nope_head_dim]  # [T, 64/TP, 448]
        k_rope = kv[:, :, self.nope_head_dim:]  # [T, 64/TP, 64]
        
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
        q = torch.cat([q_nope, q_rope], dim=-1)  # [T, 64/TP, 512]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [T, 64/TP, 512]
        
        # V: same as kv (for Phase 1, v_head_dim == head_dim == 512)
        v = kv  # [T, 64/TP, 512]
        
        # Attention
        attn_output = self._simple_attention(q, k, v, attn_metadata)  # [T, 64/TP, 512]
        
        # Inverse RoPE on output (optional for Phase 1, can skip)
        # o_nope = attn_output[:, :, :self.nope_head_dim]
        # o_rope = attn_output[:, :, self.nope_head_dim:]
        # o_rope = self.rotary_emb.inverse(positions, o_rope)
        # attn_output = torch.cat([o_nope, o_rope], dim=-1)
        
        # Output projection
        attn_output = attn_output.contiguous().view(-1, self.num_local_heads * self.v_head_dim)  # [T, 4096] when TP=8
        out = self.wo_a(attn_output)  # [T, 1024] when TP=8
        out = self.wo_b(out)  # [T, 5120]
        out = tensor_model_parallel_all_reduce(out)  # [T, 1024]
        
        # Restore original leading shape
        out = out.view(*orig_shape[:-1], out.shape[-1])
        return out
    
    def _simple_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: Optional[HPUAttentionMetadata],
    ) -> torch.Tensor:
        """Phase 1: Simple PyTorch attention with causal mask, no PagedAttention."""
        # q, k, v: [T, H, D]  →  transpose to [H, T, D]
        q = q.transpose(0, 1)   # [H, T, D]
        k = k.transpose(0, 1)   # [H, T, D]
        v = v.transpose(0, 1)   # [H, T, D]

        T = q.shape[1]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [H, T, T]

        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=q.device, dtype=attn_weights.dtype),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask  # [H, T, T]
        attn_weights = attn_weights + self.attn_sink.view(-1, 1, 1)  # [H, 1, 1] broadcast

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [H, T, D]
        attn_output = attn_output.transpose(0, 1)   # [T, H, D]
        return attn_output


class DeepseekV41MoE(nn.Module):
    """Phase 1: Only shared experts, no routed experts."""
    
    def __init__(self, config: DeepSeekV41Config, layer_idx: int):
        super().__init__()
        from vllm.model_executor.layers.activation import SiluAndMul
        
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        
        # Shared experts only (gate is ignored in Phase 1)
        self.gate = nn.Linear(self.hidden_size, config.n_routed_experts, bias=False)
        self.w1 = nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(self.moe_intermediate_size, self.hidden_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False)
        self.act_fn = SiluAndMul()
    


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Phase 1: Only use shared experts
        x1 = self.w1(hidden_states)
        x3 = self.w3(hidden_states)
        gate_up = torch.cat([x1, x3], dim=-1)
        x = self.act_fn(gate_up)
        x = self.w2(x)
        return x


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        vllm_config: Optional[VllmConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV41Attention(
            config, layer_idx, cache_config,
            prefix=f"{prefix}.self_attn",
            vllm_config=vllm_config,
        )
        self.mlp = DeepseekV41MoE(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
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
                config, i, cache_config,
                prefix=f"{prefix}.layers.{i}",
                vllm_config=vllm_config,
            )
            for i in range(self.start_layer, self.end_layer)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        with open("/tmp/dsv41_debug.log", "a") as _f:
            _ctx = vllm_config.compilation_config.static_forward_context
            _f.write(f"[DSV41] Model.__init__ done, ctx has {len(_ctx)} entries\n")
            for _k in list(_ctx.keys())[:8]:
                _f.write(f"[DSV41]   ctx[{_k}] = {type(_ctx[_k]).__name__}\n")
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
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
                positions, hidden_states, residual
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
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=f"{prefix}.lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
    

    def load_weights(self, weights):
        """Load weights from checkpoint with FP8 dequantization."""
        weights = list(weights)   # FIX: weights 是生成器，必须缓存成 list，否则第二次遍历为空
        params_dict = dict(self.named_parameters())
        
        # Cache all .scale tensors for FP8 dequantization
        scale_cache = {}
        for name, loaded_weight in weights:
            if name.endswith(".scale"):
                scale_cache[name[:-6]] = loaded_weight  # Remove ".scale" suffix
        
        # Build weight mapping
        weight_map = {}
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            # Attention weights - SEPARATE wq_a and wkv
            weight_map[f"layers.{layer_idx}.attn.wq_a.weight"] = f"{prefix}.self_attn.wq_a.weight"
            weight_map[f"layers.{layer_idx}.attn.wkv.weight"] = f"{prefix}.self_attn.wkv.weight"
            weight_map[f"layers.{layer_idx}.attn.wq_b.weight"] = f"{prefix}.self_attn.wq_b.weight"
            weight_map[f"layers.{layer_idx}.attn.wo_a.weight"] = f"{prefix}.self_attn.wo_a.weight"
            weight_map[f"layers.{layer_idx}.attn.wo_b.weight"] = f"{prefix}.self_attn.wo_b.weight"
            weight_map[f"layers.{layer_idx}.attn.q_norm.weight"] = f"{prefix}.self_attn.q_norm.weight"
            weight_map[f"layers.{layer_idx}.attn.kv_norm.weight"] = f"{prefix}.self_attn.kv_norm.weight"
            weight_map[f"layers.{layer_idx}.attn.attn_sink"] = f"{prefix}.self_attn.attn_sink"
            # MLP weights
            weight_map[f"layers.{layer_idx}.ffn.gate.weight"] = f"{prefix}.mlp.gate.weight"
            weight_map[f"layers.{layer_idx}.ffn.shared_experts.w1.weight"] = f"{prefix}.mlp.w1.weight"
            weight_map[f"layers.{layer_idx}.ffn.shared_experts.w2.weight"] = f"{prefix}.mlp.w2.weight"
            weight_map[f"layers.{layer_idx}.ffn.shared_experts.w3.weight"] = f"{prefix}.mlp.w3.weight"
            # Norms
            weight_map[f"layers.{layer_idx}.attn_norm.weight"] = f"{prefix}.input_layernorm.weight"
            weight_map[f"layers.{layer_idx}.ffn_norm.weight"] = f"{prefix}.post_attention_layernorm.weight"
        
        # Global weights
        weight_map["embed.weight"] = "model.embed_tokens.weight"
        weight_map["norm.weight"] = "model.norm.weight"
        weight_map["head.weight"] = "lm_head.weight"

        loaded = 0
        shape_bad = 0
        
        def dequantize_fp8_block(w_fp8: torch.Tensor, scale: torch.Tensor, block_size: int = 32) -> torch.Tensor:
            """Dequantize FP8 E4M3 weights with E8M0 block-wise scales."""
            out_dim, in_dim = w_fp8.shape
            w = w_fp8.to(torch.float32)
            s = scale.to(torch.float32)
            
            # Repeat scale blocks
            s = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
            s = s[:out_dim, :in_dim]
            
            return (w * s).to(torch.bfloat16)
        
        for name, loaded_weight in weights:
            # Skip multimodal and metadata
            if any(skip in name for skip in [
                "vision.", "aligner.", "mtp.", "image_", "hc_", 
                ".scale", "experts."  # Phase 1: skip routed experts
            ]):
                continue
            
            if name in weight_map:
                param_name = weight_map[name]
                if param_name in params_dict:
                    param = params_dict[param_name]
                    
                    # FP8 dequantization
                    if loaded_weight.dtype == torch.float8_e4m3fn:
                        if name in scale_cache:
                            loaded_weight = dequantize_fp8_block(loaded_weight, scale_cache[name])
                        else:
                            loaded_weight = loaded_weight.to(torch.bfloat16)
                    
                    # TP sharding for specific weights
                    if "wq_b.weight" in param_name or "wo_a.weight" in param_name:
                        # Shard along output dimension
                        tp_rank = get_tensor_model_parallel_rank()
                        tp_size = get_tensor_model_parallel_world_size()
                        shard_size = loaded_weight.shape[0] // tp_size
                        loaded_weight = loaded_weight[tp_rank * shard_size:(tp_rank + 1) * shard_size]
                    elif "wo_b.weight" in param_name:
                        # Shard along input dimension
                        tp_rank = get_tensor_model_parallel_rank()
                        tp_size = get_tensor_model_parallel_world_size()
                        shard_size = loaded_weight.shape[1] // tp_size
                        loaded_weight = loaded_weight[:, tp_rank * shard_size:(tp_rank + 1) * shard_size]
                    
                    # Handle attn_sink: checkpoint is [64], model is [64/TP]
                    if "attn_sink" in param_name:
                        tp_rank = get_tensor_model_parallel_rank()
                        tp_size = get_tensor_model_parallel_world_size()
                        shard_size = loaded_weight.shape[0] // tp_size
                        loaded_weight = loaded_weight[tp_rank * shard_size:(tp_rank + 1) * shard_size]
                    
                    # lm_head is a ParallelLMHead: use its weight_loader for vocab-dim sharding
                    if param_name == "lm_head.weight":
                        self.lm_head.weight_loader(param, loaded_weight)
                    else:
                        if tuple(param.shape) != tuple(loaded_weight.shape):
                            shape_bad += 1
                            print(f"SHAPE MISMATCH {param_name}: param={tuple(param.shape)} vs ckpt={tuple(loaded_weight.shape)}")
                            continue
                        param.data.copy_(loaded_weight)
                        loaded += 1

        print(f"[LOAD] loaded={loaded} shape_bad={shape_bad}")
