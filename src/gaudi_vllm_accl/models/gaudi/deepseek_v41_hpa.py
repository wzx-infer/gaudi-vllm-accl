"""DeepSeek V4.1 Flash - HPA MLA integration."""
import torch
import torch.nn as nn
from typing import Optional

from vllm.config import VllmConfig, CacheConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    ReplicatedLinear, ColumnParallelLinear, RowParallelLinear,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors

import vllm_gaudi.attention.oot_mla  # noqa: F401
from vllm_gaudi.attention.oot_mla import HPUMultiHeadLatentAttentionWrapper

from gaudi_vllm_accl.models.deepseek_v41_flash import DeepSeekV41Config


class DeepseekV41Attention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepSeekV41Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim ** -0.5

        if q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                hidden_size, q_lora_rank, bias=False,
                quant_config=quant_config, prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank, num_heads * self.qk_head_dim, bias=False,
                quant_config=quant_config, prefix=f"{prefix}.q_b_proj",
            )
            self.q_proj = None
        else:
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
            self.q_proj = ColumnParallelLinear(
                hidden_size, num_heads * self.qk_head_dim, bias=False,
                quant_config=quant_config, prefix=f"{prefix}.q_proj",
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False,
            quant_config=quant_config, prefix=f"{prefix}.kv_b_proj",
        )

        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj",
        )

        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        if not isinstance(rope_params, dict):
            rope_params = {}
        rope_params.setdefault("rope_type", "default")
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_params,
            is_neox_style=False,
        )

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
            indexer_rotary_emb=None,
        )

        self.mla_attn = HPUMultiHeadLatentAttentionWrapper(
            hidden_size=hidden_size,
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            skip_topk=True,
        )

    def forward(self, positions, hidden_states):
        return self.mla_attn(positions, hidden_states)


class DeepseekV41MoE(nn.Module):
    def __init__(self, config: DeepSeekV41Config, layer_idx: int):
        super().__init__()
        from vllm.model_executor.layers.activation import SiluAndMul
        inter = getattr(config, "intermediate_size", 12288)
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size, 2 * inter, bias=False,
            prefix=f"model.layers.{layer_idx}.mlp.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            inter, config.hidden_size, bias=False,
            prefix=f"model.layers.{layer_idx}.mlp.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekV41Config, layer_idx: int, vllm_config=None,
                 cache_config=None, quant_config=None, prefix=""):
        super().__init__()
        self.self_attn = DeepseekV41Attention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=getattr(config, "num_attention_heads", 64),
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 128),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(config, "v_head_dim", 128),
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=getattr(config, "kv_lora_rank", 512),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = DeepseekV41MoE(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual=None):
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
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.layers = nn.ModuleList([
            DeepseekV41DecoderLayer(
                config, i, vllm_config=vllm_config,
                cache_config=cache_config, quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV41ForCausalLM(nn.Module, SupportsPP):
    is_text_generation_model = True
    is_pooling_model = False
    supports_multimodal = False
    supports_multimodal_raw_input_only = False
    supports_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: DeepSeekV41Config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.model = DeepseekV41Model(vllm_config=vllm_config, prefix=f"{prefix}.model")
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False,
            prefix=f"{prefix}.lm_head",
        )

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        logits, _ = self.lm_head(hidden_states)
        return logits

    def compute_logits(self, hidden_states):
        logits, _ = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights):
        SKIP = (".indexer.", ".indexers_proj.", ".compressor.", "vision.", "aligner.")
        filtered = ((n, w) for n, w in weights if not any(p in n for p in SKIP))
        params = dict(self.named_parameters())
        for name, w in filtered:
            if name in params:
                params[name].data.copy_(w)
