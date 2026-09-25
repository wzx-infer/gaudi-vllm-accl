#!/usr/bin/env python3
"""Test weight filtering logic for DeepSeek V4.1 Flash model.

Mirrors the allow-list filtering in
src/gaudi_vllm_accl/models/gaudi/model.py::DeepseekV41ForCausalLM.load_weights.
"""

import sys
sys.path.insert(0, 'src')

ALLOWED_ATTN_LEAVES = ("qkv_proj.weight", "o_proj.weight")
ALLOWED_FFN_LEAVES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
ALLOWED_LAYER_KEYS = ("attn_norm.weight", "ffn_norm.weight")


def filter_and_map(name):
    """Re-implements the filtering/mapping logic from load_weights().

    Returns (keep: bool, mapped_name_or_skip_reason: str).
    """
    if name.startswith(("aligner.", "vision.", "image_start", "image_end", "image_newline")):
        return False, "skip_multimodal"

    if name.startswith("mtp."):
        return False, "skip_mtp"

    if name.startswith("layers."):
        if ".attn." in name:
            leaf = name.rsplit(".attn.", 1)[1]
            keep = leaf in ALLOWED_ATTN_LEAVES
        elif ".ffn." in name:
            leaf = name.rsplit(".ffn.", 1)[1]
            keep = leaf in ALLOWED_FFN_LEAVES
        else:
            keep = name.endswith(ALLOWED_LAYER_KEYS)

        if not keep:
            return False, "skip_unsupported_layer"

    mapped_name = name
    if name == "embed.weight":
        mapped_name = "model.embed_tokens.weight"
    elif name == "head.weight":
        mapped_name = "lm_head.weight"
    elif name.startswith("layers."):
        mapped_name = "model." + name
        mapped_name = mapped_name.replace(".attn.", ".self_attn.")
        mapped_name = mapped_name.replace(".ffn.", ".mlp.")
        mapped_name = mapped_name.replace(".attn_norm.", ".input_layernorm.")
        mapped_name = mapped_name.replace(".ffn_norm.", ".post_attention_layernorm.")
    elif name == "norm.weight":
        mapped_name = "model.norm.weight"

    return True, mapped_name


def test_weight_filtering():
    """Test the allow-list weight filtering and mapping logic."""

    test_cases = [
        # Top-level weights that should be kept
        ("embed.weight", True, "model.embed_tokens.weight"),
        ("head.weight", True, "lm_head.weight"),
        ("norm.weight", True, "model.norm.weight"),

        # Per-layer norms - kept
        ("layers.0.attn_norm.weight", True, "model.layers.0.input_layernorm.weight"),
        ("layers.0.ffn_norm.weight", True, "model.layers.0.post_attention_layernorm.weight"),

        # Simplified attention/FFN weights - kept
        ("layers.0.attn.qkv_proj.weight", True, "model.layers.0.self_attn.qkv_proj.weight"),
        ("layers.0.attn.o_proj.weight", True, "model.layers.0.self_attn.o_proj.weight"),
        ("layers.3.ffn.gate_proj.weight", True, "model.layers.3.mlp.gate_proj.weight"),
        ("layers.3.ffn.up_proj.weight", True, "model.layers.3.mlp.up_proj.weight"),
        ("layers.3.ffn.down_proj.weight", True, "model.layers.3.mlp.down_proj.weight"),

        # Multimodal weights - skipped
        ("aligner.proj.weight", False, "skip_multimodal"),
        ("vision.encoder.layer.0.weight", False, "skip_multimodal"),
        ("image_start", False, "skip_multimodal"),
        ("image_end", False, "skip_multimodal"),
        ("image_newline", False, "skip_multimodal"),

        # MTP - skipped
        ("mtp.embed.weight", False, "skip_mtp"),
        ("mtp.layers.0.attn.qkv_proj.weight", False, "skip_mtp"),

        # Full MLA attention internals - skipped (not in the allow-list)
        ("layers.0.attn.wq_a.weight", False, "skip_unsupported_layer"),
        ("layers.0.attn.wq_b.weight", False, "skip_unsupported_layer"),
        ("layers.0.attn.wkv.weight", False, "skip_unsupported_layer"),
        ("layers.0.attn.wo_a.weight", False, "skip_unsupported_layer"),
        ("layers.0.attn.wo_b.weight", False, "skip_unsupported_layer"),
        ("layers.0.attn.attn_sink", False, "skip_unsupported_layer"),
        ("layers.0.attn.q_norm.weight", False, "skip_unsupported_layer"),
        ("layers.0.attn.kv_norm.weight", False, "skip_unsupported_layer"),
        ("layers.0.attn.wq_a.scale", False, "skip_unsupported_layer"),
        ("layers.2.attn.compressor.proj.weight", False, "skip_unsupported_layer"),
        ("layers.2.attn.indexer.wq_b.weight", False, "skip_unsupported_layer"),
        ("layers.2.attn.indexer.wk.weight", False, "skip_unsupported_layer"),
        ("layers.2.attn.swa_cache_layer.weight", False, "skip_unsupported_layer"),

        # MegaMoE expert weights - skipped (not in the allow-list)
        ("layers.0.ffn.experts.0.w1.weight", False, "skip_unsupported_layer"),
        ("layers.0.ffn.experts.127.w2.weight", False, "skip_unsupported_layer"),
        ("layers.0.ffn.experts.383.w3.weight", False, "skip_unsupported_layer"),
        ("layers.0.ffn.gate.weight", False, "skip_unsupported_layer"),
        ("layers.10.ffn.shared_expert.gate_proj.weight", False, "skip_unsupported_layer"),
        ("layers.10.ffn.shared_experts.gate_proj.weight", False, "skip_unsupported_layer"),

        # Hyperconnections and Engram - per-layer siblings of attn/ffn, skipped
        ("layers.1.hc_attn_base.weight", False, "skip_unsupported_layer"),
        ("layers.1.hc_ffn_base.weight", False, "skip_unsupported_layer"),
        ("layers.1.engram.weight", False, "skip_unsupported_layer"),
    ]

    counts = {"skip_multimodal": 0, "skip_mtp": 0, "skip_unsupported_layer": 0, "kept": 0}

    for name, expected_keep, expected_detail in test_cases:
        keep, detail = filter_and_map(name)

        assert keep == expected_keep, (
            f"{name}: expected keep={expected_keep}, got keep={keep} (detail={detail})"
        )
        if keep:
            assert detail == expected_detail, (
                f"{name}: expected mapped name {expected_detail!r}, got {detail!r}"
            )
            counts["kept"] += 1
            print(f"[keep]   {name:45s} -> {detail}")
        else:
            assert detail == expected_detail, (
                f"{name}: expected skip reason {expected_detail!r}, got {detail!r}"
            )
            counts[detail] += 1
            print(f"[skip]   {name:45s} ({detail})")

    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"  Kept:                        {counts['kept']}")
    print(f"  Skipped (multimodal):        {counts['skip_multimodal']}")
    print(f"  Skipped (mtp):               {counts['skip_mtp']}")
    print(f"  Skipped (unsupported layer): {counts['skip_unsupported_layer']}")
    print(f"{'=' * 80}")
    print("\nAll weight filtering tests passed!")

    return True


if __name__ == "__main__":
    try:
        test_weight_filtering()
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
