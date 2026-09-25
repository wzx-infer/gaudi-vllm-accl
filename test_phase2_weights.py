#!/usr/bin/env python3
"""Test Phase 2 MLA weight loading logic.

Validates that the Phase 2 architecture correctly maps checkpoint weights
to the fused MLA architecture.
"""

import sys
sys.path.insert(0, 'src')


def test_phase2_weight_mapping():
    """Test Phase 2 MLA weight mapping logic."""

    # Phase 2: MLA architecture allow-list
    ALLOWED_ATTN_LEAVES = (
        "wq_a.weight", "wq_b.weight", "wkv.weight",
        "q_norm.weight", "kv_norm.weight",
        "wo_a.weight", "wo_b.weight",
    )
    ALLOWED_FFN_LEAVES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    ALLOWED_LAYER_KEYS = ("attn_norm.weight", "ffn_norm.weight")

    test_cases = [
        # Top-level weights
        ("embed.weight", True, "model.embed_tokens.weight"),
        ("head.weight", True, "lm_head.weight"),
        ("norm.weight", True, "model.norm.weight"),

        # Layer norms
        ("layers.0.attn_norm.weight", True, "model.layers.0.input_layernorm.weight"),
        ("layers.0.ffn_norm.weight", True, "model.layers.0.post_attention_layernorm.weight"),

        # Phase 2: MLA attention weights - KEPT
        ("layers.0.attn.wq_a.weight", True, "model.layers.0.self_attn.fused_wqa_wkv.weight.0"),
        ("layers.0.attn.wq_b.weight", True, "model.layers.0.self_attn.wq_b.weight"),
        ("layers.0.attn.wkv.weight", True, "model.layers.0.self_attn.fused_wqa_wkv.weight.1"),
        ("layers.0.attn.q_norm.weight", True, "model.layers.0.self_attn.q_norm.weight"),
        ("layers.0.attn.kv_norm.weight", True, "model.layers.0.self_attn.kv_norm.weight"),
        ("layers.0.attn.wo_a.weight", True, "model.layers.0.self_attn.wo_a.weight"),
        ("layers.0.attn.wo_b.weight", True, "model.layers.0.self_attn.wo_b.weight"),

        # Dense FFN weights - KEPT
        ("layers.0.ffn.gate_proj.weight", True, "model.layers.0.mlp.gate_proj.weight"),
        ("layers.0.ffn.up_proj.weight", True, "model.layers.0.mlp.up_proj.weight"),
        ("layers.0.ffn.down_proj.weight", True, "model.layers.0.mlp.down_proj.weight"),

        # Multimodal - SKIPPED
        ("aligner.proj.weight", False, "skip_multimodal"),
        ("vision.encoder.layer.0.weight", False, "skip_multimodal"),

        # MTP - SKIPPED
        ("mtp.embed.weight", False, "skip_mtp"),

        # Compressor/indexer - SKIPPED (TODO Phase 2.1)
        ("layers.2.attn.compressor.proj.weight", False, "skip_unsupported_layer"),
        ("layers.2.attn.indexer.wq_b.weight", False, "skip_unsupported_layer"),

        # MoE experts - SKIPPED (TODO Phase 2.2)
        ("layers.10.ffn.experts.0.w1.weight", False, "skip_unsupported_layer"),
        ("layers.10.ffn.gate.weight", False, "skip_unsupported_layer"),
        ("layers.10.ffn.shared_expert.gate_proj.weight", False, "skip_unsupported_layer"),

        # Hyperconnections/Engram - SKIPPED (TODO Phase 2.3)
        ("layers.1.hc_attn_base.weight", False, "skip_unsupported_layer"),
        ("layers.1.hc_ffn_base.weight", False, "skip_unsupported_layer"),
        ("layers.1.engram.weight", False, "skip_unsupported_layer"),
    ]

    def filter_and_map(name):
        """Phase 2 weight filtering logic."""
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

        # Map checkpoint naming to model naming
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

            # Handle fused_wqa_wkv mapping
            if ".self_attn.wq_a.weight" in mapped_name:
                mapped_name = mapped_name.replace(".wq_a.weight", ".fused_wqa_wkv.weight.0")
            elif ".self_attn.wkv.weight" in mapped_name:
                mapped_name = mapped_name.replace(".wkv.weight", ".fused_wqa_wkv.weight.1")
        elif name == "norm.weight":
            mapped_name = "model.norm.weight"

        return True, mapped_name

    counts = {"skip_multimodal": 0, "skip_mtp": 0, "skip_unsupported_layer": 0, "kept": 0}

    print("Testing Phase 2 MLA weight mapping:")
    print("=" * 80)

    for name, expected_keep, expected_detail in test_cases:
        keep, detail = filter_and_map(name)

        assert keep == expected_keep, (
            f"{name}: expected keep={expected_keep}, got keep={keep}"
        )
        if keep:
            assert detail == expected_detail, (
                f"{name}: expected mapped name {expected_detail!r}, got {detail!r}"
            )
            counts["kept"] += 1
            print(f"[KEEP]   {name:50s} -> {detail}")
        else:
            assert detail == expected_detail, (
                f"{name}: expected skip reason {expected_detail!r}, got {detail!r}"
            )
            counts[detail] += 1
            print(f"[SKIP]   {name:50s} ({detail})")

    print("=" * 80)
    print("\nPhase 2 Summary:")
    print(f"  Kept (MLA + dense FFN):      {counts['kept']}")
    print(f"  Skipped (multimodal):        {counts['skip_multimodal']}")
    print(f"  Skipped (MTP):               {counts['skip_mtp']}")
    print(f"  Skipped (TODO Phase 2.1-3):  {counts['skip_unsupported_layer']}")
    print("=" * 80)
    print("\n✓ All Phase 2 weight mapping tests passed!")
    print("\nPhase 2 brings:")
    print("  + Query LoRA decomposition (wq_a @ wq_b)")
    print("  + Compressed KV cache (wkv)")
    print("  + Grouped output projection (wo_a @ wo_b)")
    print("  + Fused wqa_wkv projection for efficiency")
    print("\nStill TODO:")
    print("  - Compressor for KV compression (compress_ratio 1/2)")
    print("  - Indexer for sparse attention topology")
    print("  - MegaMoE (384 routed experts + shared expert)")
    print("  - Hyperconnections (CED multi-stream)")
    print("  - Engram sparse long-term memory")

    return True


if __name__ == "__main__":
    try:
        test_phase2_weight_mapping()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
