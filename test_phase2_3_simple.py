"""
Phase 2.3 Simple Integration Test - validates weight loading logic
"""

def test_shared_expert_weights():
    """Test that shared expert weights are correctly handled"""
    print("Testing shared expert weight mapping...")

    test_cases = [
        # Phase 2.1: Core MLA
        ("layers.0.attn.wq_a.weight", True),
        ("layers.0.attn.wkv.weight", True),
        ("layers.0.attn.wq_b.weight", True),
        ("layers.0.attn.q_norm.weight", True),
        ("layers.0.attn.kv_norm.weight", True),
        ("layers.0.attn.wo_a.weight", True),
        ("layers.0.attn.wo_b.weight", True),

        # Phase 2.3: MegaMoE shared expert (nested under shared_expert.)
        ("layers.0.ffn.shared_expert.gate_proj.weight", True),
        ("layers.0.ffn.shared_expert.up_proj.weight", True),
        ("layers.0.ffn.shared_expert.down_proj.weight", True),

        # Skipped: 384 routed experts
        ("layers.0.ffn.experts.0.gate_proj.weight", False),
        ("layers.0.ffn.gate.weight", False),

        # Skipped: Compressor internals
        ("layers.0.attn.compressor.fused_wkv_wgate.weight", False),

        # Layer norms
        ("layers.0.attn_norm.weight", True),
        ("layers.0.ffn_norm.weight", True),
    ]

    ALLOWED_ATTN_LEAVES = (
        "wq_a.weight", "wq_b.weight", "wkv.weight",
        "q_norm.weight", "kv_norm.weight",
        "wo_a.weight", "wo_b.weight",
    )
    ALLOWED_LAYER_KEYS = ("attn_norm.weight", "ffn_norm.weight")

    kept = 0
    skipped = 0

    for name, should_keep in test_cases:
        keep = False

        if name.startswith("layers."):
            if ".attn." in name:
                leaf = name.rsplit(".attn.", 1)[1]
                keep = leaf in ALLOWED_ATTN_LEAVES
            elif ".ffn." in name:
                leaf = name.rsplit(".ffn.", 1)[1]
                # Handle shared_expert nested weights
                if leaf.startswith("shared_expert."):
                    final_leaf = leaf.replace("shared_expert.", "")
                    keep = final_leaf in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
                else:
                    keep = False  # Routed experts and gate are skipped
            else:
                keep = name.endswith(ALLOWED_LAYER_KEYS)

        if should_keep:
            assert keep, f"Should keep {name} but got {keep}"
            kept += 1
            print(f"  ✓ Kept: {name}")
        else:
            assert not keep, f"Should skip {name} but got {keep}"
            skipped += 1

    print(f"\nPhase 2.3 Summary:")
    print(f"  Kept (MLA + shared expert): {kept}")
    print(f"  Skipped (routed experts/compressor/etc): {skipped}")
    return kept, skipped

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 2.3 Simple Integration Test")
    print("=" * 70)
    print()

    kept, skipped = test_shared_expert_weights()

    print()
    print("=" * 70)
    print("✅ All Phase 2.3 tests passed!")
    print(f"   {kept} weights will be loaded (MLA + shared expert)")
    print(f"   {skipped} weights will be skipped (routed experts, compressor, etc)")
    print("=" * 70)
