import sys
try:
    from gaudi_vllm_accl.register import register
    register()
    print("[autoreg] gaudi-vllm-accl registered OK", file=sys.stderr, flush=True)
except Exception as e:
    import traceback
    print(f"[autoreg] registration FAILED: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
