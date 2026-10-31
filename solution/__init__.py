"""v44: corrected team training path plus Nurzhigit's lossless frozen storage.

Runtime modules/tests derive from Merey stage2, which credits Vladimir.
See TEAM_INTEGRATION.md. No full-task v44 result is claimed before measurement.
"""
_applied = False


def apply():
    global _applied
    if _applied:
        return
    from .lazy_train_cli import patch_lazy_train_cli
    patch_lazy_train_cli()
    from . import warmup
    from .layer import patch_qwen2_decoder_layer
    from .mlp import patch_qwen2_mlp_4bit
    from .nf4 import patch_bnb_matmul_4bit, enable_resident_weights
    from .rmsnorm import patch_qwen2_rmsnorm
    from .rope import patch_qwen2_rope
    from .swiglu import patch_qwen2_mlp
    from .unpad import patch_qwen2_causal_lm
    from .frozen_storage import patch_frozen_storage
    _applied = True
    enable_resident_weights()
    warmup.start(unpad=True)
    patch_qwen2_mlp()
    patch_qwen2_mlp_4bit()
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    patch_qwen2_rope()
    patch_bnb_matmul_4bit()
    patch_qwen2_causal_lm()
    from .private_scope import install as install_private_scope
    install_private_scope()
    from .tokenization_map import install as install_tokenization_map
    install_tokenization_map()
    from .package_metadata_snapshot import install as install_metadata
    install_metadata()
    # wraps() retains the private-forward marker. Unsupported weights retain
    # the ordinary path; exact values and FP32 embedding outputs are preserved.
    patch_frozen_storage()
