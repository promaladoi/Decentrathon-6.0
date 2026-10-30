"""Combined team candidate: Vladimir unpadding + Nurzhigit parallel vocabulary reductions.

See TEAM_INTEGRATION.md for provenance and exact component revisions.
"""

_applied = False


def apply():
    global _applied
    if _applied:
        return
    _applied = True
    from .metadata_scan import patch_metadata_scan
    patch_metadata_scan()
    from .tokenizer_fingerprint import patch_tokenizer_fingerprint
    patch_tokenizer_fingerprint()
    from .launcher_warmup import patch_launcher_warmup
    from .lazy_train_cli import patch_lazy_train_cli
    patch_lazy_train_cli()
    from .team_vladimir import apply as apply_team
    from .parallel_head import patch_parallel_head
    from .compact_dropout import patch_compact_dropout
    from .paired_nf4 import patch_paired_nf4
    from .frozen_storage import patch_frozen_storage
    from .frozen_weight_cache import patch_frozen_weight_cache
    from .varlen_attention import patch_varlen_attention
    # Register the active head and compile signatures before the shared
    # apply() starts its model-loading-time compilation thread.
    patch_parallel_head()
    patch_compact_dropout()
    patch_paired_nf4()
    # Balanced variant: retain small attention projections, decode the much
    # larger MLP matrices on demand. Same values, less resident memory.
    patch_frozen_weight_cache(max_elements=4_194_304,max_bytes=400_000_000)
    patch_launcher_warmup()
    apply_team()
    patch_frozen_storage()
    patch_varlen_attention()
