"""Combined team candidate: Vladimir unpadding + Nurzhigit parallel vocabulary reductions.

See TEAM_INTEGRATION.md for provenance and exact component revisions.
"""

_applied = False


def apply():
    global _applied
    if _applied:
        return
    _applied = True
    from .team_vladimir import apply as apply_team
    from .parallel_head import patch_parallel_head
    from .compact_dropout import patch_compact_dropout
    from .paired_nf4 import patch_paired_nf4
    from .frozen_storage import patch_frozen_storage
    # Register the active head and compile signatures before the shared
    # apply() starts its model-loading-time compilation thread.
    patch_parallel_head()
    patch_compact_dropout()
    patch_paired_nf4()
    apply_team()
    patch_frozen_storage()
