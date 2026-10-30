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
    apply_team()
    patch_parallel_head()
    from .compact_dropout import patch_compact_dropout
    patch_compact_dropout()
