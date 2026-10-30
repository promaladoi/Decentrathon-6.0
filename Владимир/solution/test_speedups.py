"""The fp16 lm_head weight cache (lmhead.py) and the stage timeline wrappers (timeline.py).

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_half_weight_cache_follows_the_parameter():
    from solution.lmhead import _half_weight

    weight = torch.randn(300, 64, device="cuda")
    first = _half_weight(weight)
    assert torch.equal(first, weight.to(torch.float16))
    assert _half_weight(weight) is first  # converted once
    with torch.no_grad():
        weight.add_(1.0)  # an in-place change bumps the version: converted again
    second = _half_weight(weight)
    assert second is not first and torch.equal(second, weight.to(torch.float16))
    other = torch.randn(300, 64, device="cuda")  # another parameter: its own values
    assert torch.equal(_half_weight(other), other.to(torch.float16))


def test_timeline_wrappers_pass_results_and_errors_through(capsys):
    from solution import timeline

    class Stage:
        def run(self, value, *, scale=1):
            if value < 0:
                raise ValueError("negative")
            return value * scale

    timeline._wrap(Stage, "run", "stage")
    timeline._wrap(Stage, "run", "stage")  # idempotent: one wrapper
    assert Stage().run(3, scale=2) == 6
    with pytest.raises(ValueError, match="negative"):
        Stage().run(-1)
    err = capsys.readouterr().err
    assert err.count("stage start") == 2 and err.count("stage end") == 2
    assert timeline._age() > 0
