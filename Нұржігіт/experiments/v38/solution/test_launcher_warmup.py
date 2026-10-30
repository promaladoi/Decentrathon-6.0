from types import SimpleNamespace

import pytest

from .launcher_warmup import prepare


class FakeJIT:
    def __init__(self,entries): self.device_caches=entries


def test_prepare_each_compiled_kernel_once_and_do_not_launch():
    calls=[]
    a=SimpleNamespace(_init_handles=lambda: calls.append('a'))
    b=SimpleNamespace(_init_handles=lambda: calls.append('b'))
    fn=FakeJIT({0:({'a':a,'alias':a},None,None),1:{'b':b}})
    modules=[SimpleNamespace(a=fn,alias=fn,other=object()),SimpleNamespace(same=fn)]
    assert prepare(modules,FakeJIT)==2
    assert calls==['a','b']


def test_empty_missing_and_unknown_caches():
    mod=SimpleNamespace(empty=FakeJIT({}),unknown=FakeJIT({0:None}),
                        old=FakeJIT({0:{'no_init':object()}}))
    assert prepare([mod],FakeJIT)==0


def test_failed_preparation_is_reported_to_existing_warmup_fallback():
    def fail(): raise RuntimeError('normal launch will initialize')
    mod=SimpleNamespace(fn=FakeJIT({0:{'a':SimpleNamespace(_init_handles=fail)}}))
    with pytest.raises(RuntimeError,match='normal launch'):
        prepare([mod],FakeJIT)
