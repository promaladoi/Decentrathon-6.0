"""Preserve native dashboard state/rendering and unknown implementation fallback."""
from io import StringIO
from types import SimpleNamespace
import pytest


def _native(monkeypatch):
    from rich.console import Console
    from soup_cli.monitoring import display
    cls=display.TrainingDisplay
    monkeypatch.setattr(cls,'start',cls.start)
    monkeypatch.setattr(cls,'update',cls.update)
    monkeypatch.setattr(display,'console',Console(file=StringIO(),force_terminal=True,width=90))
    return display,cls


def test_all_dashboard_fields_sticky_eval_render_and_lifecycle_match_native(monkeypatch):
    from solution.synchronous_display import install
    display,cls=_native(monkeypatch)
    config=SimpleNamespace(training=SimpleNamespace(epochs=1),experiment_name='exact dashboard',base='model')
    reference=cls(config)
    native_update=cls.update
    assert install() is True
    assert install() is False
    observed=cls(config)
    reference.total_steps=64
    observed.start(64)
    try:
        assert observed._live.auto_refresh is False
        assert observed._live._refresh_thread is None
        for step,loss in ((0,None),(1,2.1),(2,2.0)):
            kwargs=dict(grad_norm=.2,speed=3.,gpu_mem='1200 MiB')
            if step==1:kwargs.update(val_loss=2.2,val_reward=.7)
            native_update(reference,step,step/64,loss,2e-4,**kwargs)
            observed.update(step,step/64,loss,2e-4,**kwargs)
            for name in ('current_step','total_steps','current_epoch','loss','lr','grad_norm',
                         'speed','gpu_mem','val_loss','val_reward'):
                assert getattr(observed,name)==getattr(reference,name)
            assert observed._render().renderable==reference._render().renderable
    finally:
        observed.stop()
        observed.stop()
    assert observed._live is None


@pytest.mark.parametrize('method',['start','update'])
def test_unknown_source_keeps_native_or_observer_implementation(monkeypatch,method):
    from solution.synchronous_display import install
    display,cls=_native(monkeypatch)
    def observer(*args,**kwargs):return 'observer'
    monkeypatch.setattr(cls,method,observer)
    assert install() is False
    assert getattr(cls,method) is observer
