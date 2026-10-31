"""Keep native training dashboard updates in the main training thread.

The native Rich refresh worker was the observed reason for rejecting direct
dropout's single CUDA-RNG-thread contract. Rendering synchronously removes
that worker without weakening the RNG guard or changing recorded metrics.
"""
import functools
import hashlib
import inspect

_START_SHA='be125aa570eef728d3242ec82b68618b1f0ca20e91cdcdcce2d2975af48dbaf4'
_UPDATE_SHA='081addf174993845caba466962fe467a7ecd5d36af9a40ff215942246948b4c8'


def install():
    from rich.live import Live
    from soup_cli.monitoring import display
    cls=display.TrainingDisplay
    old_start,old_update=cls.start,cls.update
    if getattr(old_start,'_soup_sync_display',False):
        return False
    try:
        if (display.Live is not Live
                or hashlib.sha256(inspect.getsource(old_start).encode()).hexdigest()!=_START_SHA
                or hashlib.sha256(inspect.getsource(old_update).encode()).hexdigest()!=_UPDATE_SHA):
            return False
    except (OSError,TypeError):
        return False

    @functools.wraps(old_start)
    def start(self,total_steps):
        if type(self) is not cls:
            return old_start(self,total_steps)
        self.total_steps=total_steps
        self._live=Live(self._render(),console=display.console,
                        refresh_per_second=2,auto_refresh=False)
        self._live.start()

    @functools.wraps(old_update)
    def update(self,*args,**kwargs):
        result=old_update(self,*args,**kwargs)
        if type(self) is cls and self._live is not None and self._live.auto_refresh is False:
            self._live.refresh()
        return result

    start._soup_sync_display=True
    cls.start,cls.update=start,update
    return True
