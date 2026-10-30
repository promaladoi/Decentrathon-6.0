"""Defer TRL 0.29.1's native package/distribution scan until its first use.

Only the original import-time call in the installed trl.import_utils module
receives a lazy mapping. Every availability helper remains native, and the
first mapping operation runs the untouched full stdlib scan. No distribution
mapping is fabricated, persisted, or borrowed from another caller/process.
"""

from collections.abc import MutableMapping
from contextlib import contextmanager
import copy
import functools
import importlib
import importlib.machinery
import importlib.metadata as metadata
import importlib.util
import os
import sys
import threading
import types


_TARGET = "trl.import_utils"
_NATIVE_FUNCTION = metadata.packages_distributions
_PRELOAD_LOCK = threading.RLock()


class _LazyPackageMapping(MutableMapping):
    """A fresh native dictionary, materialized once at the first operation."""

    def __init__(self, factory):
        self._factory = factory
        self._value = None
        self._lock = threading.RLock()

    def _materialize(self):
        if self._value is None:
            with self._lock:
                if self._value is None:
                    # Do not cache a failure: native lookup errors propagate,
                    # and a later operation can retry the native factory.
                    self._value = self._factory()
        return self._value

    def __getitem__(self, key):
        return self._materialize()[key]

    def __setitem__(self, key, value):
        self._materialize()[key] = value

    def __delitem__(self, key):
        del self._materialize()[key]

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self):
        return len(self._materialize())

    def __contains__(self, key):
        return key in self._materialize()

    def __reversed__(self):
        return reversed(self._materialize())

    def __repr__(self):
        return repr(self._materialize())

    def __str__(self):
        return str(self._materialize())

    def __eq__(self, other):
        if isinstance(other, _LazyPackageMapping):
            other = other._materialize()
        return self._materialize() == other

    def __or__(self, other):
        if isinstance(other, _LazyPackageMapping):
            other = other._materialize()
        return self._materialize().__or__(other)

    def __ror__(self, other):
        return self._materialize().__ror__(other)

    def __ior__(self, other):
        self._materialize().__ior__(other)
        return self

    def get(self, *args, **kwargs):
        return self._materialize().get(*args, **kwargs)

    def keys(self):
        return self._materialize().keys()

    def values(self):
        return self._materialize().values()

    def items(self):
        return self._materialize().items()

    def copy(self):
        return self._materialize().copy()

    def clear(self):
        return self._materialize().clear()

    def pop(self, *args, **kwargs):
        return self._materialize().pop(*args, **kwargs)

    def popitem(self):
        return self._materialize().popitem()

    def setdefault(self, *args, **kwargs):
        return self._materialize().setdefault(*args, **kwargs)

    def update(self, *args, **kwargs):
        return self._materialize().update(*args, **kwargs)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return copy.deepcopy(self._materialize(), memo)

    def __reduce_ex__(self, protocol):
        # Serializing this mapping is an actual access. Serialize the real
        # native dictionary, rather than a lock or a deferred scan callback.
        return dict, (self._materialize(),)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _native_function_unchanged():
    function = metadata.packages_distributions
    try:
        return (function is _NATIVE_FUNCTION and isinstance(function, types.FunctionType)
                and function.__module__ == "importlib.metadata"
                and function.__qualname__ == "packages_distributions"
                and function.__globals__ is vars(metadata)
                and metadata.__spec__ is not None and metadata.__spec__.origin is not None
                and _canonical(function.__code__.co_filename) == _canonical(metadata.__spec__.origin))
    except (AttributeError, TypeError, ValueError, OSError):
        return False


def _guarded_origin():
    """Recognize the pinned installed source without first importing TRL."""
    try:
        distribution = metadata.distribution("trl")
        if distribution.version != "0.29.1":
            return None
        parent = importlib.util.find_spec("trl")
        if (parent is None or type(parent.loader) is not importlib.machinery.SourceFileLoader
                or not parent.submodule_search_locations or not parent.origin):
            return None
        if _canonical(parent.origin) != _canonical(distribution.locate_file("trl/__init__.py")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(_TARGET, parent.submodule_search_locations)
        if (spec is None or type(spec.loader) is not importlib.machinery.SourceFileLoader
                or not spec.origin or spec.name != _TARGET):
            return None
        expected = _canonical(distribution.locate_file("trl/import_utils.py"))
        return expected if _canonical(spec.origin) == expected else None
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return None


def _exact_import_caller(frame, origin):
    namespace = frame.f_globals
    spec = namespace.get("__spec__")
    return (frame.f_code.co_name == "<module>" and namespace.get("__name__") == _TARGET
            and spec is not None and getattr(spec, "name", None) == _TARGET
            and getattr(spec, "origin", None) is not None
            and namespace.get("__file__") is not None
            and _canonical(frame.f_code.co_filename) == origin
            and _canonical(namespace["__file__"]) == origin
            and _canonical(spec.origin) == origin)


def _make_wrapper(original, origin, owner_thread):
    @functools.wraps(original)
    def packages_distributions(*args, **kwargs):
        # Even an identical caller in a different thread receives the eager
        # original. The temporary patch has no routing effect outside this
        # import's exact module frame and its zero-argument native call.
        qualified = False
        if not args and not kwargs and threading.get_ident() == owner_thread:
            try:
                qualified = _exact_import_caller(sys._getframe(1), origin)
            except (AttributeError, TypeError, ValueError, OSError):
                pass
        if qualified:
            return _LazyPackageMapping(original)
        return original(*args, **kwargs)

    return packages_distributions


@contextmanager
def defer_trl_package_scan():
    """Keep the existing import order, narrowly deferring the native TRL scan.

    This context imports no TRL module. Unsupported/already-loaded setups are
    untouched. Original caller exceptions propagate after function restoration.
    """
    with _PRELOAD_LOCK:
        if _TARGET in sys.modules or not _native_function_unchanged():
            yield False
            return
        origin = _guarded_origin()
        if origin is None:
            yield False
            return
        original = metadata.packages_distributions
        wrapper = _make_wrapper(original, origin, threading.get_ident())
        try:
            metadata.packages_distributions = wrapper
            yield True
        finally:
            metadata.packages_distributions = original


def preload_trl_import_utils():
    """Optional standalone diagnostic API; production uses the context above.

    If this explicit preload fails, restore first and retry the ordinary native
    import. Native failure then propagates normally; unsupported setups no-op.
    """
    failed = False
    with defer_trl_package_scan() as active:
        if not active:
            return False
        try:
            module = importlib.import_module(_TARGET)
        except Exception:
            failed = True
    if failed:
        importlib.import_module(_TARGET)
        return False
    return isinstance(getattr(module, "PACKAGE_DISTRIBUTION_MAPPING", None), _LazyPackageMapping)
