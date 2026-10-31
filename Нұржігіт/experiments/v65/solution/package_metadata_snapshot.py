"""Reuse one native package-metadata snapshot in the fixed installed task.

The harness imports Transformers 5.16.1 before solution.apply(). Its native
import utility already owns the result of packages_distributions(). TRL
0.29.1 immediately repeats that scan. For this exact installed environment,
with no package installation/removal or metadata mutation between imports,
give only TRL's native import-time call an independent deep copy of the
existing native dictionary. Native availability/version helpers are unchanged.

This is not equivalence for dynamic installation or arbitrary mutable package
metadata. No external/process cache, fabricated entry, availability override,
or GPU state is involved. Unrecognized callers/environments use the original
stdlib function, restored when the context exits even after an exception.
"""

from contextlib import contextmanager
import copy
import functools
import importlib.machinery
import importlib.metadata as metadata
import importlib.util
import os
import sys
import threading
import types


_TRL_MODULE = "trl.import_utils"
_TF_MODULE = "transformers.utils.import_utils"
_SOUP_MODULE = "soup_cli.trainer.sft"
_SETUP_MARKER = "_merey_package_metadata_snapshot_setup"
_NATIVE_FUNCTION = metadata.packages_distributions
_CONTEXT_LOCK = threading.RLock()


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


def _plain_mapping(value):
    return (type(value) is dict
            and all(type(key) is str and type(names) is list
                    and all(type(name) is str for name in names)
                    for key, names in value.items()))


def _transformers_source():
    """Return the already-loaded native owner/map/helper; never import it."""
    module = sys.modules.get(_TF_MODULE)
    if type(module) is not types.ModuleType:
        return None
    try:
        distribution = metadata.distribution("transformers")
        if distribution.version != "5.16.1":
            return None
        origin = _canonical(distribution.locate_file("transformers/utils/import_utils.py"))
        namespace = vars(module)
        spec = namespace.get("__spec__")
        mapping = namespace.get("PACKAGE_DISTRIBUTION_MAPPING")
        function = namespace.get("_is_package_available")
        if (namespace.get("__name__") != _TF_MODULE or spec is None
                or spec.name != _TF_MODULE or type(spec.loader) is not importlib.machinery.SourceFileLoader
                or _canonical(spec.origin) != origin or _canonical(namespace.get("__file__")) != origin
                or not isinstance(function, types.FunctionType)
                or function.__module__ != _TF_MODULE or function.__qualname__ != "_is_package_available"
                or not isinstance(function.__code__, types.CodeType)
                or function.__code__.co_name != "_is_package_available"
                or _canonical(function.__code__.co_filename) != origin
                or function.__globals__ is not namespace
                or function.__globals__.get("PACKAGE_DISTRIBUTION_MAPPING") is not mapping
                or "PACKAGE_DISTRIBUTION_MAPPING" not in function.__code__.co_names
                or not _plain_mapping(mapping)):
            return None
        return module, mapping, function
    except (ImportError, AttributeError, TypeError, ValueError, OSError, RuntimeError):
        return None


def _trl_origin():
    """Find the pinned installed source without importing any TRL module."""
    try:
        distribution = metadata.distribution("trl")
        if distribution.version != "0.29.1":
            return None
        parent = importlib.util.find_spec("trl")
        if (parent is None or type(parent.loader) is not importlib.machinery.SourceFileLoader
                or not parent.submodule_search_locations or not parent.origin
                or _canonical(parent.origin) != _canonical(distribution.locate_file("trl/__init__.py"))):
            return None
        spec = importlib.machinery.PathFinder.find_spec(_TRL_MODULE, parent.submodule_search_locations)
        if (spec is None or type(spec.loader) is not importlib.machinery.SourceFileLoader
                or not spec.origin or spec.name != _TRL_MODULE):
            return None
        expected = _canonical(distribution.locate_file("trl/import_utils.py"))
        return expected if _canonical(spec.origin) == expected else None
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return None


def _exact_trl_caller(frame, origin):
    namespace = frame.f_globals
    spec = namespace.get("__spec__")
    module = sys.modules.get(_TRL_MODULE)
    return (type(module) is types.ModuleType and namespace is vars(module)
            and frame.f_code.co_name == "<module>" and namespace.get("__name__") == _TRL_MODULE
            and spec is not None and getattr(spec, "name", None) == _TRL_MODULE
            and type(getattr(spec, "loader", None)) is importlib.machinery.SourceFileLoader
            and _canonical(getattr(spec, "origin", None)) == origin
            and _canonical(namespace.get("__file__")) == origin
            and _canonical(frame.f_code.co_filename) == origin)


def _source_is_current(source):
    module, mapping, function = source
    return (sys.modules.get(_TF_MODULE) is module
            and vars(module).get("PACKAGE_DISTRIBUTION_MAPPING") is mapping
            and vars(module).get("_is_package_available") is function
            and function.__globals__ is vars(module)
            and function.__globals__.get("PACKAGE_DISTRIBUTION_MAPPING") is mapping
            and _plain_mapping(mapping))


def _make_wrapper(original, origin, source, owner_thread):
    @functools.wraps(original)
    def packages_distributions(*args, **kwargs):
        if not args and not kwargs and threading.get_ident() == owner_thread:
            try:
                if (_exact_trl_caller(sys._getframe(1), origin) and _source_is_current(source)):
                    snapshot = copy.deepcopy(source[1])
                    if _plain_mapping(snapshot):
                        return snapshot
            except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
                # Unsupported source mutation/routing falls back to the real
                # scan rather than returning incomplete or invented metadata.
                pass
        return original(*args, **kwargs)

    return packages_distributions


@contextmanager
def reuse_transformers_package_metadata():
    """Preserve import order; reuse only the exact native TRL import-time scan."""
    with _CONTEXT_LOCK:
        if _TRL_MODULE in sys.modules or not _native_function_unchanged():
            yield False
            return
        source = _transformers_source()
        origin = _trl_origin() if source is not None else None
        if origin is None:
            yield False
            return
        original = metadata.packages_distributions
        try:
            metadata.packages_distributions = _make_wrapper(
                original, origin, source, threading.get_ident())
            yield True
        finally:
            metadata.packages_distributions = original


def _native_soup_setup():
    """Recognize the already-loaded, unwrapped Soup 0.77 native setup method."""
    module = sys.modules.get(_SOUP_MODULE)
    if type(module) is not types.ModuleType:
        return None
    try:
        distribution = metadata.distribution("soup-cli")
        if distribution.version != "0.77.0":
            return None
        namespace = vars(module)
        spec = namespace.get("__spec__")
        origin = _canonical(distribution.locate_file("soup_cli/trainer/sft.py"))
        wrapper_type = namespace.get("SFTTrainerWrapper")
        if (namespace.get("__name__") != _SOUP_MODULE or spec is None
                or spec.name != _SOUP_MODULE or type(spec.loader) is not importlib.machinery.SourceFileLoader
                or _canonical(spec.origin) != origin or _canonical(namespace.get("__file__")) != origin
                or type(wrapper_type) is not type or wrapper_type.__module__ != _SOUP_MODULE
                or wrapper_type.__qualname__ != "SFTTrainerWrapper"):
            return None
        setup = vars(wrapper_type).get("setup")
        if (not isinstance(setup, types.FunctionType) or hasattr(setup, "__wrapped__")
                or getattr(setup, _SETUP_MARKER, False)
                or setup.__module__ != _SOUP_MODULE or setup.__qualname__ != "SFTTrainerWrapper.setup"
                or setup.__code__.co_name != "setup" or setup.__code__.co_qualname != "SFTTrainerWrapper.setup"
                or _canonical(setup.__code__.co_filename) != origin
                or setup.__globals__ is not namespace
                or setup.__globals__.get("SFTTrainerWrapper") is not wrapper_type):
            return None
        return wrapper_type, setup
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return None


def _make_setup_wrapper(original):
    @functools.wraps(original)
    def setup(*args, **kwargs):
        # Soup imports TRL inside setup(), after solution.apply() has returned.
        # Keep the original import order, all arguments and the entire native
        # body; the narrow metadata routing exists only during this invocation.
        with reuse_transformers_package_metadata():
            return original(*args, **kwargs)

    setattr(setup, _SETUP_MARKER, True)
    return setup


def install():
    """Wrap the pinned native setup once, without importing Soup/TRL eagerly."""
    with _CONTEXT_LOCK:
        recognized = _native_soup_setup()
        if recognized is None:
            return False
        wrapper_type, original = recognized
        wrapper_type.setup = _make_setup_wrapper(original)
        return True
