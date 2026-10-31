"""Avoid serializing Soup's tokenizer just to name a freshly recomputed map.

Only the original text-SFT helper and its exact checked callback qualify.
Every row still runs through the original formatter, checks and Arrow writer.
A fresh OS-random fingerprint disables cache reuse for this one transform;
it does not consume Python, NumPy or Torch's seeded random streams.
"""

import contextvars
import functools
import importlib
from importlib.metadata import PackageNotFoundError, version
import types
import uuid


_SCOPE = contextvars.ContextVar("soup_recomputed_text_map", default=None)
_MARKER = "_soup_recomputed_text_map"
_HELPER_MODULE = "soup_cli.trainer.sft"
_HELPER_NAME = "_map_text_sft_rows"
_CALLBACK_NAME = _HELPER_NAME + ".<locals>.checked_format_row"


def _versions_supported():
    try:
        return version("soup-cli") == "0.77.0" and version("datasets") == "4.8.5"
    except PackageNotFoundError:
        return False


def _callback_identity(helper):
    if (not isinstance(helper, types.FunctionType)
            or helper.__module__ != _HELPER_MODULE or helper.__qualname__ != _HELPER_NAME):
        return None
    candidates = [value for value in helper.__code__.co_consts
                  if isinstance(value, types.CodeType) and value.co_qualname == _CALLBACK_NAME]
    return (candidates[0], _HELPER_MODULE, _CALLBACK_NAME) if len(candidates) == 1 else None


def _make_wrappers(helper, dataset_type):
    """Return wrappers without changing library state, for focused CPU tests."""
    identity = _callback_identity(helper)
    if identity is None:
        raise ValueError("unsupported Soup text-map helper")
    original_map = dataset_type.map

    @functools.wraps(helper)
    def rows_map(*args, **kwargs):
        token = _SCOPE.set(identity)
        try:
            return helper(*args, **kwargs)
        finally:
            _SCOPE.reset(token)

    @functools.wraps(original_map)
    def dataset_map(dataset, *args, **kwargs):
        function = args[0] if len(args) == 1 else kwargs.get("function") if not args else None
        num_proc = kwargs.get("num_proc")
        remove_columns = kwargs.get("remove_columns")
        qualified = (
            _SCOPE.get() is identity and type(dataset) is dataset_type
            and len(args) <= 1 and not (args and "function" in kwargs)
            and isinstance(function, types.FunctionType)
            and function.__code__ is identity[0]
            and function.__module__ == identity[1] and function.__qualname__ == identity[2]
            and kwargs.get("with_indices") is True
            and kwargs.get("batched", False) is False
            and kwargs.get("with_rank", False) is False
            and kwargs.get("input_columns") is None
            and type(remove_columns) is list and len(remove_columns) == 1
            and type(remove_columns[0]) is str and remove_columns[0] == "messages"
            and (num_proc is None or (type(num_proc) is int and num_proc == 1))
            and kwargs.get("new_fingerprint") is None
            and kwargs.get("cache_file_name") is None
        )
        if not qualified:
            return original_map(dataset, *args, **kwargs)
        options = dict(kwargs)
        options["new_fingerprint"] = uuid.uuid4().hex
        options["load_from_cache_file"] = False
        return original_map(dataset, *args, **options)

    setattr(rows_map, _MARKER, True)
    setattr(dataset_map, _MARKER, True)
    return rows_map, dataset_map


def install():
    """Install once on the pinned native helper; unrelated maps remain untouched."""
    if not _versions_supported():
        return False
    datasets = importlib.import_module("datasets")
    sft = importlib.import_module(_HELPER_MODULE)
    dataset_type = datasets.Dataset
    helper = getattr(sft, _HELPER_NAME, None)
    if (getattr(helper, _MARKER, False) or getattr(dataset_type.map, _MARKER, False)
            or _callback_identity(helper) is None):
        return False
    rows_map, dataset_map = _make_wrappers(helper, dataset_type)
    dataset_type.map = dataset_map
    setattr(sft, _HELPER_NAME, rows_map)
    return True
