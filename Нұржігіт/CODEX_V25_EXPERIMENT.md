# v25: equivalent package discovery with fewer filesystem calls

Experimental, no admitted speed claim yet. Builds on v24 (native variable-length
causal attention), whose CUDA correctness suite passed before its paired run.

The separate v17 CPU profile found two expensive `packages_distributions()`
calls, from Transformers and TRL. CPython 3.13's fallback inference checks every
installed RECORD file exists before deduplicating top-level module names. v25
checks only until one existing recorded file proves each top-level name. It
still scans the original records, preserves missing-file behavior, declaration
priority, namespace distribution order, and full availability/version answers.
It has no cached scan or persisted availability data. Custom/zip/egg providers,
malformed records and other Python versions keep the standard implementation.

The CPython 3.13 implementation inspected:
https://github.com/python/cpython/blob/v3.13.9/Lib/importlib/metadata/__init__.py

Tests compare actual full installed-environment mappings and synthetic missing,
namespace, comma-quoted, native-extension and custom-provider cases. On Kaggle
the same full environment equality test runs before the untouched harness.
