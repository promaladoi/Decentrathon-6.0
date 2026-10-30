"""Install Merey's native metadata reuse after Soup's original SFT import.

The team helper is preserved in package_metadata_snapshot.py. This integration
keeps import order: no eager Soup/TRL/model import, seed change or GPU work.
The exact Soup 0.77 source runs under its own native SourceFileLoader before
the helper wraps setup(). Unsupported sources/providers retain native loading.
"""
import hashlib
import importlib.abc
import importlib.machinery
import sys

from . import package_metadata_snapshot as snapshot

TARGET = 'soup_cli.trainer.sft'
SOURCE_SHA256 = 'c9215642984edf94f3fd7b5368753a973b37da7d96b01e93257bc600b0615073'


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        return self.delegate.create_module(spec)

    def exec_module(self, module):
        # The native source and its later guards see the real loader identity.
        module.__loader__ = self.delegate
        module.__spec__.loader = self.delegate
        self.delegate.exec_module(module)
        if snapshot.install():
            print('solution: native TRL metadata snapshot installed after original Soup import', flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or type(spec.loader) is not importlib.machinery.SourceFileLoader:
            return None
        try:
            source = spec.loader.get_source(fullname)
        except (ImportError, OSError, UnicodeError):
            return None
        if source is None or hashlib.sha256(source.encode('utf-8')).hexdigest() != SOURCE_SHA256:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


def install():
    if TARGET in sys.modules:
        return snapshot.install()
    if any(isinstance(finder, _Finder) for finder in sys.meta_path):
        return False
    sys.meta_path.insert(0, _Finder())
    return True
