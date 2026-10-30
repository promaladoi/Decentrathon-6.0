"""Native import execution/order, source guards and exception propagation."""
import hashlib
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types

import pytest

from solution import deferred_metadata as integration


def test_real_import_runs_native_body_then_installs_and_restores_native_loader(tmp_path, monkeypatch):
    name = '_nurzhigit_deferred_metadata_contract'
    source = 'value = 42\n'
    (tmp_path/(name+'.py')).write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(integration, 'TARGET', name)
    monkeypatch.setattr(integration, 'SOURCE_SHA256', hashlib.sha256(source.encode()).hexdigest())
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    seen = []
    def install():
        module = sys.modules[name]
        assert module.value == 42
        assert type(module.__loader__) is importlib.machinery.SourceFileLoader
        assert module.__spec__.loader is module.__loader__
        seen.append(module)
        return True
    monkeypatch.setattr(integration.snapshot, 'install', install)
    try:
        assert integration.install()
        assert not integration.install()
        assert name not in sys.modules
        module = importlib.import_module(name)
        assert seen == [module]
        assert not any(isinstance(f, integration._Finder) for f in sys.meta_path)
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize('changed_source,custom_loader', [(True, False), (False, True)])
def test_unrecognized_source_or_provider_uses_original_import(tmp_path, monkeypatch, changed_source, custom_loader):
    name = integration.TARGET
    p = tmp_path/'sft.py'
    p.write_text('value = 10\n')
    spec = importlib.util.spec_from_file_location(name, p)
    if custom_loader:
        class Custom(importlib.machinery.SourceFileLoader):
            pass
        spec.loader = Custom(name, str(p))
    monkeypatch.setattr(integration, 'SOURCE_SHA256', '0'*64 if changed_source else hashlib.sha256(p.read_bytes()).hexdigest())
    original_loader = spec.loader
    monkeypatch.setattr(importlib.machinery.PathFinder, 'find_spec', lambda *a: spec)
    finder = integration._Finder()
    monkeypatch.setattr(sys, 'meta_path', [finder])
    assert finder.find_spec(name) is None
    assert spec.loader is original_loader
    assert sys.meta_path == []


def test_other_imports_do_not_touch_finder_or_load_modules(monkeypatch):
    finder = integration._Finder()
    monkeypatch.setattr(sys, 'meta_path', [finder])
    assert finder.find_spec('unrelated.module') is None
    assert sys.meta_path == [finder]


def test_original_import_exception_propagates_without_installing(monkeypatch):
    sentinel = RuntimeError('native import failed')
    class Delegate:
        def exec_module(self, module):
            raise sentinel
    delegate = Delegate()
    module = types.ModuleType(integration.TARGET)
    module.__spec__ = types.SimpleNamespace(loader=None)
    calls = []
    monkeypatch.setattr(integration.snapshot, 'install', lambda: calls.append(1))
    with pytest.raises(RuntimeError) as failed:
        integration._Loader(delegate).exec_module(module)
    assert failed.value is sentinel and calls == []
    assert module.__spec__.loader is delegate


def test_already_loaded_module_uses_helper_without_new_finder(monkeypatch):
    monkeypatch.setitem(sys.modules, integration.TARGET, types.ModuleType(integration.TARGET))
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    before = list(sys.meta_path)
    monkeypatch.setattr(integration.snapshot, 'install', lambda: False)
    assert not integration.install()
    assert sys.meta_path == before


def test_source_hash_matches_the_installed_original_soup():
    spec = importlib.util.find_spec(integration.TARGET)
    source = spec.loader.get_source(integration.TARGET)
    assert hashlib.sha256(source.encode()).hexdigest() == integration.SOURCE_SHA256
