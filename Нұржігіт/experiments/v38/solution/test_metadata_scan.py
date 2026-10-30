import csv
import importlib.metadata as metadata
import inspect
from pathlib import Path
import sys

import pytest

from .metadata_scan import make_inferred, patch_metadata_scan, record_names


def get_name(path):
    # CPython 3.13 top-level-name contract; also exercises it on the local 3.12 CPU.
    return path.parts[0] if len(path.parts) > 1 else inspect.getmodulename(path) or str(path)


def fixture_dist(root, names, existing=(), declared=None):
    meta = root/'Example-1.0.dist-info'
    meta.mkdir(parents=True, exist_ok=True)
    (meta/'METADATA').write_text('Name: Example\nVersion: 1.0\n')
    with (meta/'RECORD').open('w', newline='') as out:
        csv.writer(out).writerows((n, '', '') for n in names)
    for name in existing:
        p = root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('')
    if declared:
        (meta/'top_level.txt').write_text(declared)
    return metadata.PathDistribution(meta)


def reference(dist):
    return {get_name(p) for p in dist.files or () if '.' not in get_name(p)}


@pytest.mark.parametrize('names,existing', [
    (['foo/__init__.py','foo/a.py','foo/missing.py'], ['foo/a.py']),
    (['foo/missing.py','bar/a.py'], ['bar/a.py']),
    (['foo/a.py','bar/missing.py'], []),
    (['module.py','module.pyc','another.pyi'], ['module.py','module.pyc']),
    (['thing','thing.pth','data.json'], ['thing','thing.pth','data.json']),
    (['native.cpython-313-x86_64-linux-gnu.so','native.py'], ['native.py']),
    (['foo/a,b.py','space name/mod.py'], ['foo/a,b.py','space name/mod.py']),
    (['./foo/a.py','foo//a.py','./bar.py'], ['foo/a.py','bar.py']),
    (['missing/a.py','missing/b.py','ok/a.py'], ['ok/a.py']),
    (['Example-1.0.dist-info/METADATA','__pycache__/other.pyc'], []),
])
def test_same_inference_and_missing_file_semantics(tmp_path, names, existing):
    dist=fixture_dist(tmp_path,names,existing)
    assert record_names(dist,dist._read_files_distinfo(),get_name)==reference(dist)


def test_one_existing_file_per_name_and_no_global_cache(tmp_path, monkeypatch):
    names=[f'foo/{i}.py' for i in range(300)]
    dist=fixture_dist(tmp_path,names,names[:1])
    calls=[]
    locate=dist.locate_file
    monkeypatch.setattr(dist,'locate_file',lambda n: calls.append(n) or locate(n))
    assert record_names(dist,dist._read_files_distinfo(),get_name)=={'foo'}
    assert calls==names[:1]
    (tmp_path/names[0]).unlink()
    assert record_names(dist,dist._read_files_distinfo(),get_name)==set()
    assert len(calls)==301


def test_custom_provider_uses_its_own_files():
    class Custom(metadata.Distribution):
        def read_text(self,n): return ''
        def locate_file(self,n): raise AssertionError('custom files must be used')
        @property
        def files(self): return [metadata.PackagePath('custom/name.py')]
    assert make_inferred(reference,get_name)(Custom())=={'custom'}


@pytest.mark.parametrize('row', ['foo/a.py,,bad','foo/a.py,,,unexpected'])
def test_malformed_record_defers_to_standard_reader(tmp_path,row):
    dist=fixture_dist(tmp_path,['foo/a.py'],['foo/a.py'])
    (dist._path/'RECORD').write_text(row)
    called=[]
    assert make_inferred(lambda d: called.append(d) or {'fallback'},get_name)(dist)=={'fallback'}
    assert called==[dist]


def test_full_installed_mapping_unchanged_and_scan_is_fresh(monkeypatch):
    expected=metadata.packages_distributions()
    original=metadata._top_level_inferred
    # Use the running interpreter's own one-file inference for its version.
    if hasattr(metadata,'_get_toplevel_name'):
        running_name=metadata._get_toplevel_name
    else:
        running_name=lambda p: p.parts[0] if len(p.parts)>1 else inspect.getmodulename(p)
    monkeypatch.setattr(metadata,'_top_level_inferred',make_inferred(original,running_name))
    assert metadata.packages_distributions()==expected
    assert metadata.packages_distributions()==expected


def test_patch_version_guard_and_idempotence(monkeypatch):
    monkeypatch.setattr(sys,'version_info',(3,12,0))
    assert patch_metadata_scan() is False
    monkeypatch.setattr(sys,'version_info',(3,13,0))
    monkeypatch.setattr(metadata,'_get_toplevel_name',get_name,raising=False)
    # This test preserves the real stdlib function; 3.12 declines its other schema.
    if sys.version_info[:2]==(3,13) and 'map' in metadata._top_level_inferred.__code__.co_names:
        monkeypatch.setattr(metadata,'_top_level_inferred',metadata._top_level_inferred)
        assert patch_metadata_scan()
        patched=metadata._top_level_inferred
        assert patch_metadata_scan() and metadata._top_level_inferred is patched
    else:
        assert patch_metadata_scan() is False
