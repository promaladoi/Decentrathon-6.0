"""CPU contracts and opt-in fresh native parity for same-process metadata reuse."""

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import types
import unittest
from unittest import mock


_spec = importlib.util.spec_from_file_location(
    "_solution_package_metadata_snapshot_contract", Path(__file__).with_name("package_metadata_snapshot.py"))
_snapshot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_snapshot)


def _native_transformers(mapping=None):
    name = "transformers.utils.import_utils"
    filename = "/native/transformers/utils/import_utils.py"
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__spec__ = types.SimpleNamespace(
        name=name, origin=filename, loader=importlib.machinery.SourceFileLoader(name, filename))
    module.PACKAGE_DISTRIBUTION_MAPPING = {"alpha": ["Alpha-Dist"], "beta": ["Beta-Dist"]} \
        if mapping is None else mapping
    exec(compile("def _is_package_available(pkg_name, return_version=False):\n"
                 "    return PACKAGE_DISTRIBUTION_MAPPING[pkg_name]\n", filename, "exec"), vars(module))
    return module


def _module_call(function, *, name="trl.import_utils", filename="/native/trl/import_utils.py",
                 spec_origin=None, nested=False, argument=False):
    module = types.ModuleType(name)
    namespace = vars(module)
    namespace.update({"__file__": filename,
                      "__spec__": types.SimpleNamespace(
                          name=name, origin=spec_origin or filename,
                          loader=importlib.machinery.SourceFileLoader(name, filename)),
                      "factory": function})
    source = "def nested():\n    return factory()\nresult = nested()" if nested else \
        "result = factory(1)" if argument else "result = factory()"
    with mock.patch.dict(sys.modules, {name: module}):
        exec(compile(source, filename, "exec"), namespace)
    return namespace["result"]


def _distribution(package, version):
    return types.SimpleNamespace(version=version, locate_file=lambda path: "/native/" + path)


class SnapshotRoutingTests(unittest.TestCase):
    def test_exact_call_gets_a_plain_independent_deep_copy_without_native_scan(self):
        shared = ["Alpha-Dist"]
        module = _native_transformers({"alpha": shared, "alias": shared, "empty": []})
        source = module, module.PACKAGE_DISTRIBUTION_MAPPING, module._is_package_available
        native = mock.Mock(return_value={"native": ["Native-Dist"]})
        wrapper = _snapshot._make_wrapper(
            native, _snapshot._canonical("/native/trl/import_utils.py"), source, threading.get_ident())
        before = random.getstate()
        with mock.patch.dict(sys.modules, {"transformers.utils.import_utils": module}):
            first, second = _module_call(wrapper), _module_call(wrapper)
        native.assert_not_called()
        self.assertIs(type(first), dict)
        self.assertEqual(first, source[1])
        self.assertIsNot(first, source[1])
        self.assertIsNot(first["alpha"], shared)
        self.assertIs(first["alpha"], first["alias"])
        self.assertIsNot(first["empty"], source[1]["empty"])
        self.assertIsNot(first["alpha"], second["alpha"])
        first["alpha"].append("changed")
        first["new"] = ["New"]
        self.assertEqual(shared, ["Alpha-Dist"])
        self.assertEqual(second, source[1])
        self.assertEqual(random.getstate(), before)

    def test_all_other_callers_threads_args_and_stale_sources_use_native(self):
        module = _native_transformers()
        source = module, module.PACKAGE_DISTRIBUTION_MAPPING, module._is_package_available
        native = mock.Mock(side_effect=lambda *args: {"native": list(args)})
        wrapper = _snapshot._make_wrapper(
            native, _snapshot._canonical("/native/trl/import_utils.py"), source, threading.get_ident())
        with mock.patch.dict(sys.modules, {"transformers.utils.import_utils": module}):
            for options in ({"name": "transformers.utils.import_utils"},
                            {"filename": "/shadow/trl/import_utils.py"},
                            {"spec_origin": "/shadow/trl/import_utils.py"},
                            {"nested": True}, {"argument": True}):
                self.assertIn("native", _module_call(wrapper, **options))
            other_thread = []
            thread = threading.Thread(target=lambda: other_thread.append(_module_call(wrapper)))
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertIn("native", other_thread[0])
            module.PACKAGE_DISTRIBUTION_MAPPING = dict(source[1])
            self.assertIn("native", _module_call(wrapper))
            module.PACKAGE_DISTRIBUTION_MAPPING = source[1]
            module._is_package_available = lambda package: True
            self.assertIn("native", _module_call(wrapper))
        self.assertEqual(native.call_count, 8)

    def test_native_errors_and_source_copy_failure_keep_native_semantics(self):
        module = _native_transformers()
        source = module, module.PACKAGE_DISTRIBUTION_MAPPING, module._is_package_available
        native = mock.Mock(side_effect=RuntimeError("native scan failed"))
        wrapper = _snapshot._make_wrapper(
            native, _snapshot._canonical("/native/trl/import_utils.py"), source, threading.get_ident())
        with mock.patch.dict(sys.modules, {"transformers.utils.import_utils": module}), \
                mock.patch.object(_snapshot.copy, "deepcopy", side_effect=RuntimeError("copy failed")):
            with self.assertRaisesRegex(RuntimeError, "native scan failed"):
                _module_call(wrapper)
        native.assert_called_once_with()


class NativeGuardTests(unittest.TestCase):
    def test_missing_transformers_is_never_imported_or_looked_up(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop("transformers.utils.import_utils", None)
            with mock.patch.object(_snapshot.metadata, "distribution") as distribution:
                self.assertIsNone(_snapshot._transformers_source())
                distribution.assert_not_called()

    def test_transformers_guard_requires_pinned_native_file_code_globals_and_plain_map(self):
        module = _native_transformers()
        distribution = _distribution("transformers", "5.16.1")
        with mock.patch.dict(sys.modules, {"transformers.utils.import_utils": module}), \
                mock.patch.object(_snapshot.metadata, "distribution", return_value=distribution):
            self.assertIs(_snapshot._transformers_source()[1], module.PACKAGE_DISTRIBUTION_MAPPING)
            distribution.version = "5.16.0"
            self.assertIsNone(_snapshot._transformers_source())
            distribution.version = "5.16.1"
            module.__spec__.origin = "/shadow/transformers/utils/import_utils.py"
            self.assertIsNone(_snapshot._transformers_source())
            module.__spec__.origin = module.__file__
            module.__spec__.loader = object()
            self.assertIsNone(_snapshot._transformers_source())
        for mapping in ({1: ["name"]}, {"name": ("dist",)}, {"name": [1]}):
            module = _native_transformers(mapping)
            with mock.patch.dict(sys.modules, {"transformers.utils.import_utils": module}), \
                    mock.patch.object(_snapshot.metadata, "distribution", return_value=distribution):
                self.assertIsNone(_snapshot._transformers_source())
        module = _native_transformers()
        module._is_package_available = lambda name: True
        with mock.patch.dict(sys.modules, {"transformers.utils.import_utils": module}), \
                mock.patch.object(_snapshot.metadata, "distribution", return_value=distribution):
            self.assertIsNone(_snapshot._transformers_source())

    def test_transformers_function_code_and_globals_must_belong_to_the_native_module(self):
        distribution = _distribution("transformers", "5.16.1")
        module = _native_transformers()
        original = module._is_package_available
        foreign = dict(vars(module))
        module._is_package_available = types.FunctionType(original.__code__, foreign)
        with mock.patch.dict(sys.modules, {"transformers.utils.import_utils": module}), \
                mock.patch.object(_snapshot.metadata, "distribution", return_value=distribution):
            self.assertIsNone(_snapshot._transformers_source())
        exec(compile("def _is_package_available(pkg_name, return_version=False):\n"
                     "    return PACKAGE_DISTRIBUTION_MAPPING[pkg_name]\n",
                     "/shadow/transformers/utils/import_utils.py", "exec"), vars(module))
        with mock.patch.dict(sys.modules, {"transformers.utils.import_utils": module}), \
                mock.patch.object(_snapshot.metadata, "distribution", return_value=distribution):
            self.assertIsNone(_snapshot._transformers_source())

    def test_stdlib_function_identity_and_trl_version_source_guards(self):
        self.assertTrue(_snapshot._native_function_unchanged())
        with mock.patch.object(_snapshot.metadata, "packages_distributions", lambda: {}):
            self.assertFalse(_snapshot._native_function_unchanged())
        distribution = _distribution("trl", "0.29.0")
        with mock.patch.object(_snapshot.metadata, "distribution", return_value=distribution), \
                mock.patch.object(_snapshot.importlib.util, "find_spec") as find:
            self.assertIsNone(_snapshot._trl_origin())
            find.assert_not_called()
        distribution.version = "0.29.1"
        parent = types.SimpleNamespace(
            loader=importlib.machinery.SourceFileLoader("trl", "/native/trl/__init__.py"),
            origin="/native/trl/__init__.py", submodule_search_locations=["/native/trl"])
        child = types.SimpleNamespace(
            loader=importlib.machinery.SourceFileLoader("trl.import_utils", "/native/trl/import_utils.py"),
            origin="/native/trl/import_utils.py", name="trl.import_utils")
        with mock.patch.object(_snapshot.metadata, "distribution", return_value=distribution), \
                mock.patch.object(_snapshot.importlib.util, "find_spec", return_value=parent), \
                mock.patch.object(_snapshot.importlib.machinery.PathFinder, "find_spec", return_value=child):
            self.assertEqual(_snapshot._trl_origin(), _snapshot._canonical(child.origin))
            child.origin = "/shadow/trl/import_utils.py"
            self.assertIsNone(_snapshot._trl_origin())


class ContextTests(unittest.TestCase):
    def test_context_preserves_import_order_restores_and_keeps_helpers_unchanged(self):
        module = _native_transformers()
        source = module, module.PACKAGE_DISTRIBUTION_MAPPING, module._is_package_available
        native = mock.Mock(return_value={"native": []})
        with mock.patch.dict(sys.modules, {"transformers.utils.import_utils": module}):
            sys.modules.pop("trl.import_utils", None)
            with mock.patch.object(_snapshot, "_native_function_unchanged", return_value=True), \
                    mock.patch.object(_snapshot, "_transformers_source", return_value=source), \
                    mock.patch.object(_snapshot, "_trl_origin", return_value=_snapshot._canonical("/native/trl/import_utils.py")), \
                    mock.patch.object(_snapshot.metadata, "packages_distributions", native):
                with self.assertRaisesRegex(RuntimeError, "original caller failed"):
                    with _snapshot.reuse_transformers_package_metadata() as active:
                        self.assertTrue(active)
                        value = _module_call(_snapshot.metadata.packages_distributions)
                        self.assertEqual(value, source[1])
                        self.assertIs(module._is_package_available, source[2])
                        self.assertIs(module.PACKAGE_DISTRIBUTION_MAPPING, source[1])
                        raise RuntimeError("original caller failed")
                self.assertIs(_snapshot.metadata.packages_distributions, native)
                native.assert_not_called()

    def test_unsupported_and_already_imported_contexts_do_not_patch(self):
        for loaded, native, source in ((True, True, None), (False, False, None), (False, True, None)):
            with mock.patch.dict(sys.modules):
                sys.modules.pop("trl.import_utils", None)
                if loaded:
                    sys.modules["trl.import_utils"] = types.ModuleType("trl.import_utils")
                with mock.patch.object(_snapshot, "_native_function_unchanged", return_value=native), \
                        mock.patch.object(_snapshot, "_transformers_source", return_value=source):
                    original = _snapshot.metadata.packages_distributions
                    with _snapshot.reuse_transformers_package_metadata() as active:
                        self.assertFalse(active)
                        self.assertIs(_snapshot.metadata.packages_distributions, original)
                    self.assertIs(_snapshot.metadata.packages_distributions, original)


@unittest.skipUnless(os.environ.get("MEREY_NATIVE_METADATA_SNAPSHOT_PARITY") == "1",
                     "explicit separate CPU diagnostic with installed pinned packages")
class NativeInstalledTests(unittest.TestCase):
    def test_native_transformers_snapshot_matches_fresh_scan_and_native_trl_helpers(self):
        source = r'''
import importlib
import importlib.util
import json
import sys
import time
import transformers

tf = sys.modules["transformers.utils.import_utils"]
source_mapping = tf.PACKAGE_DISTRIBUTION_MAPPING
spec = importlib.util.spec_from_file_location("_native_metadata_snapshot_check", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
assert "trl.import_utils" not in sys.modules
original = helper.metadata.packages_distributions
native_scan_callers = []
def profile(frame, event, arg):
    if event == "call" and frame.f_code is original.__code__:
        caller = frame.f_back
        native_scan_callers.append({"module": caller.f_globals.get("__name__"),
                                    "function": caller.f_code.co_name})
previous_profile = sys.getprofile()
started = time.perf_counter()
try:
    sys.setprofile(profile)
    with helper.reuse_transformers_package_metadata() as active:
        assert active, "pinned native source guards did not qualify"
        module = importlib.import_module("trl.import_utils")
finally:
    sys.setprofile(previous_profile)
import_seconds = time.perf_counter() - started
assert helper.metadata.packages_distributions is original
mapping = module.PACKAGE_DISTRIBUTION_MAPPING
assert type(mapping) is dict and mapping is not source_mapping
assert mapping == source_mapping
assert all(mapping[key] is not names for key, names in source_mapping.items())
started = time.perf_counter()
expected = original()
fresh_scan_seconds = time.perf_counter() - started
assert mapping == expected, "fixed-environment native snapshot differs from a fresh native scan"
lookups = {}
for package in ("transformers", "torch", "packaging", "sys", "_merey_missing_metadata_probe_7f293c"):
    actual = module._is_package_available(package, return_version=True)
    try:
        module.PACKAGE_DISTRIBUTION_MAPPING = expected
        native = module._is_package_available(package, return_version=True)
    finally:
        module.PACKAGE_DISTRIBUTION_MAPPING = mapping
    assert actual == native, package
    lookups[package] = actual
assert not any(caller["module"] == "trl.import_utils" for caller in native_scan_callers)
torch = sys.modules.get("torch")
cuda_initialized = bool(torch is not None and torch.cuda.is_initialized())
assert not cuda_initialized
print(json.dumps({"native_mapping_equal": True, "independent_dict_and_lists": True,
                  "entries": len(mapping), "helper_lookups": lookups,
                  "cuda_initialized": cuda_initialized, "trl_import_seconds": import_seconds,
                  "fresh_native_scan_seconds": fresh_scan_seconds,
                  "native_scan_callers_during_trl_import": native_scan_callers,
                  "function_restored": helper.metadata.packages_distributions is original}))
'''
        result = subprocess.run(
            [sys.executable, "-c", source, str(Path(__file__).with_name("package_metadata_snapshot.py"))],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(report["native_mapping_equal"])
        self.assertTrue(report["independent_dict_and_lists"])
        self.assertFalse(report["cuda_initialized"])
        print("native metadata snapshot parity: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    unittest.main()
