"""CPU-only native mapping and exact import-routing contracts."""

import copy
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import subprocess
import threading
import time
import types
import unittest
from unittest import mock


_spec = importlib.util.spec_from_file_location(
    "_solution_lazy_package_metadata_contract", Path(__file__).with_name("lazy_package_metadata.py"))
_lazy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lazy)


def _module_call(function, *, name="trl.import_utils", filename="/native/trl/import_utils.py",
                 spec_origin=None, nested=False, argument=False):
    namespace = {"__name__": name, "__file__": filename,
                 "__spec__": types.SimpleNamespace(name=name, origin=spec_origin or filename),
                 "factory": function}
    if nested:
        source = "def nested():\n    return factory()\nresult = nested()"
    else:
        source = "result = factory(1)" if argument else "result = factory()"
    exec(compile(source, filename, "exec"), namespace)
    return namespace["result"]


class MappingContractTests(unittest.TestCase):
    def test_real_mapping_operations_and_views_delegate_to_one_fresh_scan(self):
        native = {"alpha": ["Alpha-Dist"], "beta": ["Beta-Dist"]}
        calls = []
        value = _lazy._LazyPackageMapping(lambda: calls.append(1) or native)
        self.assertEqual(calls, [])
        self.assertIs(value["alpha"], native["alpha"])
        self.assertEqual(list(value), list(native))
        self.assertEqual(len(value), len(native))
        self.assertIn("alpha", value)
        self.assertEqual(list(reversed(value)), list(reversed(native)))
        self.assertEqual(repr(value), repr(native))
        self.assertEqual(str(value), str(native))
        self.assertEqual(value, native)
        self.assertEqual(native, value)
        keys, items, values = value.keys(), value.items(), value.values()
        value["gamma"] = ["Gamma-Dist"]
        self.assertIn("gamma", keys)
        self.assertIn(("gamma", ["Gamma-Dist"]), items)
        self.assertIn(["Gamma-Dist"], values)
        self.assertIs(value.get("alpha"), native["alpha"])
        self.assertEqual(value.get("missing", 9), 9)
        self.assertEqual(value.setdefault("delta", ["Delta"]), ["Delta"])
        value.update(epsilon=["Epsilon"])
        self.assertEqual(value.pop("epsilon"), ["Epsilon"])
        self.assertEqual(value.pop("missing", 7), 7)
        self.assertEqual(value.popitem(), ("delta", ["Delta"]))
        del value["gamma"]
        shallow = value.copy()
        self.assertIs(type(shallow), dict)
        self.assertIs(shallow["alpha"], native["alpha"])
        self.assertEqual(value | {"x": []}, native | {"x": []})
        self.assertEqual({"x": []} | value, {"x": []} | native)
        value |= {"z": ["Z"]}
        self.assertEqual(value["z"], ["Z"])
        self.assertEqual(calls, [1])
        value.clear()
        self.assertEqual(native, {})

    def test_error_copy_and_pickle_semantics_use_the_native_dictionary(self):
        value = _lazy._LazyPackageMapping(lambda: {"a": ["A"]})
        with self.assertRaises(KeyError) as missing:
            value["missing"]
        self.assertEqual(missing.exception.args, ("missing",))
        with self.assertRaises(KeyError):
            del value["missing"]
        with self.assertRaises(TypeError):
            value.pop("a", 1, 2)
        with self.assertRaises(TypeError):
            value[[]]
        self.assertIs(copy.copy(value)["a"], value["a"])
        deep = copy.deepcopy(value)
        self.assertEqual(deep, value)
        self.assertIsNot(deep["a"], value["a"])
        restored = pickle.loads(pickle.dumps(value))
        self.assertIs(type(restored), dict)
        self.assertEqual(restored, {"a": ["A"]})

    def test_materializes_once_across_threads_and_preserves_python_rng(self):
        calls, results = [], []
        barrier = threading.Barrier(8)
        before = random.getstate()

        def factory():
            calls.append(1)
            time.sleep(.01)
            return {"a": ["A"]}

        value = _lazy._LazyPackageMapping(factory)

        def read():
            barrier.wait()
            results.append(value["a"])

        threads = [threading.Thread(target=read) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(calls, [1])
        self.assertEqual(len(results), 8)
        self.assertTrue(all(item is results[0] for item in results))
        self.assertEqual(random.getstate(), before)

    def test_native_factory_exception_propagates_and_can_be_retried(self):
        calls = []

        def factory():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("native scan failed")
            return {"a": ["A"]}

        value = _lazy._LazyPackageMapping(factory)
        with self.assertRaisesRegex(RuntimeError, "native scan failed"):
            len(value)
        self.assertEqual(value["a"], ["A"])
        self.assertEqual(calls, [1, 1])


class RoutingContractTests(unittest.TestCase):
    def test_only_exact_module_frame_owner_thread_and_zero_args_are_deferred(self):
        factory = mock.Mock(side_effect=lambda *args: {"native": list(args)})
        origin = _lazy._canonical("/native/trl/import_utils.py")
        wrapper = _lazy._make_wrapper(factory, origin, threading.get_ident())
        lazy = _module_call(wrapper)
        self.assertIsInstance(lazy, _lazy._LazyPackageMapping)
        factory.assert_not_called()
        for options in ({"name": "transformers.utils.import_utils"},
                        {"filename": "/shadow/trl/import_utils.py"},
                        {"spec_origin": "/shadow/trl/import_utils.py"},
                        {"nested": True}, {"argument": True}):
            self.assertIs(type(_module_call(wrapper, **options)), dict)
        other_thread = []
        thread = threading.Thread(target=lambda: other_thread.append(_module_call(wrapper)))
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIs(type(other_thread[0]), dict)
        self.assertEqual(factory.call_count, 6)
        self.assertEqual(lazy["native"], [])
        self.assertEqual(factory.call_count, 7)

    def test_native_function_guard_rejects_replacement(self):
        self.assertTrue(_lazy._native_function_unchanged())
        with mock.patch.object(_lazy.metadata, "packages_distributions", lambda: {}):
            self.assertFalse(_lazy._native_function_unchanged())

    def test_context_never_imports_and_restores_on_original_caller_exception(self):
        factory = mock.Mock(return_value={"native": []})
        origin = _lazy._canonical("/native/trl/import_utils.py")
        with mock.patch.dict(sys.modules):
            sys.modules.pop("trl.import_utils", None)
            with mock.patch.object(_lazy, "_native_function_unchanged", return_value=True), \
                    mock.patch.object(_lazy, "_guarded_origin", return_value=origin), \
                    mock.patch.object(_lazy.metadata, "packages_distributions", factory), \
                    mock.patch.object(_lazy.importlib, "import_module") as imported:
                with self.assertRaisesRegex(RuntimeError, "original caller failed"):
                    with _lazy.defer_trl_package_scan() as active:
                        self.assertTrue(active)
                        imported.assert_not_called()
                        value = _module_call(_lazy.metadata.packages_distributions)
                        self.assertIsInstance(value, _lazy._LazyPackageMapping)
                        factory.assert_not_called()
                        raise RuntimeError("original caller failed")
                self.assertIs(_lazy.metadata.packages_distributions, factory)
                imported.assert_not_called()

    def test_version_and_native_spec_guards_reject_unsupported_sources(self):
        distribution = types.SimpleNamespace(version="0.29.0")
        with mock.patch.object(_lazy.metadata, "distribution", return_value=distribution), \
                mock.patch.object(_lazy.importlib.util, "find_spec") as find_spec:
            self.assertIsNone(_lazy._guarded_origin())
            find_spec.assert_not_called()
        distribution.version = "0.29.1"
        distribution.locate_file = lambda path: "/native/" + path
        parent = types.SimpleNamespace(
            loader=importlib.machinery.SourceFileLoader("trl", "/native/trl/__init__.py"),
            origin="/native/trl/__init__.py", submodule_search_locations=["/native/trl"])
        child = types.SimpleNamespace(
            loader=importlib.machinery.SourceFileLoader("trl.import_utils", "/native/trl/import_utils.py"),
            origin="/native/trl/import_utils.py", name="trl.import_utils")
        with mock.patch.object(_lazy.metadata, "distribution", return_value=distribution), \
                mock.patch.object(_lazy.importlib.util, "find_spec", return_value=parent), \
                mock.patch.object(_lazy.importlib.machinery.PathFinder, "find_spec", return_value=child):
            self.assertEqual(_lazy._guarded_origin(), _lazy._canonical(child.origin))
            child.origin = "/shadow/trl/import_utils.py"
            self.assertIsNone(_lazy._guarded_origin())
            child.origin = "/native/trl/import_utils.py"
            child.loader = object()
            self.assertIsNone(_lazy._guarded_origin())

    def test_preload_restores_function_and_is_idempotent_without_eager_scan(self):
        factory = mock.Mock(return_value={"a": ["A"]})
        origin = _lazy._canonical("/native/trl/import_utils.py")

        def importer(name):
            self.assertEqual(name, "trl.import_utils")
            module = types.SimpleNamespace(PACKAGE_DISTRIBUTION_MAPPING=_module_call(
                _lazy.metadata.packages_distributions))
            sys.modules[name] = module
            return module

        with mock.patch.dict(sys.modules):
            sys.modules.pop("trl.import_utils", None)
            with mock.patch.object(_lazy, "_native_function_unchanged", return_value=True), \
                    mock.patch.object(_lazy, "_guarded_origin", return_value=origin), \
                    mock.patch.object(_lazy.metadata, "packages_distributions", factory), \
                    mock.patch.object(_lazy.importlib, "import_module", side_effect=importer) as imported:
                self.assertTrue(_lazy.preload_trl_import_utils())
                self.assertIs(_lazy.metadata.packages_distributions, factory)
                factory.assert_not_called()
                self.assertFalse(_lazy.preload_trl_import_utils())
                self.assertEqual(imported.call_count, 1)
                self.assertEqual(sys.modules["trl.import_utils"].PACKAGE_DISTRIBUTION_MAPPING["a"], ["A"])
                self.assertEqual(factory.call_count, 1)

    def test_import_failure_restores_before_native_fallback_and_propagates_native_error(self):
        factory = mock.Mock(return_value={})
        origin = _lazy._canonical("/native/trl/import_utils.py")
        calls = []

        def importer(name):
            calls.append(_lazy.metadata.packages_distributions)
            if len(calls) == 1:
                raise RuntimeError("optimized import failed")
            self.assertIs(_lazy.metadata.packages_distributions, factory)
            raise ImportError("native failure")

        with mock.patch.dict(sys.modules):
            sys.modules.pop("trl.import_utils", None)
            with mock.patch.object(_lazy, "_native_function_unchanged", return_value=True), \
                    mock.patch.object(_lazy, "_guarded_origin", return_value=origin), \
                    mock.patch.object(_lazy.metadata, "packages_distributions", factory), \
                    mock.patch.object(_lazy.importlib, "import_module", side_effect=importer):
                with self.assertRaisesRegex(ImportError, "native failure"):
                    _lazy.preload_trl_import_utils()
                self.assertIs(_lazy.metadata.packages_distributions, factory)
        self.assertEqual(len(calls), 2)

    def test_unsupported_environment_does_not_import_or_patch(self):
        for native, origin in ((False, "/native/trl/import_utils.py"), (True, None)):
            with mock.patch.dict(sys.modules):
                sys.modules.pop("trl.import_utils", None)
                with mock.patch.object(_lazy, "_native_function_unchanged", return_value=native), \
                        mock.patch.object(_lazy, "_guarded_origin", return_value=origin), \
                        mock.patch.object(_lazy.importlib, "import_module") as imported:
                    original = _lazy.metadata.packages_distributions
                    self.assertFalse(_lazy.preload_trl_import_utils())
                    self.assertIs(_lazy.metadata.packages_distributions, original)
                    imported.assert_not_called()


@unittest.skipUnless(os.environ.get("MEREY_NATIVE_METADATA_PARITY") == "1",
                     "explicit separate CPU diagnostic with installed TRL 0.29.1")
class NativeInstalledTests(unittest.TestCase):
    def test_native_import_mapping_and_helpers_match_fresh_full_scan(self):
        # A fresh child keeps the real first-import path and library globals
        # separate from the focused tests and from any training process.
        source = r'''
import importlib
import importlib.util
import json
import sys
import time

spec = importlib.util.spec_from_file_location("_native_lazy_metadata_check", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
assert "trl.import_utils" not in sys.modules
original = helper.metadata.packages_distributions
started = time.perf_counter()
with helper.defer_trl_package_scan() as active:
    assert active, "pinned native source guard did not qualify"
    module = importlib.import_module("trl.import_utils")
import_seconds = time.perf_counter() - started
assert helper.metadata.packages_distributions is original
mapping = module.PACKAGE_DISTRIBUTION_MAPPING
assert isinstance(mapping, helper._LazyPackageMapping)
assert mapping._value is None
missing = module._is_package_available("_merey_missing_metadata_probe_7f293c", return_version=True)
assert missing == (False, "N/A")
assert mapping._value is None, "missing package check unexpectedly accessed mapping"
started = time.perf_counter()
actual = mapping.copy()
first_access_seconds = time.perf_counter() - started
expected = original()
assert actual == expected
lookups = {}
for package in ("packaging", "trl", "sys"):
    deferred_result = module._is_package_available(package, return_version=True)
    try:
        module.PACKAGE_DISTRIBUTION_MAPPING = expected
        native_result = module._is_package_available(package, return_version=True)
    finally:
        module.PACKAGE_DISTRIBUTION_MAPPING = mapping
    assert deferred_result == native_result, package
    lookups[package] = deferred_result
torch = sys.modules.get("torch")
cuda_initialized = bool(torch is not None and torch.cuda.is_initialized())
assert not cuda_initialized
print(json.dumps({"native_mapping_equal": True, "entries": len(actual),
                  "missing_check_kept_deferred": True, "helper_lookups": lookups,
                  "cuda_initialized": cuda_initialized, "import_seconds": import_seconds,
                  "first_actual_access_seconds": first_access_seconds,
                  "function_restored": helper.metadata.packages_distributions is original}))
'''
        result = subprocess.run(
            [sys.executable, "-c", source, str(Path(__file__).with_name("lazy_package_metadata.py"))],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(report["native_mapping_equal"])
        self.assertFalse(report["cuda_initialized"])
        print("native metadata parity: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    unittest.main()
