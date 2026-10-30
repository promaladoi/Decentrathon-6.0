"""CPU contracts for recomputed Soup map metadata and untouched generic maps."""

import importlib.util
import inspect
import random
import sys
from pathlib import Path
import types
import unittest
from unittest import mock


_spec = importlib.util.spec_from_file_location(
    "_solution_tokenization_map_contract", Path(__file__).with_name("tokenization_map.py"))
_fastmap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fastmap)


def _helper(dataset_type):
    # Own synthetic helper with the pinned module/qualname and nested callback
    # surface. It permits exercising metadata routing without loading a model.
    namespace = {"__name__": "soup_cli.trainer.sft", "Dataset": dataset_type}
    exec("""
def _map_text_sft_rows(rows, *, format_row, split='train', max_length=512):
    def checked_format_row(example, row_index):
        return format_row(example)
    return Dataset.from_list(rows).map(
        checked_format_row, with_indices=True, remove_columns=['messages'])
""", namespace)
    return namespace["_map_text_sft_rows"]


class _RecordingDataset:
    calls = []
    result = object()

    @classmethod
    def from_list(cls, rows):
        return cls()

    def map(self, *args, **kwargs):
        self.calls.append((self, args, kwargs))
        return self.result


class MapMetadataTests(unittest.TestCase):
    def setUp(self):
        _RecordingDataset.calls.clear()
        self.helper = _helper(_RecordingDataset)
        self.rows_wrapper, self.map_wrapper = _fastmap._make_wrappers(
            self.helper, _RecordingDataset)

    def _callback(self):
        # Capture the exact callback the native helper creates.
        self.helper([], format_row=lambda row: row)
        return _RecordingDataset.calls.pop()[1][0]

    def test_qualified_maps_get_fresh_fingerprints_and_forced_recomputation(self):
        initial_random = random.getstate()
        with mock.patch.object(_RecordingDataset, "map", self.map_wrapper):
            for _ in range(2):
                self.assertIs(self.rows_wrapper([], format_row=lambda row: row),
                              _RecordingDataset.result)
        first, second = [event[2] for event in _RecordingDataset.calls]
        self.assertNotEqual(first["new_fingerprint"], second["new_fingerprint"])
        self.assertEqual(len(first["new_fingerprint"]), 32)
        self.assertIs(first["load_from_cache_file"], False)
        self.assertEqual(random.getstate(), initial_random)
        self.assertIsNone(_fastmap._SCOPE.get())

    def test_generic_and_unqualified_maps_keep_all_original_arguments(self):
        function = self._callback()
        dataset = _RecordingDataset()
        base_options = {"with_indices": True, "remove_columns": ["messages"]}
        # No helper scope: even the matching callback is an ordinary map.
        self.map_wrapper(dataset, function, **base_options)
        self.assertEqual(_RecordingDataset.calls.pop()[2], base_options)
        identity = self.rows_wrapper.__closure__
        identity = next(cell.cell_contents for cell in identity
                        if isinstance(cell.cell_contents, tuple))
        token = _fastmap._SCOPE.set(identity)
        try:
            variants = [
                {"num_proc": 2}, {"num_proc": True}, {"batched": True},
                {"with_indices": False}, {"remove_columns": "messages"},
                {"remove_columns": ["other"]}, {"input_columns": ["messages"]},
                {"new_fingerprint": "caller_fingerprint"}, {"cache_file_name": "caller.arrow"},
            ]
            for changes in variants:
                with self.subTest(changes=tuple(changes)):
                    options = {**base_options, **changes}
                    self.map_wrapper(dataset, function, **options)
                    self.assertEqual(_RecordingDataset.calls.pop()[2], options)
            other_callback = lambda example, index: example
            self.map_wrapper(dataset, other_callback, **base_options)
            self.assertEqual(_RecordingDataset.calls.pop()[2], base_options)

            class Subclass(_RecordingDataset):
                pass

            self.map_wrapper(Subclass(), function, **base_options)
            self.assertEqual(_RecordingDataset.calls.pop()[2], base_options)
        finally:
            _fastmap._SCOPE.reset(token)

    def test_scope_is_restored_after_the_original_map_raises(self):
        class FailureDataset(_RecordingDataset):
            def map(self, *args, **kwargs):
                raise RuntimeError("original mapping failed")

        helper = _helper(FailureDataset)
        rows_wrapper, map_wrapper = _fastmap._make_wrappers(helper, FailureDataset)
        outer = object()
        token = _fastmap._SCOPE.set(outer)
        try:
            with mock.patch.object(FailureDataset, "map", map_wrapper):
                with self.assertRaisesRegex(RuntimeError, "original mapping failed"):
                    rows_wrapper([], format_row=lambda row: row)
            self.assertIs(_fastmap._SCOPE.get(), outer)
        finally:
            _fastmap._SCOPE.reset(token)

    def test_install_is_idempotent_and_keeps_the_original_functions(self):
        datasets = types.SimpleNamespace(Dataset=_RecordingDataset)
        sft = types.SimpleNamespace(_map_text_sft_rows=self.helper)
        original_map = _RecordingDataset.map
        modules = {"datasets": datasets, "soup_cli.trainer.sft": sft}
        try:
            with mock.patch.object(_fastmap, "_versions_supported", return_value=True), \
                    mock.patch.object(_fastmap.importlib, "import_module", side_effect=modules.__getitem__):
                self.assertTrue(_fastmap.install())
                installed_map, installed_helper = _RecordingDataset.map, sft._map_text_sft_rows
                self.assertFalse(_fastmap.install())
                self.assertIs(_RecordingDataset.map, installed_map)
                self.assertIs(sft._map_text_sft_rows, installed_helper)
                self.assertIs(installed_map.__wrapped__, original_map)
                self.assertIs(installed_helper.__wrapped__, self.helper)
        finally:
            _RecordingDataset.map = original_map

    def test_unrecognized_helper_or_versions_are_not_patched(self):
        with mock.patch.object(_fastmap, "_versions_supported", return_value=False):
            self.assertFalse(_fastmap.install())
        helper = lambda rows: rows
        self.assertIsNone(_fastmap._callback_identity(helper))


@unittest.skipUnless(importlib.util.find_spec("datasets"), "needs the installed CPU datasets library")
class NativeDatasetTests(unittest.TestCase):
    def test_all_rows_recomputed_in_order_without_hashing_the_checked_callback(self):
        from datasets import Dataset, fingerprint
        import numpy as np

        helper = _helper(Dataset)
        rows_wrapper, map_wrapper = _fastmap._make_wrappers(helper, Dataset)
        callback_code = _fastmap._callback_identity(helper)[0]
        original_map, original_dumps = Dataset.map, fingerprint.dumps
        input_rows = [{"messages": str(value), "position": value} for value in (8, 3, 11, 0)]
        callback_count = 0
        checked_hashes = 0

        def formatter(row):
            nonlocal callback_count
            callback_count += 1
            value = row["position"]
            return {"input_ids": [value, value + 1], "labels": [-100, value + 1],
                    "attention_mask": [1, 1]}

        def count_dumps(value, *args, **kwargs):
            nonlocal checked_hashes
            if isinstance(value, types.FunctionType) and value.__code__ is callback_code:
                checked_hashes += 1
            return original_dumps(value, *args, **kwargs)

        reference = helper(input_rows, format_row=formatter)
        self.assertEqual(callback_count, len(input_rows))
        initial_python, initial_numpy = random.getstate(), np.random.get_state()
        torch = None
        if importlib.util.find_spec("torch"):
            import torch

        initial_torch = torch.get_rng_state() if torch is not None else None
        try:
            fingerprint.dumps = count_dumps
            Dataset.map = map_wrapper
            first = rows_wrapper(input_rows, format_row=formatter)
            second = rows_wrapper(input_rows, format_row=formatter)
        finally:
            Dataset.map, fingerprint.dumps = original_map, original_dumps
        self.assertEqual(callback_count, 3 * len(input_rows))
        self.assertEqual(checked_hashes, 0)
        self.assertEqual(first.to_list(), reference.to_list())
        self.assertEqual(second.to_list(), reference.to_list())
        self.assertEqual(first.features, reference.features)
        self.assertEqual(second.features, reference.features)
        self.assertNotEqual(first._fingerprint, second._fingerprint)
        self.assertEqual(random.getstate(), initial_python)
        actual_numpy = np.random.get_state()
        self.assertEqual(actual_numpy[0], initial_numpy[0])
        np.testing.assert_array_equal(actual_numpy[1], initial_numpy[1])
        self.assertEqual(actual_numpy[2:], initial_numpy[2:])
        if torch is not None:
            self.assertTrue(torch.equal(torch.get_rng_state(), initial_torch))
        self.assertIsNone(_fastmap._SCOPE.get())

    @unittest.skipUnless(importlib.util.find_spec("soup_cli"), "needs the pinned installed Soup helper")
    def test_original_soup_helper_keeps_valid_rows_and_validation_errors(self):
        from datasets import Dataset
        from soup_cli.trainer.sft import _map_text_sft_rows

        if not _fastmap._versions_supported():
            self.skipTest("requires Soup 0.77.0 and datasets 4.8.5")
        helper = inspect.unwrap(_map_text_sft_rows)
        rows_wrapper, map_wrapper = _fastmap._make_wrappers(helper, Dataset)
        input_rows = [{"messages": [{"role": "user", "content": "synthetic"}],
                       "position": value} for value in (7, 2, 9)]
        calls = 0

        def valid_formatter(row):
            nonlocal calls
            calls += 1
            value = row["position"]
            return {"input_ids": [1, value, value + 1], "labels": [-100, value, value + 1],
                    "attention_mask": [1, 1, 1]}

        def invalid_formatter(row):
            return {"input_ids": [1, 2], "labels": [-100, -100], "attention_mask": [1, 1]}

        reference = helper(input_rows, format_row=valid_formatter, split="train", max_length=512)
        with self.assertRaises(ValueError) as reference_error:
            helper(input_rows, format_row=invalid_formatter, split="validation", max_length=512)
        with mock.patch.object(Dataset, "map", map_wrapper):
            actual = rows_wrapper(input_rows, format_row=valid_formatter,
                                  split="train", max_length=512)
            with self.assertRaises(ValueError) as actual_error:
                rows_wrapper(input_rows, format_row=invalid_formatter,
                             split="validation", max_length=512)
        self.assertEqual(calls, 2 * len(input_rows))
        self.assertEqual(actual.to_list(), reference.to_list())
        self.assertEqual(actual.features, reference.features)
        self.assertEqual(str(actual_error.exception), str(reference_error.exception))
        self.assertIn("validation row 1:", str(actual_error.exception))
        self.assertIsNone(_fastmap._SCOPE.get())


if __name__ == "__main__":
    unittest.main()
