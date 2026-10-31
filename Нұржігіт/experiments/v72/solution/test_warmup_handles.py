"""Handle warm-up reads the lazy property and never invokes the launcher."""

import importlib.util
from pathlib import Path
import unittest


# warmup.py itself uses only stdlib. Load it without importing solution's
# torch/Triton modules so these contract tests also run on a CPU-only host.
_spec = importlib.util.spec_from_file_location(
    "_solution_warmup_contract", Path(__file__).with_name("warmup.py"))
_warmup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_warmup)


class LoadCompiledTests(unittest.TestCase):
    def test_reads_lazy_handles_without_calling_launcher(self):
        class Compiled:
            reads = 0
            launches = 0

            @property
            def run(self):
                self.reads += 1

                def launch(*args, **kwargs):
                    self.launches += 1
                    raise AssertionError("warm-up must not launch arithmetic")

                return launch

        compiled = Compiled()
        self.assertIs(_warmup.load_compiled(compiled), compiled)
        self.assertEqual(compiled.reads, 1)
        self.assertEqual(compiled.launches, 0)

    def test_accepts_absent_handles_from_mock_warmups(self):
        for compiled in (None, object()):
            with self.subTest(compiled_type=type(compiled).__name__):
                self.assertIs(_warmup.load_compiled(compiled), compiled)

    def test_handle_load_failure_reaches_normal_warmup_fallback(self):
        class Compiled:
            @property
            def run(self):
                raise RuntimeError("handle load failed")

        with self.assertRaisesRegex(RuntimeError, "handle load failed"):
            _warmup.load_compiled(Compiled())


if __name__ == "__main__":
    unittest.main()
