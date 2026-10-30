"""CPU-only wall-clock diagnostic of the installed Soup tokenization helpers.

Run in a separate process after GPU timing has finished, for example:
    python hash_tokenization_diagnostic.py --repo /path/to/Мерей --out /path/to/hash-report.json

The original loader, formatter, Dataset.map, fingerprint dumps, and TRL
preparation execute normally. No model or solution package is loaded; no
fingerprint, cache policy, tokenization, or data-order optimization is applied.
Only metadata, durations, byte counts, and ordered output digests are reported.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types


class _DiscardOutput:
    """Do not retain or forward third-party messages that could contain rows."""

    def write(self, value):
        return len(value)

    def flush(self):
        pass

    def isatty(self):
        return False


def _type_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _describe(value, depth=0, seen=None):
    """Inspect callable identity and closure types, never values or repr()."""
    result = {"type": _type_name(value)}
    seen = set() if seen is None else seen
    if id(value) in seen or depth >= 4:
        return result
    seen.add(id(value))
    if isinstance(value, types.MethodType):
        result["owner_type"] = _type_name(value.__self__)
        result["function"] = _describe(value.__func__, depth + 1, seen)
    elif isinstance(value, types.FunctionType):
        result["callable"] = f"{value.__module__}.{value.__qualname__}"
        closure = []
        for name, cell in zip(value.__code__.co_freevars, value.__closure__ or ()):
            try:
                item = {"name": name, **_describe(cell.cell_contents, depth + 1, seen)}
            except ValueError:  # empty closure cell
                item = {"name": name, "type": "empty_cell"}
            closure.append(item)
        result["freevars"] = closure
    return result


def _source_receipt(function):
    return {
        "callable": f"{function.__module__}.{function.__qualname__}",
        "source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
    }


class _Recorder:
    def __init__(self, fingerprint, dataset_class):
        self.fingerprint, self.dataset_class = fingerprint, dataset_class
        self.original_dumps = fingerprint.dumps
        self.original_map = dataset_class.map
        self.phase = "setup"
        self.dumps_events, self.map_events, self.phases = [], [], []
        self.map_stack = []

    @contextlib.contextmanager
    def measure(self, phase):
        previous = self.phase
        self.phase = phase
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases.append({"phase": phase, "wall_s": time.perf_counter() - started})
            self.phase = previous

    def install(self):
        recorder = self

        def dumps(value, *args, **kwargs):
            started = time.perf_counter()
            status = "ok"
            size = None
            try:
                output = recorder.original_dumps(value, *args, **kwargs)
                size = len(output)
                return output
            except BaseException as exc:
                status = _type_name(exc)
                raise
            finally:
                elapsed = time.perf_counter() - started
                recorder.dumps_events.append({
                    "phase": recorder.phase,
                    "map_id": recorder.map_stack[-1] if recorder.map_stack else None,
                    "wall_s": elapsed,
                    "serialized_bytes": size,
                    "status": status,
                    "object": _describe(value),
                })

        def dataset_map(dataset, *args, **kwargs):
            function = args[0] if args else kwargs.get("function")
            map_id = len(recorder.map_events)
            event = {
                "id": map_id, "phase": recorder.phase,
                "input_rows": len(dataset), "function": _describe(function),
                "batched": kwargs.get("batched", False),
                "with_indices": kwargs.get("with_indices", False),
                "num_proc": kwargs.get("num_proc"),
                "explicit_fingerprint": kwargs.get("new_fingerprint") is not None,
            }
            recorder.map_events.append(event)
            recorder.map_stack.append(map_id)
            started = time.perf_counter()
            try:
                output = recorder.original_map(dataset, *args, **kwargs)
                event["output_rows"] = len(output)
                event["status"] = "ok"
                return output
            except BaseException as exc:
                event["status"] = _type_name(exc)
                raise
            finally:
                event["wall_s"] = time.perf_counter() - started
                recorder.map_stack.pop()

        self.fingerprint.dumps = dumps
        self.dataset_class.map = dataset_map

    def restore(self):
        self.fingerprint.dumps = self.original_dumps
        self.dataset_class.map = self.original_map

    def report(self):
        groups = {}
        for event in self.dumps_events:
            key = (event["phase"], event["object"].get("callable", event["object"]["type"]))
            group = groups.setdefault(key, {"phase": key[0], "object": key[1],
                                            "calls": 0, "wall_s": 0.0, "serialized_bytes": 0})
            group["calls"] += 1
            group["wall_s"] += event["wall_s"]
            group["serialized_bytes"] += event["serialized_bytes"] or 0
        return {"phases": self.phases, "maps": self.map_events,
                "dumps": self.dumps_events,
                "dumps_summary": sorted(groups.values(), key=lambda item: -item["wall_s"])}


def _dataset_summary(dataset):
    digest = hashlib.sha256()
    tokens, supervised = 0, 0
    for row in dataset:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
        tokens += len(row.get("input_ids", []))
        supervised += sum(int(value) != -100 for value in row.get("labels", []))
    return {"rows": len(dataset), "columns": list(dataset.column_names),
            "tokens": tokens, "supervised_labels": supervised,
            "ordered_rows_sha256": digest.hexdigest()}


def _run(args, report):
    # Set before any third-party import. This is a standalone process, so the
    # caller's training environment and GPU visibility are not modified.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    repo = Path(args.repo).resolve()
    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else repo / config_path
    os.chdir(repo)

    from datasets import Dataset, disable_progress_bars
    from datasets import fingerprint
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.data.sft_format import build_format_row
    from soup_cli.trainer.sft import _ensure_assistant_masks, _map_text_sft_rows
    from transformers import AutoTokenizer

    disable_progress_bars()
    cfg = load_config(config_path, raise_unreadable=True)
    if cfg.task != "sft" or cfg.data.format != "alpaca" or cfg.data.val_split != 0:
        raise ValueError("diagnostic requires the fixed local Alpaca SFT task")
    if args.train_file:
        train_path = Path(args.train_file)
        cfg.data.train = str(train_path if train_path.is_absolute() else repo / train_path)
    if not isinstance(cfg.data.train, str) or not Path(cfg.data.train).is_file():
        raise ValueError("diagnostic requires an existing local training JSON file")
    loaded = load_dataset(cfg.data)
    rows = loaded["train"]
    if len(rows) != args.expected_rows:
        raise ValueError("loaded row count does not match --expected-rows")
    tokenizer_kwargs = {"trust_remote_code": False,
                        "local_files_only": not args.allow_tokenizer_download}
    if args.revision:
        tokenizer_kwargs["revision"] = args.revision
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base, **tokenizer_kwargs)
    report["tokenizer_load_wall_s"] = time.perf_counter() - started
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    report["tokenizer"] = {"type": _type_name(tokenizer),
                           "resolved_commit": tokenizer.init_kwargs.get("_commit_hash")}
    report["sources"] = [_source_receipt(function) for function in
                         (load_dataset, build_format_row, _map_text_sft_rows,
                          _ensure_assistant_masks, Dataset.map, fingerprint.dumps)]
    report["versions"] = {name: importlib.metadata.version(name) for name in
                          ("soup-cli", "datasets", "transformers", "trl", "torch")}
    recorder = _Recorder(fingerprint, Dataset)
    report["passes"] = []
    recorder.install()
    try:
        for repeat in range(args.repeats):
            prefix = f"pass_{repeat + 1}"
            with recorder.measure(prefix + ".build_format_row"):
                formatter = build_format_row(tokenizer=tokenizer, data_cfg=cfg.data,
                                             training_cfg=cfg.training, console=None)
            with recorder.measure(prefix + ".soup_map_text_sft_rows"):
                dataset = _map_text_sft_rows(rows, format_row=formatter, split="train",
                                             max_length=cfg.data.max_length)
            with recorder.measure(prefix + ".soup_assistant_masks"):
                dataset = _ensure_assistant_masks(dataset)
            if not args.skip_trl_prepare:
                from trl import SFTConfig, SFTTrainer

                # _prepare_dataset needs only these configuration fields. Read
                # installed TRL defaults without constructing training args,
                # accelerator, a model, or a trainer instance.
                names = ("dataset_num_proc", "dataset_text_field", "assistant_only_loss",
                         "shuffle_dataset", "packing_strategy", "use_liger_kernel")
                defaults = {name: SFTConfig.__dataclass_fields__[name].default for name in names}
                if any(value is dataclasses.MISSING for value in defaults.values()):
                    raise ValueError("installed TRL preparation defaults need explicit review")
                defaults.update(max_length=cfg.data.max_length,
                                seed=cfg.training.seed if cfg.training.seed is not None else 42)
                prepare_args = types.SimpleNamespace(**defaults)
                model_free_trainer = types.SimpleNamespace(_is_vlm=False)
                if "input_ids" not in dataset.column_names:
                    raise ValueError("Soup did not return the expected pretokenized dataset")
                if repeat == 0:
                    report["sources"].append(_source_receipt(SFTTrainer._prepare_dataset))
                    report["trl_prepare"] = {"model_free": True, "pretokenized": True,
                                             "packing": False, "defaults": defaults}
                with recorder.measure(prefix + ".trl_prepare_pretokenized"):
                    dataset = SFTTrainer._prepare_dataset(
                        model_free_trainer, dataset, tokenizer, prepare_args, False, None, "train")
            # Summary computation is deliberately outside measured phases.
            report["passes"].append(_dataset_summary(dataset))
        report["ordered_outputs_identical"] = len({item["ordered_rows_sha256"]
                                                   for item in report["passes"]}) == 1
    finally:
        recorder.restore()
        report.update(recorder.report())
    import torch

    report["torch_cuda_initialized"] = torch.cuda.is_initialized()
    report["solution_imported"] = any(name == "solution" or name.startswith("solution.")
                                      for name in sys.modules)
    report["status"] = "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="directory containing task/, data/, solution/")
    parser.add_argument("--config", default="task/task.yaml")
    parser.add_argument("--train-file", help="optional existing local task JSONL; relative to --repo")
    parser.add_argument("--revision", help="optional exact tokenizer revision used by the training run")
    parser.add_argument("--expected-rows", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--skip-trl-prepare", action="store_true")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.expected_rows < 1:
        parser.error("--repeats and --expected-rows must be positive")
    # Resolve before _run changes cwd.
    out = Path(args.out).resolve()
    report = {"kind": "cpu_only_tokenization_hashing_wall_diagnostic",
              "status": "started", "model_loaded": False, "training_run": False,
              "fingerprint_or_cache_policy_modified": False,
              "tokenizer_download_allowed": args.allow_tokenizer_download}
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(_DiscardOutput()), contextlib.redirect_stderr(_DiscardOutput()):
            _run(args, report)
    except BaseException as exc:
        # Deliberately exclude str(exc), repr(), source lines and local values:
        # tokenizer/formatter exceptions may embed task text or credentials.
        report["status"] = "failed"
        report["error_type"] = _type_name(exc)
        report["error_locations"] = [{"file": Path(frame.filename).name,
                                      "function": frame.name, "line": frame.lineno}
                                     for frame in traceback.extract_tb(exc.__traceback__)]
    report["total_wall_s"] = time.perf_counter() - started
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(out)}, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
