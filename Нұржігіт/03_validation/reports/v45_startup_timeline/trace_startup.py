"""Coarse monotonic startup timeline around the unchanged original training script.

Diagnostic only: Python profiling adds overhead. Intervals are inclusive and
may overlap; never sum them or present this runtime as a contest score.
Only the main thread is observed, avoiding background compiler attribution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import uuid
import zipfile


def child(repo, config, metrics, timeline):
    start = time.perf_counter()
    active, intervals = {}, []
    modules = {'torch', 'transformers', 'peft', 'trl', 'bitsandbytes', 'triton',
               'soup_cli.cli', 'soup_cli.trainer.sft', 'soup_cli.trainer.rewind_hf'}
    stages = {
        'solution': {'apply'},
        'transformers.modeling_utils': {'from_pretrained', '_load_pretrained_model'},
        'transformers.tokenization_utils_base': {'from_pretrained', '_from_pretrained'},
        'transformers.trainer': {'__init__', 'train', 'save_model', '_save_checkpoint'},
        'trl.trainer.sft_trainer': {'__init__', '_prepare_dataset'},
        'soup_cli.trainer.sft': {'__init__', 'train', '_build_rewind_trainer'},
        'soup_cli.trainer.rewind_hf': {'__init__'},
        'safetensors.torch': {'save_file', 'load_file'},
    }

    def profile(frame, event, arg):
        if event not in ('call', 'return'):
            return
        key = id(frame)
        if event == 'return':
            record = active.pop(key, None)
            if record is not None:
                record['end_s'] = time.perf_counter() - start
                record['duration_s'] = record['end_s'] - record['start_s']
                intervals.append(record)
            return
        module = frame.f_globals.get('__name__', '')
        name = frame.f_code.co_name
        if ((name == '<module>' and module in modules)
                or name in stages.get(module, ())):
            active[key] = dict(module=module, function=frame.f_code.co_qualname,
                               start_s=time.perf_counter() - start)

    sys.path.insert(0, str(repo))
    sys.argv = [str(repo/'harness/_train.py'), '--config', str(config),
                '--metrics', str(metrics), '--solution']
    sys.setprofile(profile)
    try:
        runpy.run_path(str(repo/'harness/_train.py'), run_name='__main__')
    finally:
        sys.setprofile(None)
        report = dict(kind='instrumented main-thread timeline; not a contest score',
                      total_instrumented_seconds=time.perf_counter() - start,
                      intervals_are_inclusive=True, intervals_must_not_be_summed=True,
                      intervals=sorted(intervals, key=lambda row: row['start_s']))
        timeline.write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--metrics', type=Path)
    parser.add_argument('--timeline', type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.child:
        return child(repo, args.config, args.metrics, args.timeline)
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('This diagnostic requires the user Kaggle session')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Both GPUs must be idle')
    if subprocess.check_output(['git','diff','HEAD','--','solution','harness','task','tests'],cwd=repo):
        raise RuntimeError('Use a clean, exact selected source checkout')
    baseline = json.loads((repo/'04_submission/immutable_baseline.json').read_text())['sha256']
    assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest() == digest
               for name, digest in baseline.items())
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    sys.path.insert(0, str(repo/'harness'))
    from common import OFFLINE_ENV, run_logged
    from run_task import make_config, warm_up
    output = repo/('diagnostic_startup_'+uuid.uuid4().hex[:8])
    output.mkdir()
    warm_up(output)
    config = make_config(output, repo/'data/train.jsonl', 1)
    timeline, metrics = output/'timeline.json', output/'metrics.json'
    env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(output/'experiments.db'),
               TRITON_CACHE_DIR=str(output/'triton_cache'),
               TORCHINDUCTOR_CACHE_DIR=str(output/'inductor_cache'))
    rc, wall, peak, gpu_start = run_logged(
        [sys.executable, str(Path(__file__).resolve()), '--child', '--repo', str(repo),
         '--config', str(config), '--metrics', str(metrics), '--timeline', str(timeline)],
        output/'train.log', 900, env)
    report = json.loads(timeline.read_text())
    report.update(source_commit=source_commit, exit_code=rc,
                  diagnostic_outer_wall_s=wall, diagnostic_peak_mib=peak)
    timeline.write_text(json.dumps(report,indent=2)+'\n')
    print('STARTUP TIMELINE: instrumented inclusive intervals, not a score',flush=True)
    for row in report['intervals']:
        if row['duration_s'] >= .05:
            print(json.dumps(row),flush=True)
    archive = Path('/kaggle/working/codex_startup_'+output.name.rsplit('_',1)[1]+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in ('timeline.json','metrics.json','train.log','warmup.log','task.yaml'):
            if (output/name).is_file():
                z.write(output/name,name)
        z.write(Path(__file__), 'trace_startup.py')
    print('STARTUP_DIAGNOSTIC',json.dumps(dict(archive=str(archive),source_commit=source_commit,
          sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),exit_code=rc)),flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
