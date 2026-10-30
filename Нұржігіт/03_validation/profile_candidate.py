"""Separate diagnostic runs; never present profiler runtimes as contest scores.

The organizer harness remains unchanged. A normal solution-only run followed by
its built-in short GPU profile identifies the current kernels. Optional cProfile
and import-time logging identify CPU startup/dispatch work in a separate process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import uuid
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    repo = a.repo.resolve()
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run the diagnostic in Kaggle')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                 '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Another GPU process is active')
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if subprocess.check_output(['git', 'diff', 'HEAD', '--', 'solution', 'harness', 'task', 'tests'], cwd=repo):
        raise RuntimeError('Use a clean exact checkout')
    baseline = json.loads((repo/'04_submission/immutable_baseline.json').read_text())['sha256']
    if any(hashlib.sha256((repo/n).read_bytes()).hexdigest() != h for n, h in baseline.items()):
        raise RuntimeError('Organizer files changed')
    out = repo/('diagnostic_'+uuid.uuid4().hex[:8])
    out.mkdir()

    def run(command, log, env=None):
        with (out/log).open('w', encoding='utf-8') as f:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in child.stdout:
                f.write(line)
                f.flush()
                if not line.startswith('import time:'):
                    print(line, end='', flush=True)
            rc = child.wait()
        if rc:
            raise subprocess.CalledProcessError(rc, command)

    run([sys.executable, 'harness/run_task.py', '--only', 'solution', '--profile',
         '--out', str(out/'gpu')], 'gpu_diagnostic.log')
    gpu = json.loads((out/'gpu/solution_profile/profile.json').read_text())
    print('DIAGNOSTIC GPU TOP 15 (not a contest score):',
          json.dumps(gpu['kernels'][:15]), flush=True)
    if a.cpu:
        import yaml
        config = yaml.safe_load((out/'gpu/solution_1/task.yaml').read_text())
        config['output'] = str(out/'cpu_adapter')
        cfg_path = out/'cpu_task.yaml'
        cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        # The same environment policy as the unchanged harness; fresh compiler
        # caches also for this explicitly non-scored diagnostic.
        sys.path.insert(0, str(repo/'harness'))
        from common import OFFLINE_ENV
        env = dict(OFFLINE_ENV, TRITON_CACHE_DIR=str(out/'cpu_triton_cache'),
                   TORCHINDUCTOR_CACHE_DIR=str(out/'cpu_inductor_cache'),
                   SOUP_DB_PATH=str(out/'cpu_experiments.db'))
        run([sys.executable, '-X', 'importtime', '-m', 'cProfile', '-o', str(out/'cpu.prof'),
             str(repo/'harness/_train.py'), '--config', str(cfg_path), '--solution',
             '--metrics', str(out/'cpu_metrics.json')], 'cpu_diagnostic.log', env)
        stats = pstats.Stats(str(out/'cpu.prof'))
        rows = [{'file': key[0], 'line': key[1], 'function': key[2], 'calls': data[1],
                 'self_s': data[2], 'cumulative_s': data[3]} for key, data in stats.stats.items()]
        cpu = {'total_profiled_s': stats.total_tt,
               'top_self': sorted(rows, key=lambda r: r['self_s'], reverse=True)[:100],
               'top_cumulative': sorted(rows, key=lambda r: r['cumulative_s'], reverse=True)[:100]}
        (out/'cpu_summary.json').write_text(json.dumps(cpu, indent=2)+'\n', encoding='utf-8')
        print('DIAGNOSTIC CPU TOP 15:', json.dumps(cpu['top_self'][:15]), flush=True)
    manifest = {'code_commit': code, 'kind': 'diagnostic only, not an admitted paired result',
                'gpu_active_steps': gpu['active_steps'], 'cpu_profile': a.cpu}
    (out/'diagnostic_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = Path('/kaggle/working/codex_diagnostic_'+code[:7]+'.zip')
    if archive.exists():
        raise FileExistsError('Preserve previous diagnostic archive')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file() and f.suffix in ('.json', '.log', '.prof'):
                z.write(f, f.relative_to(out).as_posix())
        for f in (repo/'solution').rglob('*.py'):
            z.write(f, f.relative_to(repo).as_posix())
    print('DIAGNOSTIC ARCHIVE:', archive, flush=True)


if __name__ == '__main__':
    main()
