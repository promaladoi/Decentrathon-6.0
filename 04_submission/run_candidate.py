"""Serial GPU validation with exact sources, untouched-task preflight and archives.

Run from the repository on Kaggle. A correctness failure stops before training.
Archives retain failed-test logs too; no result/check JSON is invented from logs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v15')
    parser.add_argument('--tests-only', action='store_true')
    args = parser.parse_args()
    if not args.version.replace('_', '').isalnum():
        raise ValueError('version must contain only letters, numbers and underscores')
    repo = Path(__file__).resolve().parents[1]
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run this GPU experiment in the Kaggle notebook')
    out = repo / ('runs_codex_' + args.version)
    if out.exists():
        raise FileExistsError(f'{out} already exists; use a different --version to preserve it')
    out.mkdir()
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    outcome = {'code_commit': code, 'version': args.version, 'status': 'starting'}
    print(f'{args.version.upper()} code {code}', flush=True)

    def run(command, name, required=True):
        with (out / name).open('w', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            rc = child.wait()
        if rc and required:
            raise subprocess.CalledProcessError(rc, command)
        return rc

    rc = 1
    try:
        baseline = json.loads((repo/'04_submission/immutable_baseline.json').read_text())
        for name, expected in baseline['sha256'].items():
            if hashlib.sha256((repo/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Fixed organizer file changed: {name}')
        names = subprocess.check_output(['nvidia-smi', '--query-gpu=name',
                                         '--format=csv,noheader'], text=True)
        if 'T4' not in names:
            raise RuntimeError('Select GPU T4 x2 in Kaggle')
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('Another GPU process is active. Stop it before measurement.')
        outcome['status'] = 'testing'
        # Regression for the v14 failure is first; do not spend training time
        # until the dropout values, gradients and RNG sequence all match.
        run([sys.executable, '-m', 'pytest', 'solution/test_compact_dropout.py', '-q'],
            'dropout_tests.log')
        run([sys.executable, '-m', 'pytest', 'solution', '-q'], 'kernel_tests.log')
        env_command = [sys.executable, '-c',
                       "import os,pytest;os.environ['SOUP_APPLY_SOLUTION']='1';"
                       "raise SystemExit(pytest.main(['tests','-q']))"]
        run(env_command, 'soup_tests.log')
        outcome['status'] = 'tests_passed'
        if not args.tests_only:
            outcome['status'] = 'benchmarking'
            run([sys.executable, 'harness/run_task.py', '--repeats', '2', '--out', str(out)],
                'benchmark_console.log')
            run([sys.executable, 'harness/check_rules.py', '--result', str(out/'result.json')],
                'checks_console.log', required=False)
            run([sys.executable, 'harness/score.py', str(out/'result.json')],
                'score_console.log', required=False)
            checks = json.loads((out/'checks.json').read_text())['checks']
            accepted = len(checks) == 10 and all(c['ok'] for c in checks)
            outcome['status'] = 'accepted' if accepted else 'rejected'
            outcome['checks_passed'] = sum(bool(c['ok']) for c in checks)
            rc = 0 if accepted else 1
        else:
            rc = 0
    except Exception as exc:
        outcome['error'] = str(exc)
        print(f'EXPERIMENT STOPPED: {exc}', flush=True)
    finally:
        (out/'experiment_outcome.json').write_text(json.dumps(outcome, indent=2)+'\n')
        archive = Path('/kaggle/working') / ('codex_' + args.version + '.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in out.iterdir():
                if f.is_file():
                    z.write(f, f.name)
            for f in sorted((repo/'solution').rglob('*.py')):
                z.write(f, f.relative_to(repo).as_posix())
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                f = out/'solution_1/adapter'/name
                if f.exists():
                    z.write(f, 'adapter/'+name)
        print(f'ARCHIVE: {archive}\nSTATUS: {outcome["status"]}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
