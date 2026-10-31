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
    parser.add_argument('--version', default='v16')
    parser.add_argument('--tests-only', action='store_true')
    parser.add_argument('--nf4-analysis', action='store_true',
                        help='run an independent decoder microbenchmark before the contest harness')
    args = parser.parse_args()
    if not args.version.replace('_', '').isalnum():
        raise ValueError('version must contain only letters, numbers and underscores')
    repo = Path(__file__).resolve().parents[1]
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run this GPU experiment in the Kaggle notebook')
    archive = Path('/kaggle/working') / ('codex_' + args.version + '.zip')
    if archive.exists():
        raise FileExistsError(f'Preserve the existing {archive}; use a new version')
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
        if (repo/'solution/test_lazy_train_cli.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_lazy_train_cli.py', '-q'],
                'lazy_cli_tests.log')
        if (repo/'solution/test_resident_nf4.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_resident_nf4.py', '-q'],
                'resident_nf4_tests.log')
        if (repo/'solution/test_lora_remat.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_lora_remat.py',
                 'solution/test_lora_scaling.py', '-q'],
                'lora_rematerialization_tests.log')
        if (repo/'solution/test_nf4_launch_geometry.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_nf4_launch_geometry.py', '-q'],
                'nf4_launch_geometry_tests.log')
        if (repo/'solution/test_narrow_nf4.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_narrow_nf4.py', '-q'],
                'narrow_nf4_tests.log')
        if (repo/'solution/test_mlp_recompute.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_mlp_recompute.py', '-q'],
                'mlp_recompute_tests.log')
        if (repo/'solution/test_small_dropout_dispatch.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_small_dropout_dispatch.py', '-q'],
                'small_dropout_dispatch_tests.log')
        if (repo/'solution/test_target_metadata.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_target_metadata.py',
                 'solution/test_batch_metadata.py', '-q'], 'target_metadata_tests.log')
        if (repo/'solution/deferred_metadata.py').is_file():
            run([sys.executable, '03_validation/verify_startup_metadata.py'],
                'startup_metadata_parity.log')
        # The earlier path and the corrected low-memory team path use different
        # dropout modules. Both retain their complete strict tests.
        dropout_test = ('solution/test_compact_dropout.py'
                        if (repo/'solution/test_compact_dropout.py').is_file()
                        else 'solution/test_dropout.py')
        run([sys.executable, '-m', 'pytest', dropout_test, '-q'],
            'dropout_tests.log')
        if (repo/'solution/test_storage_integration.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_storage_integration.py', '-q'],
                'storage_integration_tests.log')
            run([sys.executable, '-c',
                 "import os,pytest;os.environ['MEREY_NATIVE_METADATA_SNAPSHOT_PARITY']='1';"
                 "raise SystemExit(pytest.main(['solution/test_package_metadata_snapshot.py',"
                 "'-q','-k','real_native_setup_prefix']))"], 'native_metadata_parity.log')
        if (repo/'solution/test_head_lifetime.py').is_file():
            run([sys.executable, '-m', 'pytest', 'solution/test_head_lifetime.py',
                 'solution/test_lmhead_scaled.py', '-q'], 'head_lifetime_and_gradient_tests.log')
        if args.nf4_analysis:
            run([sys.executable, '-m', 'pytest', 'solution/test_paired_nf4.py', '-q'],
                'paired_nf4_tests.log')
        run([sys.executable, '-m', 'pytest', 'solution', '-q'], 'kernel_tests.log')
        env_command = [sys.executable, '-c',
                       "import os,pytest;os.environ['SOUP_APPLY_SOLUTION']='1';"
                       "raise SystemExit(pytest.main(['tests','-q']))"]
        run(env_command, 'soup_tests.log')
        outcome['status'] = 'tests_passed'
        if not args.tests_only:
            if args.nf4_analysis:
                outcome['status'] = 'decoder_analysis'
                run([sys.executable, '03_validation/benchmark_nf4_pair.py',
                     '--output', str(out/'nf4_microbenchmark.json')], 'nf4_microbenchmark.log')
            outcome['status'] = 'benchmarking'
            run([sys.executable, 'harness/run_task.py', '--repeats', '2', '--out', str(out)],
                'benchmark_console.log')
            run([sys.executable, 'harness/check_rules.py', '--result', str(out/'result.json')],
                'checks_console.log', required=False)
            run([sys.executable, 'harness/score.py', str(out/'result.json')],
                'score_console.log', required=False)
            checks = json.loads((out/'checks.json').read_text())['checks']
            accepted = len(checks) == 10 and all(c['ok'] for c in checks)
            from importlib.util import module_from_spec, spec_from_file_location
            spec = spec_from_file_location('user_limits', repo/'03_validation/user_constraints.py')
            policy = module_from_spec(spec)
            spec.loader.exec_module(policy)
            limits = policy.evaluate(json.loads((out/'result.json').read_text()),
                                     json.loads((out/'checks.json').read_text()))
            (out/'user_constraints.json').write_text(json.dumps(limits, indent=2)+'\n')
            print('USER CONSTRAINTS', json.dumps(limits), flush=True)
            outcome['status'] = 'accepted' if accepted else 'rejected'
            outcome['checks_passed'] = sum(bool(c['ok']) for c in checks)
            outcome['eligible_within_memory_and_quality'] = limits['eligible_within_memory_and_quality']
            outcome['target_total_seconds'] = limits['target_total_seconds']
            outcome['target_met'] = limits['target_met']
            rc = 0 if accepted and limits['eligible_within_memory_and_quality'] else 1
        else:
            rc = 0
    except Exception as exc:
        outcome['failed_stage'] = outcome['status']
        outcome['status'] = 'failed'
        outcome['error'] = str(exc)
        print(f'EXPERIMENT STOPPED: {exc}', flush=True)
    finally:
        (out/'experiment_outcome.json').write_text(json.dumps(outcome, indent=2)+'\n')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for f in out.iterdir():
                if f.is_file():
                    z.write(f, f.name)
            for f in sorted((repo/'solution').rglob('*.py')):
                z.write(f, f.relative_to(repo).as_posix())
            # Preserve the real startup/compile logs as well as the summary.
            # This happens after all measured processes and checks have exited.
            for run_dir in sorted(out.iterdir()):
                if run_dir.is_dir() and run_dir.name.startswith(('base_', 'solution_')):
                    for name in ('train.log', 'metrics.json'):
                        f = run_dir/name
                        if f.is_file():
                            z.write(f, 'runs/'+run_dir.name+'/'+name)
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                f = out/'solution_1/adapter'/name
                if f.exists():
                    z.write(f, 'adapter/'+name)
        print(f'ARCHIVE: {archive}\nSTATUS: {outcome["status"]}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
