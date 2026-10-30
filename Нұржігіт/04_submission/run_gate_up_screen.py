"""Preserve/verify v41 originals, then run the inactive v42 T4 screen once."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid
import zipfile

V41 = 'db4908e8b486874fda8c6eacc230166116ccad01'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    working = Path('/kaggle/working')
    if not working.is_dir() or os.name != 'posix':
        raise RuntimeError('Run in the existing Kaggle notebook containing codex_v41.zip')
    output = args.output.resolve()
    if not output.is_relative_to(working.resolve()) or output.suffix != '.zip' or output.exists():
        raise ValueError('Use a new ZIP path inside /kaggle/working')
    repo = Path(__file__).resolve().parents[1]
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    out = working/('codex_gate_up_screen_'+uuid.uuid4().hex[:8])
    out.mkdir()
    source = working/'codex_v41.zip'
    outcome = dict(kind='isolated inactive research, not a complete-task score',
                   screen_code_commit=code, status='verifying_v41',
                   production_solution_changed=False, target_30_seconds_claimed=False)
    rc = 1
    try:
        collector = load('artifact_collector', repo/'03_validation/collect_artifact.py')
        proof = collector.collect(source, V41, 'v41', repo)
        if not proof['accepted']:
            raise RuntimeError('v41 archive does not contain all ten accepted original checks')
        evidence = repo/'03_validation/reports/codex_v41'
        policy = load('user_policy', repo/'03_validation/user_constraints.py')
        constraints = policy.evaluate(json.loads((evidence/'result.json').read_text()),
                                      json.loads((evidence/'checks.json').read_text()))
        (out/'v41_collection_proof.json').write_text(json.dumps(proof, indent=2))
        (out/'v41_user_constraints.json').write_text(json.dumps(constraints, indent=2))
        print('V41 ORIGINAL ARCHIVE VERIFIED', json.dumps(proof), flush=True)
        print('V41 USER CONSTRAINTS', json.dumps(constraints), flush=True)
        outcome['status'] = 'research_testing_and_timing'
        print('V42 SCREEN: numerical checks first, then isolated kernel timing. No model training.', flush=True)
        child = subprocess.Popen([sys.executable, '03_validation/benchmark_gate_up.py',
                                  '--output', str(out/'timings.json')], cwd=repo,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, start_new_session=True)
        try:
            logs, _ = child.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                logs, _ = child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                logs, _ = child.communicate()
            (out/'screen.log').write_text(logs)
            raise RuntimeError('Finite 15-minute diagnostic timeout; own worker group stopped')
        (out/'screen.log').write_text(logs)
        print(logs, flush=True)
        outcome['research_exit_code'] = child.returncode
        outcome['status'] = 'screen_complete' if child.returncode == 0 else 'research_rejected'
        rc = child.returncode
        # Timing alone is never admission to the competition or the user limits.
        outcome['full_task_admission'] = 'not run; production unchanged'
    except Exception as exc:
        outcome.update(failed_stage=outcome['status'], status='failed', error=str(exc))
        print('SCREEN STOPPED:', str(exc), flush=True)
    finally:
        (out/'screen_outcome.json').write_text(json.dumps(outcome, indent=2))
        with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
            for f in sorted(out.iterdir()):
                if f.is_file():
                    archive.write(f, f.name)
            if source.is_file():
                archive.write(source, 'codex_v41.zip')
            for name in ('02_kernel_research/gate_up_forward.py',
                         '02_kernel_research/test_gate_up_forward.py',
                         '03_validation/benchmark_gate_up.py'):
                archive.write(repo/name, name)
        print('SCREEN OUTCOME', json.dumps(outcome), flush=True)
        print('SCREEN ARCHIVE', output, flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
