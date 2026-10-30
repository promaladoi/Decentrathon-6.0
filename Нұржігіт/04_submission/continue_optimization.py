"""Finish queued exact-source experiments, then profile an admitted fast candidate.

Finite work for this optimization task, no concurrent GPU jobs and no automatic
submission. Profiling is separate and never scored as a paired contest result.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from coordinate_experiments import REPOSITORY, archived_outcome, parse_queue


def main():
    control = Path(__file__).resolve().parents[1]
    working = Path('/kaggle/working')
    if not working.is_dir():
        raise RuntimeError('Run on Kaggle after the active experiment cell')
    subprocess.run([sys.executable, '04_submission/coordinate_experiments.py',
                    '--max-jobs', '3', '--idle-minutes', '0', '--max-minutes', '35'],
                   cwd=control, check=True)
    # Read the queue fetched between workers, not an assumed moving branch tip.
    raw = subprocess.check_output(['git', 'show', 'FETCH_HEAD:04_submission/experiment_queue.json'], cwd=control)
    jobs = parse_queue(raw)
    jobs += [{'version': 'v16', 'commit': '4a5cfd20fd79e597e15919f3f19f959a6b4d689a',
              'nf4_analysis': False}]
    admitted = []
    for job in jobs:
        archive = working/('codex_'+job['version']+'.zip')
        if archive.exists():
            outcome = archived_outcome(archive, job['commit'])
            if outcome['accepted']:
                admitted.append(dict(job, summary=outcome['summary']))
    if not admitted:
        raise RuntimeError('No admitted candidate available to profile')
    best = max(admitted, key=lambda j: j['summary']['ratios']['speedup'])
    print('SELECTED FOR DIAGNOSIS (paired ratio, all ten checks):', json.dumps(best), flush=True)
    repo = working/('codex_analysis_'+uuid.uuid4().hex[:8])
    subprocess.run(['git', 'clone', '-q', REPOSITORY, str(repo)], check=True, timeout=120)
    subprocess.run(['git', '-C', str(repo), 'checkout', '-q', best['commit']], check=True)
    source = working/'nurzhigit_v14/data'
    if not all((source/n).is_file() for n in ('train.jsonl', 'eval_public.jsonl')):
        raise FileNotFoundError('Keep the same task data for the diagnostic')
    shutil.copytree(source, repo/'data')
    subprocess.run([sys.executable, str(control/'03_validation/profile_candidate.py'),
                    '--repo', str(repo), '--cpu'], check=True)
    status = {'status': 'diagnosis_complete', 'selected': best,
              'diagnostic_file': 'codex_diagnostic_'+best['commit'][:7]+'.zip',
              'diagnostic_runtime_is_contest_score': False}
    (working/'codex_followup_status.json').write_text(json.dumps(status, indent=2)+'\n')
    print('FOLLOWUP COMPLETE:', json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
