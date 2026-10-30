"""Bounded, serial Kaggle experiments from this team's exact Git commits.

Only fetches the manifest between jobs. Never changes an active worker checkout.
No GitHub credential or public posting is used; artifacts stay in Kaggle working.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile

REPOSITORY = 'https://github.com/promaladoi/Decentrathon-6.0.git'
BRANCH = 'codex/kernel-v1'


def parse_queue(raw):
    value = json.loads(raw)
    if set(value) != {'schema', 'jobs'} or value['schema'] != 1:
        raise ValueError('unsupported queue schema')
    if not isinstance(value['jobs'], list) or len(value['jobs']) > 8:
        raise ValueError('queue must contain at most eight jobs')
    versions = set()
    for job in value['jobs']:
        if set(job) != {'version', 'commit', 'nf4_analysis'}:
            raise ValueError('job may contain only version, commit and nf4_analysis')
        if not re.fullmatch(r'v[0-9]+[a-z0-9_]*', job['version']):
            raise ValueError('invalid version')
        if not re.fullmatch(r'[0-9a-f]{40}', job['commit']):
            raise ValueError('commit must be a complete SHA')
        if type(job['nf4_analysis']) is not bool:
            raise ValueError('nf4_analysis must be boolean')
        if job['version'] in versions:
            raise ValueError('versions must be unique')
        versions.add(job['version'])
    return value['jobs']


def archived_outcome(path, expected_commit):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('invalid archive CRC')
        outcome = json.loads(z.read('experiment_outcome.json'))
        if outcome['code_commit'] != expected_commit:
            raise ValueError('archive has another source revision; do not overwrite it')
        summary = None
        accepted = False
        if 'result.json' in z.namelist() and 'checks.json' in z.namelist():
            raw = z.read('result.json')
            checks = json.loads(z.read('checks.json'))
            if hashlib.sha256(raw).hexdigest() != checks['result_sha256']:
                raise ValueError('report hash mismatch')
            accepted = (outcome['status'] == 'accepted' and len(checks['checks']) == 10
                        and all(c['ok'] for c in checks['checks']))
            summary = json.loads(raw)['summary']
        return {'status': outcome['status'], 'accepted': accepted, 'summary': summary}


def main():
    import fcntl  # Kaggle Linux; module remains importable for CPU schema tests.
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-jobs', type=int, default=3)
    parser.add_argument('--idle-minutes', type=float, default=10)
    parser.add_argument('--max-minutes', type=float, default=50)
    parser.add_argument('--data-source', default='/kaggle/working/nurzhigit_v14/data')
    args = parser.parse_args()
    if not 1 <= args.max_jobs <= 3 or not 0 <= args.idle_minutes <= 15 or not 1 <= args.max_minutes <= 60:
        raise ValueError('coordinator limits exceeded')
    working = Path('/kaggle/working')
    if not working.is_dir():
        raise RuntimeError('Run the coordinator on Kaggle')
    control = Path(__file__).resolve().parents[1]
    baseline = json.loads((control/'04_submission/immutable_baseline.json').read_text())['sha256']
    lock = (working/'codex_queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = Path(args.data_source)
    if not all((data/n).is_file() for n in ('train.jsonl', 'eval_public.jsonl')):
        data = control/'data'
        subprocess.run([sys.executable, 'task/prepare_data.py'], cwd=control, check=True)
    state = {'state': 'starting', 'history': [], 'max_jobs': args.max_jobs}
    status_path = working/'codex_queue_status.json'
    started = last_activity = time.monotonic()
    seen = set()
    attempted = 0
    child = None
    def terminate(signum, frame):
        raise KeyboardInterrupt('coordinator stopped')
    signal.signal(signal.SIGTERM, terminate)

    def publish():
        state['updated_unix'] = time.time()
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(status_path)
        print('COORDINATOR', json.dumps(state), flush=True)

    try:
        while attempted < args.max_jobs and time.monotonic()-started < args.max_minutes*60:
            # A worker has finished before this fetch; no network/polling work
            # runs concurrently with a measured training process.
            subprocess.run(['git', 'fetch', '-q', 'origin', BRANCH], cwd=control,
                           check=True, timeout=120)
            raw = subprocess.check_output(['git', 'show', 'FETCH_HEAD:04_submission/experiment_queue.json'], cwd=control)
            jobs = parse_queue(raw)
            pending = [j for j in jobs if j['version'] not in seen]
            if not pending:
                state['state'] = 'idle'
                publish()
                if time.monotonic()-last_activity >= args.idle_minutes*60:
                    break
                time.sleep(20)
                continue
            job = pending[0]
            subprocess.run(['git', 'merge-base', '--is-ancestor', job['commit'], 'FETCH_HEAD'],
                           cwd=control, check=True)
            version = job['version']
            archive = working/('codex_'+version+'.zip')
            if archive.exists():
                result = archived_outcome(archive, job['commit'])
                seen.add(version)
                state['history'].append(dict(job, **result, existing_archive=True))
                publish()
                continue
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                              '--format=csv,noheader'], text=True).strip()
            if active:
                state['state'] = 'gpu_busy'
                publish()
                return 1
            state['state'], state['current'] = 'preparing', job
            publish()
            worker = working/('codex_worker_'+version+'_'+uuid.uuid4().hex[:8])
            subprocess.run(['git', 'clone', '-q', REPOSITORY, str(worker)], check=True, timeout=120)
            subprocess.run(['git', '-C', str(worker), 'checkout', '-q', job['commit']], check=True)
            for name, digest in baseline.items():
                if hashlib.sha256((worker/name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError('fixed organizer file changed: '+name)
            shutil.copytree(data, worker/'data', dirs_exist_ok=True)
            command = [sys.executable, '04_submission/run_candidate.py', '--version', version]
            if job['nf4_analysis']:
                command.append('--nf4-analysis')
            state['state'] = 'running'
            publish()
            remaining = args.max_minutes*60-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('coordinator deadline reached before worker start')
            child = subprocess.Popen(command, cwd=worker, start_new_session=True)
            try:
                child.wait(timeout=min(2400, remaining))
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if child.poll() is None and os.getpgid(child.pid) == child.pid:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise
            child = None
            result = archived_outcome(archive, job['commit'])
            attempted += 1
            seen.add(version)
            state['history'].append(dict(job, **result))
            state.pop('current', None)
            state['state'] = 'job_finished'
            publish()
            last_activity = time.monotonic()
        state['state'] = 'complete'
        publish()
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        state['state'], state['error'] = 'stopped', str(exc)
        publish()
        return 1
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
