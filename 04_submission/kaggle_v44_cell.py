"""Paste/run once in the current Kaggle session: bounded full v44 validation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

from IPython.display import display, FileLink

SOURCE = '2d171ddc1f6d43e593f5dd80e1a86731f27ad6f7'
CONTROL = '2d171ddc1f6d43e593f5dd80e1a86731f27ad6f7'
QUEUE = 'eff793254e6ba005955bee26f6dd4d9c5ec2187c'
working = Path('/kaggle/working')
if os.name != 'posix' or not working.is_dir():
    raise RuntimeError('Run in Kaggle with GPU T4 x2 and Internet on')
control = working/('codex_v44_control_'+uuid.uuid4().hex[:8])
subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--single-branch',
                '--branch', 'codex/kernel-v1', '--no-checkout',
                'https://github.com/promaladoi/Decentrathon-6.0.git', str(control)],
               check=True, timeout=180)
subprocess.run(['git', '-C', str(control), 'checkout', '-q', CONTROL], check=True)
# Coordinator checks GPU idleness, creates an exact separate worker, prepares
# the original fixed split if needed, and bounds the complete worker to 20 min.
# It never changes the checkout of a running measured process.
done = subprocess.run([sys.executable, '04_submission/coordinate_experiments.py',
                       '--max-jobs', '1', '--idle-minutes', '0', '--max-minutes', '20',
                       '--queue-commit', QUEUE],
                      cwd=control)
print('V44 COORDINATOR EXIT:', done.returncode, flush=True)
archive = working/'codex_v44.zip'
if archive.is_file():
    spec = importlib.util.spec_from_file_location('v44_artifact_collector',
                                                 control/'03_validation/collect_artifact.py')
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    proof = collector.collect(archive, SOURCE, 'v44', control)
    print('V44 ORIGINAL ARCHIVE VERIFIED', json.dumps(proof), flush=True)
    with zipfile.ZipFile(archive) as z:
        if 'user_constraints.json' in z.namelist():
            print('V44 USER CONSTRAINTS', z.read('user_constraints.json').decode(), flush=True)
        else:
            print('V44 STOPPED BEFORE FULL RESULT', z.read('experiment_outcome.json').decode(), flush=True)
    os.chdir(working)
    display(FileLink(archive.name))
else:
    print('No v44 archive: inspect the coordinator error above. No completed v44 result is claimed.', flush=True)
