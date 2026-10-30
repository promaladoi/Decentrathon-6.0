"""Run once after previous GPU work ends: exact v46 tests and paired admission."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import uuid

SOURCE = '30075981acb89e26e8b0a2fa4659d8b4c100daed'
working = Path('/kaggle/working')
if not working.is_dir():
    raise RuntimeError('Run in Kaggle with T4 and Internet enabled')
if (working/'codex_v46.zip').exists():
    raise FileExistsError('Preserve the existing v46 archive; do not overwrite it')
if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
    raise RuntimeError('Wait until the existing GPU job ends')
repo = working/('codex_worker_v46_' + uuid.uuid4().hex[:8])
subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--single-branch',
                '--branch', 'codex/kernel-v1', '--no-checkout',
                'https://github.com/promaladoi/Decentrathon-6.0.git', str(repo)], check=True, timeout=180)
subprocess.run(['git', '-C', str(repo), 'checkout', '-q', SOURCE], check=True)
print('V46 SOURCE', SOURCE, flush=True)
subprocess.run([sys.executable, 'task/prepare_data.py'], cwd=repo, check=True, timeout=300)
subprocess.run([sys.executable, '-m', 'pytest', 'solution/test_dropout_half.py', '-q'],
               cwd=repo, check=True, timeout=180)
result = subprocess.run([sys.executable, '04_submission/run_candidate.py', '--version', 'v46'],
                        cwd=repo, timeout=1200)
print('V46 EXIT', result.returncode, flush=True)
archive = working/'codex_v46.zip'
if archive.exists():
    print('V46 ARCHIVE', archive.name, flush=True)
    from IPython.display import FileLink, display
    display(FileLink(str(archive)))
