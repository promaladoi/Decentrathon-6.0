# Paste into ONE new Code cell in the existing Kaggle notebook, then Shift+Enter.
# This checks the v41 archive and screens research; it is not a full training score.
import os, sys, uuid, subprocess
from pathlib import Path
from IPython.display import display, FileLink

root = Path('/kaggle/working')
tag = uuid.uuid4().hex[:8]
repo = root / ('codex_v42_screen_' + tag)
archive = root / ('codex_v42_screen_' + tag + '.zip')
source = 'aea2dbb8076c0c3a1e1d4c6ea361abf2df5b01e7'
subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--single-branch',
                '--branch', 'codex/kernel-v1', '--no-checkout',
                'https://github.com/promaladoi/Decentrathon-6.0.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '-q', source], check=True)
done = subprocess.run([sys.executable, '04_submission/run_gate_up_screen.py',
                       '--output', str(archive)], cwd=repo)
print('SCREEN EXIT:', done.returncode)
os.chdir(root)
if archive.is_file():
    display(FileLink(archive.name))
