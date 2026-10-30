# Run this ONE cell after the failed v15 tests. Existing data and reports are preserved.
import subprocess
from pathlib import Path
from IPython.display import display, FileLink
p = Path('/kaggle/working/nurzhigit_v14')
assert p.is_dir(), 'Open the notebook where v14 just failed.'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU is busy'
subprocess.run(['git','-C',str(p),'fetch','-q','origin','codex/kernel-v1'],check=True,timeout=120)
subprocess.run(['git','-C',str(p),'checkout','-q','4a5cfd20fd79e597e15919f3f19f959a6b4d689a'],check=True)
completed = subprocess.run(['python','04_submission/run_candidate.py','--version','v16'],cwd=p)
display(FileLink('codex_v16.zip'))
assert completed.returncode == 0, 'Test or rule check failed; inspect output/archive.'
