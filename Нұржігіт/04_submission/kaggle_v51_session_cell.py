import os
from pathlib import Path
import shutil
import subprocess
import uuid
from IPython.display import display, FileLink

control = Path('/kaggle/working/codex_v50_screen_ec2c862b')
revision = '10b8f2a71a85de2863399d351f043c92073ca305'
subprocess.run(['git','-C',str(control),'fetch','-q','origin','codex/kernel-v1'],check=True)
repo = Path('/kaggle/working/codex_v51_'+uuid.uuid4().hex[:8])
subprocess.run(['git','-C',str(control),'worktree','add','--detach',str(repo),revision],check=True)
shutil.copytree('/kaggle/working/codex_v45_startup_42426065/data',repo/'data')
print('V51 ORIGINAL VALIDATION',revision,flush=True)
result = subprocess.run(['python','04_submission/run_candidate.py','--version','v51'],
                        cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),timeout=1800)
print('V51 VALIDATION EXIT',result.returncode,flush=True)
display(FileLink('codex_v51.zip'))
