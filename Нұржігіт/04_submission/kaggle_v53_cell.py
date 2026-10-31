# Paste into ONE new Kaggle code cell and press Shift+Enter once.
# GPU T4 x2, Internet On. Uses one T4. Original paired harness + all ten checks.
# Experimental v53; the verified public selection remains v45 until admitted.
import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys
import uuid
from IPython.display import display, FileLink

working = Path('/kaggle/working')
assert working.is_dir(), 'Run in Kaggle'
names = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
assert 'T4' in names, 'Select Settings > Accelerator > GPU T4 x2'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
    '--format=csv,noheader'],text=True).strip(), 'Wait until both GPUs are idle'
assert not (working/'codex_v53.zip').exists(), 'Preserve the existing v53 archive; do not rerun this cell'
pins = {'soup-cli':'0.77.0','transformers':'5.16.1','peft':'0.20.0',
        'trl':'0.29.1','bitsandbytes':'0.50.2','accelerate':'1.14.0'}
def installed(name,version):
    try:
        return metadata.version(name) == version
    except metadata.PackageNotFoundError:
        return False
if not all(installed(name,version) for name,version in pins.items()):
    subprocess.run([sys.executable,'-m','pip','install','-q','soup-cli[train]==0.77.0',
        *[name+'=='+version for name,version in pins.items() if name != 'soup-cli'],'pytest'],check=True)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
except Exception:
    pass
revision = '373c904735bc5061c7ea06dd3cad553ee695be4c'
repo = working/('codex_v53_'+uuid.uuid4().hex[:8])
subprocess.run(['git','clone','-q','--single-branch','--branch','codex/kernel-v1',
    'https://github.com/promaladoi/Decentrathon-6.0.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach',revision],check=True)
subprocess.run([sys.executable,'task/prepare_data.py'],cwd=repo,check=True,timeout=600)
result = subprocess.run([sys.executable,'04_submission/run_candidate.py','--version','v53'],
    cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),timeout=1800)
print('V53 VALIDATION EXIT',result.returncode)
display(FileLink('codex_v53.zip'))
