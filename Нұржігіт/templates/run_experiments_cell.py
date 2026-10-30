# One bounded coordinator launch: later queued revisions run without more edits.
import json, os, signal, subprocess, uuid
from pathlib import Path
from IPython.display import display, FileLink
working = Path('/kaggle/working')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'Previous GPU work is still active'
# A fresh Kaggle session may keep old notebook outputs but lose installed packages.
# Restore the fixed dependency versions before tests, outside measured training.
subprocess.run(['python','-m','pip','install','-q','soup-cli[train]==0.77.0','transformers==5.16.1','peft==0.20.0','trl==0.29.1','bitsandbytes==0.50.2','accelerate==1.14.0','pytest'],check=True)
control = working/('codex_control_'+uuid.uuid4().hex[:8])
subprocess.run(['git','clone','-q','https://github.com/promaladoi/Decentrathon-6.0.git',str(control)],check=True,timeout=120)
subprocess.run(['git','-C',str(control),'checkout','-q','73698fd050e3ebccc6fff46e19d80a57e69a3351'],check=True)
child = subprocess.Popen(['python','04_submission/coordinate_experiments.py','--max-jobs','3','--idle-minutes','10','--max-minutes','50'],cwd=control,start_new_session=True)
try:
    completed = child.wait()
except KeyboardInterrupt:
    if child.poll() is None and os.getpgid(child.pid) == child.pid:
        os.killpg(child.pid,signal.SIGTERM)
        child.wait(timeout=20)
    raise
status = working/'codex_queue_status.json'
if status.exists():
    display(FileLink(status.name))
    for job in json.loads(status.read_text()).get('history',[]):
        name = 'codex_'+job['version']+'.zip'
        if (working/name).exists(): display(FileLink(name))
assert completed == 0, 'Coordinator stopped; inspect its output and status.'
