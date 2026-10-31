"""Same-session performance control only; no new ten-check admission claim."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile
from IPython.display import display, FileLink

repo=Path('/kaggle/working/codex_v45_startup_42426065')
revision='7a3db8fd5e24bbcb58affb475007a5cf93c56f07'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==revision
assert not subprocess.check_output(['git','diff','HEAD','--','solution','harness','task','tests'],cwd=repo)
baseline=json.loads((repo/'04_submission/immutable_baseline.json').read_text())['sha256']
assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest for name,digest in baseline.items())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                   '--format=csv,noheader'],text=True).strip()
out=repo/('runs_control_v45_'+uuid.uuid4().hex[:8])
print('V45 SAME-SESSION CONTROL: original solution-only harness, two repeats; not full admission',flush=True)
subprocess.run([sys.executable,'harness/run_task.py','--only','solution','--repeats','2',
                '--out',str(out)],cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),check=True,timeout=900)
result=json.loads((out/'result.json').read_text())
manifest=dict(kind='same-session solution-only performance control; not ten-check admission',
              source_commit=revision,summary=result['summary'],immutable_files_verified=len(baseline),
              full_task_admission=False)
(out/'control_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=Path('/kaggle/working/codex_v45_control_'+out.name.rsplit('_',1)[1]+'.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for name in ('result.json','control_manifest.json'):
        z.write(out/name,name)
    for directory in sorted(out.glob('solution_*')):
        for name in ('metrics.json','train.log'):
            if (directory/name).is_file(): z.write(directory/name,'runs/'+directory.name+'/'+name)
    for f in sorted((repo/'solution').rglob('*.py')):
        z.write(f,f.relative_to(repo).as_posix())
print('CONTROL COMPLETE',json.dumps(dict(archive=archive.name,
      sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),**manifest)),flush=True)
display(FileLink(archive.name))

# Sparse startup markers provide a second diagnostic, never a scored runtime.
control=Path('/kaggle/working/codex_v50_screen_ec2c862b')
subprocess.run(['git','-C',str(control),'fetch','-q','origin','codex/kernel-v1'],check=True)
subprocess.run(['git','-C',str(control),'checkout','--detach',
                '381713e0d6f46e144d009d80d472bc47b5f19fe0'],check=True)
subprocess.run([sys.executable,str(control/'03_validation/trace_startup.py'),
                '--repo',str(repo),'--mode','statements'],cwd=repo,
               env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),check=True,timeout=1200)
for f in sorted(Path('/kaggle/working').glob('codex_startup_*.zip')):
    display(FileLink(f.name))
