# Екінші кезең: GPU профилін алу

Алғашқы starter нәтижесі: 8 PASS, №3 және №4 FAIL. Енді қай операция уақыт алатынын нақты өлшеп, өз кернелімізге нысана таңдаймыз.

## Kaggle-де орындайтын бір ұяшық

Бірінші Run All толық аяқталғаннан кейін, сол жұмыс істеп тұрған сессияда төмендегі `+ Code` басыңыз. Жаңа ұяшыққа осы кодты қойып, тек сол ұяшықтың ▶ батырмасын басыңыз. Бүкіл Run All қайта орындалмайды.

```python
import json
import subprocess
from pathlib import Path

repo = Path("/kaggle/working/repo")
subprocess.run(
    ["python", "harness/run_task.py", "--only", "solution",
     "--profile", "--out", "runs_profile"],
    cwd=repo, check=True,
)
p = json.loads((repo / "runs_profile/solution_profile/profile.json").read_text())
print("\n=== TOP 15 GPU KERNELS ===")
for row in p["kernels"][:15]:
    share = 100 * row["time_us"] / max(p["total_gpu_us"], 1)
    print(f"{share:6.2f}% | {row['count']:6d} calls | {row['name']}")
```

Бұл команда solution-ды бір рет толық оқытып, сапасын бағалайды, содан кейін қысқа profiler run жасайды. Қосымша GPU уақыты жұмсалады. Қатар басқа GPU жұмысын қоспаңыз.

Бастапқы `runs/result.json` және `runs/checks.json` сақталады: жаңа талдау `runs_profile/` ішінде жазылады. Жаңа профиль негізгі benchmark немесе барлық тексерулердің PASS болғанының дәлелі емес; мұнда base қайта орындалмайды.

Нәтижеден `TOP 15 GPU KERNELS` бөлімін толық көшіріп жіберіңіз. Қате болса, traceback жіберіңіз. Токендер бұл кодқа керек емес: репозиторий ағымдағы сессияға жүктеліп қойған.

Операция әлі таңдалған жоқ. Профиль нәтижесінен кейін нақты код, forward/backward дәлдігі және ≥5% уақыт талабы бірге қаралады.
