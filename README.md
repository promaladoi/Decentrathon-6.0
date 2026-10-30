# IVNEX: Soup Kernel Challenge

Latest fully collected and verified candidate: [v16 result](CODEX_V16_RESULT.md),
**1.4082x speedup (29.0% less total runtime)**, **61.0% less GPU memory**, **10/10 PASS**.
Two paired runs: **85.07 → 60.41 s**, **14345 → 5601 MiB**.
Public eval loss **2.04966 → 2.04968 (+0.001%, passes 0.5% limit)**.
Code: `4a5cfd2`. **115 kernel tests + 79 Soup tests**, own-kernel share **12.9%**.
Exact source/report hashes and raw archive verified locally.
[Reproduce v16](templates/codex_v16.ipynb) · [Pinned candidate](templates/codex_candidate.ipynb).

The branch tip contains [v17 paired-byte NF4 experiment](CODEX_V17_EXPERIMENT.md).
Its 104 CPU layout cases pass; T4 parity, performance, quality and ten checks are pending.
[Current status](CURRENT_STATUS.md) · [All variants](03_validation/VERSION_COMPARISON.md).
The final team submission is not selected; no 70–90% time reduction or victory is claimed.

## Earlier v3 candidate (archive)

Code commit: `8f82109fdec5f239c0cd4df21ed27fded849cd18`, branch `codex/kernel-v1`.
Тапсырма баптауларын сақтап, vocabulary loss/entropy, frozen RMSNorm және NF4
декванттауына алты өз Triton кернеліміз қосылды. Дайын SwiGLU мысалы сақталған.
Kaggle T4-де 33 кернел тесті, 79 Soup тесті және барлық 10 ереже тексеруі өтті.
Екі solution іске қосуындағы ең аз өз-кернел үлесі: **7.1%**.

| Көрсеткіш | Base | Codex |
|---|---:|---:|
| Толық оқыту уақыты, с (2 іске қосудың медианасы) | 88.08 | 85.30 |
| Peak GPU memory, MiB | 14345 | 12385 |
| Public eval loss | 2.04963 | 2.04963 |

Speedup **1.0326x** (5%-дан аз, scoring бойынша тең), memory ratio **1.1583x**.
Жады шамамен 13.7% азайды; бұл үнем негізінен retained starter SwiGLU мысалынан.
Код, қайта іске қосу, нақты есептер және шектеулер: [CODEX_VARIANT.md](CODEX_VARIANT.md).
v3 reproduction details: [CODEX_VARIANT.md](CODEX_VARIANT.md).
[Команда нұсқаларын салыстыру](03_validation/VERSION_COMPARISON.md).
Өзге қатысушылардың нұсқалары әлі берілген жоқ; финалдық команда нұсқасы таңдалмаған.

Кейінгі [v4 эксперименті](CODEX_V4_EXPERIMENT.md): 37 кернел тесті, 79 Soup тесті,
10/10 check; speedup 1.0359x. v3-тен анық жақсырақ екені дәлелденген жоқ.
[Командаға көрінетін жұмыс пен commit тарихы](04_submission/CODEX_PROGRESS.md).

## Бастапқы starter іске қосуының есебі

Алғашқы кезеңде қатысушы Kaggle T4-де starter-ді іске қосып, нәтижелерін жіберді.
Ол кезде өз кернеліміз болмаған: 10 тексерудің 8-і PASS, №3 және №4 күтілген FAIL болған.
[Нақты өлшеу есебі](01_launch/BASELINE.md): base 87.87 с / 14345 MiB / loss 2.04969;
starter solution 91.62 с / 12385 MiB / loss 2.04967.
Жады 13.7% азайды, уақыт 4.3% ұзарды. Бұл дайын мысалдың нәтижесі, біздің жеке кернеліміз емес.
Осыдан кейін [GPU профилін алып](02_kernel_research/PROFILE.md), жоғарыда көрсетілген өз нұсқамызды әзірледік.

## Төрт адамға жұмыс бөлу

| Қатысушы | Жеке жұмыс папкасы | Міндеті |
|---|---|---|
| 1, компьютер қолжетімді | [01_launch](01_launch/README.md) | Kaggle, baseline, кейін кодты біріктіру |
| 2 | [02_kernel_research](02_kernel_research/README.md) | Операцияларды зерттеу, кейін кернел әзірлеуге қатысу |
| 3 | [03_validation](03_validation/README.md) | Нәтижелерді, тесттерді және ережелерді тексеру |
| 4 | [04_submission](04_submission/README.md) | README, презентация, видео және тапсыру |

Папканы бөлісу GitHub рұқсатын шектемейді: бұл жұмыс бөлу тәсілі.
Қатысушылар өз GitHub аккаунтынан өз өзгерістерін commit жасасын; бір аккаунтты бөліспеңіздер.
Телефоннан файлды ашып, Edit → Commit changes арқылы өз папкаңыздағы жазбаларды өзгертуге болады.
Тікелей өзгерту құқығы болмаса, капитан платформа арқылы қолжетімділікті реттейді.

Бұл төрт папка міндеттер мен жазбаларға арналған. Барлық жұмыс коды және өз тесттеріміз
`solution/` ішінде тұрады: жюри тек осы код папкасын таза starter-ге көшіреді.
`harness/`, `task/`, `tests/` файлдары өзгертілмейді.

## Қазір не істеу керек

1. [Kaggle](https://www.kaggle.com/) → New Notebook. Settings: **GPU T4 x2**, **Internet On**.
   GPU қолжетімсіз болса, телефон растауын және аккаунттың GPU квотасын тексеріңіз.
2. Осы репозиторийден [kaggle_notebook.ipynb](templates/kaggle_notebook.ipynb) жүктеп алыңыз
   (файлды ашу → Download raw file). Kaggle: File → Import Notebook → файлды таңдаңыз.
3. Бірінші код ұяшығында команданың URL-і алдын ала қойылған:

   ```python
   REPO_URL = "https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai.git"
   COMMIT = ""
   ```

4. Репозиторий private болса: Kaggle Add-ons → Secrets → `GITHUB_TOKEN`, осы notebook үшін On.
   GitHub токеніне осы репозиторий үшін тек Contents: Read-only рұқсатын беріңіз.
   Токенді кодқа немесе чатқа жазбаңыз. Толығырақ: [іске қосу нұсқаулығы](01_launch/README.md).
5. **Run All**. Шаблон бойынша шамамен 10 минут; алғашқы жүктеу ұзағырақ болуы мүмкін.
   Екі GPU берілгенімен, жоба біреуін пайдаланады. Өлшеу кезінде қатар GPU жұмысын қоспаңыз.
6. Уақыт, peak GPU memory, eval loss және тексерулер кестесін
   [BASELINE.md](01_launch/BASELINE.md) ішіне көшіріңіз. Қате болса, қате мәтіні мен ұяшық нөмірін сақтаңыз.

Starter-дегі №3 және №4 тексерулерінің FAIL болуы осы кезеңде күтіледі:
мысал SwiGLU біздің өз кернеліміз болып есептелмейді. Қалған тексерулерді нақты output бойынша тексеріңіз.
Бірінші сәтті іске қосудан кейін ғана операцияны таңдап, өз Triton кернелімізді жазамыз.

## Тапсырманың негізгі талаптары

Мақсат: белгіленген Soup training іске қосуын тездету немесе GPU жадын азайту, сапасын сақтау.
Бағалау: дұрыстық 20, жылдамдық 35, жады 30, README / қайталап іске қосу 15 балл.
Финалда барлық 10 тексеру өтуі тиіс; өз кернелдеріміз бірге training уақытының кемінде 5%-ын алуы керек.
Жеке папкалар мен starter-дің өзі финалдық шешім емес.

Тапсыру: платформадағы репозиторий, нәтижесі сақталған public Kaggle notebook,
3 минутқа дейінгі видео және презентация. Соңғы мерзім: **31.10.2026, 23:59, Астана (UTC+5)**.
[Тапсыру тізімі](04_submission/README.md).

## Starter көзі

Ұйымдастырушылардың жобасы: https://github.com/MakazhanAlpamys/soup-kernel-challenge
Starter commit: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
Оның коды мен мысал кернелі біздің авторлық жұмыс ретінде көрсетілмейді.
`LICENSE` және `NOTICE` сақталған. Команданың бастапқы README-і Git тарихында сақталған.
Төменде ұйымдастырушылардың бастапқы README мәтіні берілген; біздің дайын нәтижеміз емес.

---

# Soup Kernel Challenge

Make `soup train` faster and lighter on a free Kaggle T4 with your own Triton kernels.
Case of [Soup](https://github.com/MakazhanAlpamys/Soup) at Decentrathon 6.0,
October 30 12:00 – October 31 23:59 (Astana time, UTC+5; the deadline is October 31 18:59 UTC).
Teams of up to 4, one result per team. AI agents are allowed.

## The task

`soup train` with `soup-cli==0.77.0` fine-tunes **Qwen2.5-1.5B-Instruct** (4-bit NF4 weights,
LoRA r=16) on 256 Kazakh instructions: batch 4, up to 512 tokens, one epoch (64 steps), on one
of the two GPUs Kaggle gives. Public rows: 0–255 for training, 40000–40255 for your own eval; the
jury trains and evaluates on other, hidden rows. Make this same run take less time or less GPU memory, without
changing what it trains. Your own `@triton.jit` kernels must do real work in it.

Base Soup on a Kaggle T4 (our measurement; yours comes from `harness/run_task.py`):

| | Base Soup 0.77.0 |
|---|---|
| Whole `soup train` command | about 79–83 s (56–61 s of it are the 64 training steps) |
| Peak GPU memory (nvidia-smi) | about 14,350 MiB of 15,360 |
| Held-out loss: no adapter → after training | 2.271 → 2.050 (−9.7%) |

Where the GPU time of a training step goes (our torch.profiler run of base Soup): matrix multiplies about
36%, element-wise ops and dtype casts about 44%, softmax / cross-entropy over the 151,936-token
vocabulary about 7%, attention about 5%, NF4 dequantization about 5%. For scale: Unsloth trains a
similar LoRA on the same data and GPU in about 2,400 MiB (with its own gradient offloading and
LoRA on all seven projections).

## Quick start on Kaggle

1. Work in the GitHub repository that the hackathon platform created for your team: work in other
   repositories is not checked. If it is empty or holds only a README, copy this starter kit into
   it (the **Use this template** button only makes a new repository and cannot fill an existing
   one):

   ```bash
   git clone <your platform repository> && cd <its folder>
   git fetch https://github.com/MakazhanAlpamys/soup-kernel-challenge.git main
   git checkout FETCH_HEAD -- .
   git commit -m "Soup Kernel Challenge starter kit"
   git push origin HEAD
   ```

   The kit's files replace files of the same name (README.md, for example); nothing else changes.
   Without git: **Code → Download ZIP**, unpack it and upload the files.
2. In Kaggle: **Create → New Notebook → File → Import Notebook**, pick
   `templates/kaggle_notebook.ipynb`. Settings: **Accelerator GPU T4 x2**, **Internet on** (a
   phone-verified Kaggle account; the free quota is about 30 GPU hours a week). For a private
   repository add a Kaggle secret `GITHUB_TOKEN` with a read-only token.
3. Put your repository URL in the first code cell and press **Run All** (about 10 minutes). You
   get the base and your solution side by side, `runs/result.json` and the table of checks.
4. Write your kernels in `solution/`, push, press Run All again (it clones your code afresh).

Locally on a Linux machine with a CUDA GPU:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py          # data/train.jsonl, data/eval_public.jsonl
python -m pytest solution -q         # your kernels against plain PyTorch
python harness/run_task.py           # base, then solution -> runs/result.json
python harness/check_rules.py        # the checks -> runs/checks.json
```

Only numbers from a Kaggle T4 count; on Ampere and newer GPUs Soup trains in bf16, a different
path.

## How your code plugs in

`harness/run_task.py` starts each run in a fresh process (`harness/_train.py`), which imports
your package `solution/` and calls `solution.apply()` before Soup builds the model and the
trainer. From there you can patch anything in transformers, peft, bitsandbytes or Soup, the way
Unsloth patches models. Everything of yours lives in `solution/`: the jury copies only that
folder into a fresh copy of this kit.

`solution/swiglu.py` is a complete example: a fused SwiGLU forward and backward kernel in Triton,
an autograd function around it, and the patch of `Qwen2MLP.forward`. It works out of the box, so
the first Run All already goes through the whole path. Its kernels do not count as yours under any
name or rewrite (`check_rules.py` recognizes renamed copies and flags close rewrites, which the
jury reads by hand), so on a fresh
template checks 3 and 4 fail until you add kernels of your own. You may keep the example running
next to them: on its own it lowers peak memory by about 2,000 MiB (14%), keeps the speed within a
few percent and takes about 5% of the training time.

What the base run does on a T4 (Soup 0.77.0 with this config), so you know where time and memory
go:

- weights in 4-bit NF4 (bitsandbytes, double quant), compute in fp16; `prepare_model_for_kbit_training`
  keeps the norms, biases and the tied embedding / `lm_head` in fp32;
- LoRA r=16, alpha=32, dropout 0.05 on `q_proj` and `v_proj` only (`target_modules: auto`);
- no gradient checkpointing, SDPA attention, plain `adamw_torch`, cosine schedule;
- the trainer is TRL's `SFTTrainer` (Soup's subclass `RewindSFTTrainer`): the loss covers the
  assistant tokens only; every step TRL also computes entropy and token accuracy from the full
  logits, and Soup writes each row's loss to `rewind.jsonl`. These are Soup outputs: keep them
  with the same values (you may compute them faster).

Places to hook in `apply()` (all looked up at call time):

- model classes, e.g. `transformers.models.qwen2.modeling_qwen2.Qwen2MLP.forward` (the example);
- `soup_cli.trainer.sft.SFTTrainerWrapper._setup_transformers`: when it returns, `self.model` is
  the final 4-bit PEFT model, before the trainer is built;
- `soup_cli.trainer.sft.SFTTrainerWrapper.train`: the trainer and the model exist, training has
  not started;
- `trl.SFTTrainer` (Soup builds its trainer class from it during setup), peft and bitsandbytes
  functions.

Patch narrowly. `apply()` also runs before Soup's regression tests (check 10), where some tests
call these methods with fake transformers/peft modules; a patch that assumes a real Qwen model
everywhere fails them. Guard it (`if not isinstance(self.model, torch.nn.Module): return ...`) and
run `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q` yourself. Do not add trainable parameters:
the adapter must stay the same LoRA (check 5). Launch your kernels normally, not through CUDA
graphs, or the harness cannot see them.

| Path | What it is | Yours to edit |
|---|---|---|
| `solution/` | your kernels, `apply()` and their tests | yes |
| `task/task.yaml` | the fixed Soup config | no |
| `task/prepare_data.py` | downloads the data, writes the public rows | no |
| `harness/run_task.py` | runs base and solution, measures time, memory, quality | no |
| `harness/_train.py` | one `soup train` run in a process, with or without `apply()` | no |
| `harness/eval_adapter.py` | quality: held-out loss of an adapter | no |
| `harness/check_rules.py` | the checks, one line each | no |
| `harness/score.py` | ratios and points | no |
| `harness/common.py`, `harness/example_reference.py` | shared settings, a frozen copy of the example | no |
| `tests/` | Soup's regression tests: 79 tests in five files | no |
| `templates/` | Kaggle notebook, `SUBMISSION.md` | copy and fill |

## What is measured

- **Time**: the wall time of the whole `soup train` command, from start to saved adapter, timed
  from outside the process. Triton compilation and autotuning count: every run starts with empty
  compiler caches. Base and solution run in the same session; the score uses the ratio.
- **Memory**: peak GPU memory, sampled by `nvidia-smi` from outside the process every 100 ms and
  summed over all GPUs. Both sides run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  so the peak follows the tensors that are alive rather than allocator fragmentation.
- **Quality**: `harness/eval_adapter.py`, the mean loss on the answer tokens of held-out rows. It
  loads your adapter with plain transformers + peft, without your code.
- **Your kernels' share**: inside the same measured runs, every launch of a `@triton.jit` kernel
  defined in `solution/` is timed with CUDA events from step 3 on, against the training time of
  those steps.

The jury runs the same scripts on hidden rows of the same dataset with another seed, base and
solution twice each in the order base, solution, solution, base, in a fresh copy of this kit with
only your `solution/`.

## Rules

Allowed: your own Triton kernels, autograd functions and patches; recomputation or offloading in
your own code (its time counts); `torch.compile` (its kernels are not yours); packages torch,
triton, transformers, peft, trl, bitsandbytes and `soup-cli==0.77.0`; ideas from other kernels
when you name the source.

Not allowed:

- ready-made kernel libraries: unsloth, liger-kernel, xformers, flash-attn, torchao and the rest
  of the list in `harness/common.py`, and installing packages during the run;
- copying other projects' kernel code as your own; kernels whose output training does not use
  (busy work) do not count;
- changing the task: model, data, batches and their order, number of steps, LoRA settings
  (dropout included), optimizer and schedule, Soup's switches (`backend: unsloth`,
  `gradient_checkpointing`,
  `activation_offloading`, `fast_lora`, `use_liger`, `use_cut_ce`, `packing`, `padding_free` and
  the like), and Soup's outputs. Removing padding inside a batch in your own code is fine; mixing
  rows across batches is not;
- getting around the measurement: another GPU, background processes, different behaviour when
  measured or tested, other allocator settings, replacing the saved adapter, prebuilt compiled
  caches;
- reading the hidden rows or any files outside your repository and the task data;
- changing `harness/`, `tests/` or `task/` (the jury uses its own copy anyway).

## Checks

`harness/check_rules.py` prints one line per check and writes `runs/checks.json`. Every check
must pass on the jury's run, and the jury reads your code: its review decides in disputes.

| # | Check | Limit |
|---|---|---|
| 1 | every base and solution run finished | all runs exit 0 |
| 2 | no ready-made kernel libraries | not imported in `solution/`; not loaded in the runs (torchao is checked in `solution/` only, peft may load it by itself); no packages installed during the run |
| 3 | your `@triton.jit` kernels in `solution/` run in training | launched in every solution run; the example's kernels do not count |
| 4 | your kernels together take enough of the training time | at least 5%, CUDA events, steps 3+, lowest run |
| 5 | same task and Soup outputs | steps, adapter config, adapter tensor names and shapes, trainer settings, logged metric names, tokens per step, rewind rows and token counts equal; entropy, accuracy and rewind row losses within 3% |
| 6 | loss curve follows the base | first step within 1%, mean of all steps within 3% |
| 7 | quality | eval loss at most 0.5% above the base |
| 8 | not worse than the base | time and memory at most 10% above the base |
| 9 | clean measurement | a Tesla T4, GPUs idle at start (under 1,000 MiB), one GPU, harness allocator, nvidia-smi peak at least 97% of PyTorch's reserved peak |
| 10 | Soup regression tests with your patches | 79 tests pass |

When a check fails, its line says why. Logs of every run are in `runs/<run>/train.log` and
`runs/eval_*.log`; per-step metrics in `runs/<run>/adapter/checkpoint-*/trainer_state.json`.

While iterating: `python harness/run_task.py --only solution --out runs_try` trains only your
solution (checks 1 and 5–8 need the base, so they fail there); `--profile` adds one profiled run with
the GPU time of every CUDA kernel in `runs/solution_profile/profile.json`.

## Scoring (100 points)

| Criterion | Points |
|---|---|
| Works and correct: every check passes and the code review agrees | 20 |
| Speed: `35 * ln(s) / ln(best s)`, s = base time / your time | 35 |
| Memory: `30 * ln(m) / ln(best m)`, m = base peak / your peak | 30 |
| README and reproducibility | 15 |

A gain under 5% counts as a tie (ratio 1.0). "Best" is the best admitted team. A team that fails a
check, or whose code or README the jury does not accept, is not admitted and gets no speed or
memory points. `harness/score.py` prints your ratios.

The jury gives the README points in three parts, up to 5 each: what you did, how to run it,
results and limits. If two teams have the same total, the one with the better memory result ranks
higher.

## Submit

Everything is submitted on the hackathon platform: your repository, the video (up to 3 minutes)
and the presentation. Rewrite this README for your solution (what you did, how to run it,
results, limits) and put the link to your public Kaggle notebook in it. Fill
`templates/SUBMISSION.md` and put it in the root of your repository: links to the final commit,
the public Kaggle notebook (a saved version with `COMMIT` set to that commit), the video and the
presentation. The Kaggle link must be in both the README and `SUBMISSION.md`. By the deadline the
repository must be open to the jury: public, or with access for the jury through the platform.

## License

Apache-2.0, see `LICENSE` and `NOTICE`. The five test files come from Soup v0.77.0 (Apache-2.0).
Data: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2)
(MIT on its card), a Kazakh translation of [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
(CC BY-NC 4.0); it is downloaded at run time for this benchmark. Model:
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0).
