# Операцияны таңдау жазбалары

Күйі: өз кернелдеріміз жазылып, `solution/` ішінде жұмыс істейді. Өлшенген нәтижелер: `01_launch/BASELINE.md`.

| Кандидат операция | Қай функцияны алмастырады | Күтілетін пайда / негіздеме | Backward / дәлдік тәуекелі | Идея көзі |
|---|---|---|---|---|
| NF4 dequant (`nf4.py`) | bitsandbytes 4-bit салмақтарын ашу | Қатып тұрған салмақтар fp16-ға бір кернелде ашылады | Салмақ оқытылмайды; градиент тек кіріске өтеді | QLoRA NF4 форматы, bitsandbytes |
| Residual add + RMSNorm (`rmsnorm.py`) | `Qwen2RMSNorm` және оның алдындағы қосу | Бір өту, жадқа аз жазу | Backward кернелі бар; қосынды fp32-де | Qwen2 RMSNorm анықтамасы |
| Нақты токендер + RoPE (`unpad.py`) | padding бар attention жолы | Padding токендері есептелмейді; RoPE мен head-тарды қайта орналастыру бір кернелде | Backward бар; batch-тер араласпайды | Varlen attention идеясы |
| Chunked lm_head + CE (`lmhead.py`) | lm_head және CrossEntropy, толық logits | `[B, L, 151936]` logits жадта сақталмайды; online softmax | Градиент сол кернелде; GradScaler масштабы ескеріледі | Online softmax (Milakov & Gimelshein, 2018) |
| LoRA dropout нақты жолдарда (`dropout.py`) | `nn.Dropout` padding бар пішінде | Тек нақты жолдар өңделеді | RNG нәтижесі base-пен бірдей сақталады | Мерейдің `66bcd9b` шешімі |

- Бірінші baseline қорытындысы: base 79–80 с, peak 14345 MiB; оқыту қадамдары шамамен 55–58 с, қадамнан тыс шамамен 22–24 с.
- Таңдалған операциялар және себебі: уақыт пен жадтың көбі қатып тұрған 4-bit салмақтарда, padding токендерінде және үлкен lm_head logits-та жұмсалды.
- Код жазылған `solution/` файлдары: `nf4.py`, `rmsnorm.py`, `unpad.py`, `lmhead.py`, `dropout.py`, `mlp.py`, `rope.py`, `layer.py`; warm-up: `warmup.py`.
- Тексеру жоспары: PyTorch-пен forward және backward салыстыруы (`solution/test_*.py`), нақты shapes/dtypes, padding жоқ batch сияқты шеткі жағдайлар.
- Нақты өлшеу: өз кернелдерінің үлесі v1-де 5.2%, v2-де 11.5% (steps 3+); ≥5% талабы орындалады.
