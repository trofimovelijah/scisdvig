# Содержание инструкции запуска

> [!NOTE]
> Заполнить вместе с первым работающим запуском, не как пустую заготовку.

| Раздел       | Обязательные сведения                                                |
| :----------- | :------------------------------------------------------------------- |
| Среда        | ОС, архитектура, память, версии языка, менеджера пакетов и runtime   |
| Модель       | Точный артефакт, revision, формат, квантование, лицензия             |
| Подготовка   | Загрузка весов и вспомогательных моделей, каталоги, свободное место  |
| Конфигурация | Endpoint, пути, контекст, параметры генерации, лимиты                |
| Запуск       | Реальные команды и проверка готовности                               |
| Остановка    | Отмена, остановка процесса, судьба временных файлов                  |
| Диагностика  | Логи, run_id, call_id, коды ошибок, безопасный диагностический пакет |
| Smoke-тест   | Разрешённый контрольный документ и ожидаемые свойства результата     |

## Запуск на локальном инференсе. Общее
Для списка достаточно трёх практических маршрутов:

1. MLX / `mlx-lm`: TranslateGemma 4B, 12B, 27B; TowerInstruct после конвертации.
2. llama.cpp / GGUF: Index-Translate 2B/9B, Hunyuan-MT, Chimera, Qwen3; также MADLAD, но без chat template.
3. Дополнительный Transformers / PyTorch: OPUS-MT и M2M100; резервный, наиболее прямой путь для MADLAD.

Ollama — альтернативный способ управлять совместимыми моделями и HTTP API, а не универсальный конвертер любых весов. MLX-квантизация, GGUF-квантизация и исходный Safetensors — разные артефакты. Наличие расширения `.gguf` само по себе не гарантирует совместимость архитектуры и метаданных. 

TranslateGemma 27B в 4-bit — реалистичный кандидат для 64 GB: опубликованный MLX-артефакт занимает около 15.2 GB. Это не измерение полного расхода RAM и не обещание определённой скорости на M5. Qwen3-30B-A3B также следует брать квантованным: у него 30.5B параметров всего и 3.3B активных, но хранить приходится не только активных экспертов.

## Что установить с нуля — до скачивания моделей

### Проверить Mac и подготовить инструменты

Все команды выполнять в Terminal, нативно в macOS. Для этой инструкции не нужны CUDA, Linux VM или контейнеры. llama.cpp использует Metal на macOS; Ollama поддерживает CPU/GPU Apple M-series.

```bash
uname -m
sw_vers
system_profiler SPHardwareDataType
sysctl -n hw.memsize
df -h "$HOME"
```

Ожидается `arm64`. Если видно `x86_64`, сначала проверить запуск Terminal через Rosetta. Сведения `system_profiler` могут содержать серийный номер — не публиковать их целиком.

Установить Command Line Tools, дождаться завершения GUI-установки, затем продолжить:

```bash
xcode-select --install
```

Для новых окон Terminal выполнить предложенные установщиком команды настройки PATH. Не добавлять одну и ту же строку в `~/.zprofile` многократно.

```bash
brew install cmake jq llama.cpp
mkdir -p "$HOME/local-translation"/{scripts,models,data,results,logs}
cd "$HOME/local-translation"
```

### Отдельное Python-окружение для MLX

Установка `mlx-lm` и запуск моделей описаны в официальном [MLX LM](https://github.com/ml-explore/mlx-lm).-

```bash
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade mlx-lm huggingface_hub sentencepiece
python -c 'import platform, mlx.core as mx; print(platform.machine()); print(mx.default_device()); print(mx.metal.is_available())'
mlx_lm.generate --help
llama-cli --version
llama-server --help > logs/llama-server-help.txt
```

Контроль: архитектура `arm64`, Metal доступен (`True`), утилиты находятся. При ошибке Metal сначала проверить обновления macOS и совместимость установленного MLX; не переходить к скачиванию 27B.

В каждом новом окне:

```bash
cd "$HOME/local-translation"
source .venv-mlx/bin/activate
```

### Hugging Face и доступ к Gemma

Для исходных Google Gemma требуется принять условия на странице модели, войдя в Hugging Face. Если конкретная загрузка отвечает 401/403, проверить доступ к этому репозиторию, а не менять код инференса.

```bash
hf auth login
hf auth whoami
```

Вставлять личный read-токен только в интерактивный запрос. Не записывать токен в Markdown, Git, логи или команды агента.

### Ollama — опционально

Установить приложение с официальной страницы `https://ollama.com/download`, переместить в Applications и запустить. При первом старте разрешить настройку CLI. Это документированный способ установки для macOS.

```bash
ollama --version
curl --fail http://127.0.0.1:11434/api/tags
```

Если GUI-приложение уже обслуживает порт, не запускать параллельно `ollama serve`. Если выбран только CLI и сервер не запущен, `ollama serve` выполнять в отдельном Terminal.

### Входной текст для первого запроса

```bash
cat > data/source.en.txt <<'EOF'
The UE shall initiate the registration procedure when the timer expires.
The server MUST NOT reuse the nonce. See clause 5.3.2 and RFC 9110.
EOF
```

Сначала запускать одну модель и один запрос. Не скачивать весь список одновременно. Практический резерв: оставить не менее 100 GB свободного SSD; это эксплуатационная рекомендация, не требование производителей. Исходные веса, локальные конверсии, Hugging Face cache и Ollama могут занимать место одновременно.

## Полная группировка списка

Обозначения: «выбран» — основной путь этой инструкции; «альтернатива» — отдельный формат той же исходной модели, не обязательно идентичная квантизация; «не выбран» — здесь нет подтверждённого и достаточно простого рецепта, а не доказанная невозможность.

| Модель пользователя | Точный исходный ID / выбранный артефакт | Выбранный маршрут | Другие варианты / особенности |
|---|---|---|---|
| translategemma-12b-it-8bit | `mlx-community/translategemma-12b-it-8bit` ← `google/translategemma-12b-it` | MLX, §3 | Ollama 12b — другой артефакт; не считать его теми же 8-bit весами. |
| translategemma-4b-it-4bit | `mlx-community/translategemma-4b-it-4bit` | MLX, §3 | Быстрый предпросмотр — предполагаемая роль, скорость проверить локально. |
| translateGemma-27B-it | `google/translategemma-27b-it`; выбран `mlx-community/translategemma-27b-it-4bit` | MLX 4-bit, §3 | Ollama `translategemma:27b`; BF16 не брать стартовым вариантом. |
| Index-Translate-2B | `IndexTeam/Index-Translate-2B-GGUF` | llama.cpp Q8_0, §4 | Официальный GGUF; Ollama — альтернатива. |
| Index-Translate-9B | `IndexTeam/Index-Translate-9B-GGUF` | llama.cpp Q8_0, §4 | Q4_K_M — более компактная альтернатива. |
| Hunyuan-MT-7B | `tencent/Hunyuan-MT-7B`; `DevQuasar/tencent.Hunyuan-MT-7B-GGUF` | llama.cpp, §5 | Есть MLX `mlx-community/Hunyuan-MT-7B-8bit`, но общий рецепт ниже использует GGUF. Не заменять на HY-MT1.5 / Hy-MT2. |
| Hunyuan-MT-Chimera-7B | `tencent/Hunyuan-MT-Chimera-7B`; `KnutJaegersberg/Hunyuan-MT-Chimera-7B-Q8_0-GGUF` | llama.cpp Q8_0, §5 | Это синтез нескольких переводов, а не обычный первый переводчик. |
| MADLAD400-3B | Здесь принят переводческий checkpoint `google/madlad400-3b-mt` | llama.cpp T5 GGUF, §8 | Transformers CPU — резерв. Старые Candle GGUF нельзя механически передавать llama.cpp. |
| OPUS-MT EN→RU | Здесь принят `Helsinki-NLP/opus-mt-en-ru` | Transformers CPU, §9 | Marian encoder-decoder; готовый подтверждённый путь в тройке MLX/llama.cpp/Ollama не выбран. |
| M2M100-418M | `facebook/m2m100_418M` | Transformers CPU, §9 | Отдельные source language и forced target language; не чат. |
| TowerInstruct-7B-v0.2 | `Unbabel/TowerInstruct-7B-v0.2` | Конвертация в MLX 8-bit, §6 | Llama-архитектура, но ChatML-шаблон; нельзя подменять v0.1. |
| Qwen3-30B-A3B-Instruct-2507 | `Qwen/Qwen3-30B-A3B-Instruct-2507`; GGUF от bartowski | llama.cpp Q4_K_M, §7 | MLX-конверсия также существует; базовый `qwen3:30b` не использовать как синоним 2507. |

Если условие «только MLX, llama.cpp, Ollama» абсолютно жёсткое, OPUS-MT и M2M100 следует пометить `BLOCKED_RUNTIME`, а §9 не выполнять. Наличие 64 GB RAM не устраняет несовместимость загрузчика и архитектуры.

## TranslateGemma: один скрипт для трёх размеров

### Важный контракт запроса

Использовать специальный шаблон Google: `content` — список из одного объекта с `type`, `source_lang_code`, `target_lang_code`, `text`. Не добавлять `system message`. Автоматически показанный в некоторых карточках пример «Write a story about Einstein» не соответствует документированному контракту перевода. Официальная карточка указывает общий входной контекст 2K токенов; указанное у Ollama архитектурное окно 128K не доказывает качество перевода на 128K.

Создать `scripts/translate_mlx.py`:

```python
import argparse
import sys
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
p.add_argument('--max-tokens', type=int, default=768)
a = p.parse_args()
text = Path(a.input).read_text(encoding='utf-8').strip()
if not text:
    raise SystemExit('EMPTY_INPUT')
model, tokenizer = load(a.model)
messages = [{'role': 'user', 'content': [{
    'type': 'text', 'source_lang_code': 'en',
    'target_lang_code': 'ru', 'text': text,
}]}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
input_tokens = len(tokenizer.encode(prompt))
print(f'input_tokens={input_tokens}', file=sys.stderr)
if input_tokens > 1800:
    raise SystemExit('INPUT_TOO_LONG: split text into shorter segments')
answer = generate(model, tokenizer, prompt=prompt,
                  max_tokens=a.max_tokens,
                  sampler=make_sampler(temp=0), verbose=False)
if not answer.strip():
    raise SystemExit('EMPTY_OUTPUT')
Path(a.output).write_text(answer + '\n', encoding='utf-8')
print(answer)
```

Порог 1800 — консервативный локальный барьер для входа, не новый заявленный лимит модели. Размер ответа задаётся отдельно. API `load`/`generate` и greedy sampler поддерживаются MLX LM. 

Запускать последовательно:

```bash
python scripts/translate_mlx.py \
  --model mlx-community/translategemma-4b-it-4bit \
  --input data/source.en.txt --output results/tg4.ru.txt

python scripts/translate_mlx.py \
  --model mlx-community/translategemma-12b-it-8bit \
  --input data/source.en.txt --output results/tg12.ru.txt

python scripts/translate_mlx.py \
  --model mlx-community/translategemma-27b-it-4bit \
  --input data/source.en.txt --output results/tg27.ru.txt
```

MLX-веса: приблизительно 2.18 GB / 12.5 GB / 15.2 GB соответственно. Полный расход памяти выше из-за активаций, кэша и работы runtime. На 64 GB все три выглядят подходящими для поочерёдного текстового теста; это оценка вместимости, не результат замера. [web:64][web:74][web:65]

Если `apply_chat_template` отклоняет структурированный `content`, остановиться и проверить `chat_template.jinja` выбранного snapshot. Не исправлять шаблон вслепую и не подменять модель. Для исходного Google-шаблона допустимо загрузить tokenizer из соответствующего `google/translategemma-*-it`, если доступ получен, но такую замену обязательно записать в журнал эксперимента.

### 3.2. Альтернатива через Ollama

В Ollama используется готовый текстовый prompt из её [руководства TranslateGemma](https://ollama.com/library/translategemma), а не вложенный HF `content`. [web:70]

```bash
ollama pull translategemma:12b
python - <<'PY' > data/tg-ollama-request.json
import json
from pathlib import Path
text = Path('data/source.en.txt').read_text()
prompt = (
 'You are a professional English (en) to Russian (ru) translator. '
 'Your goal is to accurately convey the meaning and nuances of the original '
 'English text while adhering to Russian grammar, vocabulary, and cultural sensitivities.\n'
 'Produce only the Russian translation, without any additional explanations or commentary. '
 'Please translate the following English text into Russian:\n\n\n' + text
)
print(json.dumps({'model': 'translategemma:12b',
 'messages': [{'role': 'user', 'content': prompt}],
 'stream': False, 'keep_alive': 0,
 'options': {'temperature': 0, 'num_ctx': 4096, 'num_predict': 768}}))
PY
curl --fail --max-time 900 http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  --data-binary @data/tg-ollama-request.json \
  > results/tg-ollama.json
jq -er '.message.content | select(length > 0)' results/tg-ollama.json
```

Для 4B/27B менять значение `model` в JSON и команду `pull` на `translategemma:4b` / `translategemma:27b`. Эти теги существуют, но не обещают совпадения с выбранными MLX-квантами. Вход для первичных опытов всё равно оставлять коротким. [web:70]

## 4. Index-Translate 2B / 9B: llama.cpp

Выбраны официальные GGUF Q8_0: около 2.08 GB для 2B и 9.79 GB для 9B. Для 64 GB нет необходимости начинать с самого агрессивного сжатия. Рекомендация Q8_0 здесь основана на запасе памяти, не на собственном сравнении качества. [web:66][web:7]

### 4.1. Поднять одну модель

```bash
llama-server \
  -hf IndexTeam/Index-Translate-2B-GGUF:Q8_0 \
  -ngl 99 -c 4096 --parallel 1 \
  --jinja --chat-template-kwargs '{"enable_thinking":false}' \
  --host 127.0.0.1 --port 8080 --alias index-translate
```

Для 9B заменить только репозиторий на `IndexTeam/Index-Translate-9B-GGUF:Q8_0`. Для компактного теста вместо Q8_0 использовать Q4_K_M. Дождаться готовности сервера. В другом Terminal:

```bash
curl --fail http://127.0.0.1:8080/health
```

Официальный prompt и отключение thinking описаны в карточках Index. Для текстового ввода `mmproj` не требуется. [web:66][web:7]

### 4.2. Первый запрос

```bash
python - <<'PY' > data/index-request.json
import json
from pathlib import Path
text = Path('data/source.en.txt').read_text()
prompt = '请将以下文本翻译为俄语，直接输出翻译结果，不要进行任何解释。\n\n' + text
print(json.dumps({'model': 'index-translate',
 'messages': [{'role': 'user', 'content': prompt}],
 'temperature': 0, 'max_tokens': 768, 'stream': False}, ensure_ascii=False))
PY
curl --fail --max-time 900 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @data/index-request.json > results/index.json
jq -er '.choices[0].message.content | select(length > 0)' results/index.json \
  > results/index.ru.txt
cat results/index.ru.txt
```

Альтернатива загрузки через Ollama, документированная на страницах GGUF:

```bash
ollama run hf.co/IndexTeam/Index-Translate-2B-GGUF:Q8_0
# либо
ollama run hf.co/IndexTeam/Index-Translate-9B-GGUF:Q8_0
```

Это альтернативный маршрут: не считать поддержку одной версии llama.cpp гарантией поддержки в установленной Ollama. Если возникают ошибки архитектуры `qwen35`, сначала обновить runtime и повторить короткий тест; не скачивать другую модель под тем же именем. [web:66][web:7]

## 5. Hunyuan-MT и Chimera: одинаковый движок, разные входы

Hunyuan-MT переводит исходник; Chimera получает исходник и несколько переводов-кандидатов и синтезирует итог. Утверждение разработчиков об улучшении качества не означает, что любой ансамбль на ваших стандартах обязательно будет лучше. [web:33][web:115]

### 5.1. Hunyuan-MT-7B

Есть community GGUF исходной модели в `DevQuasar/tencent.Hunyuan-MT-7B-GGUF`. Это не `Hunyuan-7B-Instruct` и не следующее поколение HY-MT. [web:109]

Используем непосредственно подтверждённый F16-файл, не угадывая наличие конкретного кванта:

```bash
llama-server \
  -hf DevQuasar/tencent.Hunyuan-MT-7B-GGUF \
  --hf-file tencent.Hunyuan-MT-7B.f16.gguf \
  -ngl 99 -c 4096 --parallel 1 \
  --host 127.0.0.1 --port 8080 --alias hunyuan-mt
```

F16-веса этого размера — реалистичный последовательный тест на 64 GB, но следить за фактическим memory pressure. Веса загрузятся один раз. Если chat template отсутствует, не применять шаблон Llama по догадке: остановиться для настройки с tokenizer исходной `tencent/Hunyuan-MT-7B` и `apply_chat_template(..., add_generation_prompt=False)`. Не продолжать эксперимент с неподтверждённым форматом.

В другом Terminal создать JSON и отправить запрос:

```bash
python - <<'PY' > data/hunyuan-request.json
import json
from pathlib import Path
prompt = ('Translate the following segment into Russian, without additional explanation.\n\n'
          + Path('data/source.en.txt').read_text())
print(json.dumps({'model': 'hunyuan-mt',
 'messages': [{'role': 'user', 'content': prompt}],
 'temperature': 0.7, 'top_p': 0.6, 'top_k': 20,
 'repeat_penalty': 1.05, 'max_tokens': 768, 'stream': False}))
PY
curl --fail --max-time 900 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @data/hunyuan-request.json > results/hunyuan.json
jq -er '.choices[0].message.content | select(length > 0)' results/hunyuan.json \
  > results/hunyuan.ru.txt
```

Параметры 0.7 / 0.6 / 20 / 1.05 взяты из карточки Tencent. Для отдельного воспроизводимого greedy-эксперимента можно задать temperature=0, отметив отличие от авторского пресета. [web:8][web:33]

### 5.2. Chimera

Остановить предыдущий сервер через Ctrl+C. Запустить подтверждённую Q8_0-конверсию: [web:114]

```bash
llama-server \
  -hf KnutJaegersberg/Hunyuan-MT-Chimera-7B-Q8_0-GGUF:Q8_0 \
  -ngl 99 -c 8192 --parallel 1 \
  --host 127.0.0.1 --port 8080 --alias chimera
```

Сначала получить переводы одного и того же исходника в `results/tg12.ru.txt`, `results/index.ru.txt`, `results/hunyuan.ru.txt`. Число кандидатов три — наш минимальный практический пример; официальный пример показывает шесть. Не дополнять набор фиктивными или повторёнными ответами. [web:33]

```bash
python - <<'PY' > data/chimera-request.json
import json
from pathlib import Path
source = Path('data/source.en.txt').read_text().strip()
files = ['results/tg12.ru.txt', 'results/index.ru.txt', 'results/hunyuan.ru.txt']
translations = [Path(f).read_text().strip() for f in files]
if not all(translations):
    raise SystemExit('EMPTY_CANDIDATE')
prompt = (
 'Analyze the following multiple Russian translations of the English segment '
 'surrounded in triple backticks and generate a single refined Russian translation. '
 'Only output the refined translation, do not explain.\n\n'
 f'The English segment:\n```{source}```\n\n'
 'The multiple Russian translations:\n' +
 '\n'.join(f'{i}. ```{text}```' for i, text in enumerate(translations, 1))
)
print(json.dumps({'model': 'chimera',
 'messages': [{'role': 'user', 'content': prompt}],
 'temperature': 0.7, 'top_p': 0.6, 'top_k': 20,
 'repeat_penalty': 1.05, 'max_tokens': 768, 'stream': False}))
PY
curl --fail --max-time 900 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @data/chimera-request.json > results/chimera.json
jq -er '.choices[0].message.content | select(length > 0)' results/chimera.json \
  > results/chimera.ru.txt
```

Для обоих Hunyuan сверить GGUF chat template с исходным tokenizer: Tencent приводит `add_generation_prompt=False`, поэтому нет оснований обещать побитовое совпадение произвольного chat-server с авторским Transformers-примером. Если добавление assistant prefix меняет результат, это отдельная конфигурационная переменная, которую нужно зафиксировать. [web:33]

## 6. TowerInstruct-7B-v0.2: локальная конвертация в MLX

Берём именно v0.2. Репозиторий MLX Tower v0.1 не является её заменой. У Tower v0.2 Llama-архитектура, но обучающий формат ChatML без system prompt; русский входит в поддерживаемые языки. Лицензия включает CC-BY-NC-4.0 и условия Llama 2 — до рабочего/коммерческого использования проверить применимость. [web:17][web:50][web:60][web:55]

В активном `.venv-mlx`:

```bash
mlx_lm.convert \
  --hf-path Unbabel/TowerInstruct-7B-v0.2 \
  --mlx-path models/tower-v02-8bit \
  -q --q-bits 8
```

Это конвертация, а не fine-tuning. Нужны место под исходные веса и отдельный результат. Команда и параметры поддерживаются конвертером MLX; конкретный итог проверить smoke-тестом. Если выходной каталог уже существует, конвертер останавливается: не удалять автоматически, выбрать новый каталог. [web:77]

```bash
mlx_lm.generate \
  --model models/tower-v02-8bit \
  --prompt "Translate the following text from English into Russian.
English: The UE shall initiate the registration procedure when the timer expires.
Russian:" \
  --max-tokens 256 --temp 0
```

`mlx_lm.generate` должен применить сохранённый tokenizer chat template; не оборачивать prompt дополнительно в `[INST]`. При отсутствии шаблона остановиться и проверить tokenizer исходного репозитория. Пример инструкции English/Russian — адаптация официального формата задания перевода. [web:60]

## 7. Qwen3-30B-A3B-Instruct-2507: llama.cpp

Выбран community GGUF от bartowski, явно производный от checkpoint 2507. Это универсальная instruct MoE, не узкоспециализированный переводчик. Она non-thinking: отключение thinking для этой версии не требуется. [web:83][web:75]

```bash
llama-server \
  -hf bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M \
  -ngl 99 -c 8192 --parallel 1 \
  --host 127.0.0.1 --port 8080 --alias qwen-translate
```

Q4_K_M выбран как стартовый баланс памяти и качества, не как доказанно оптимальный квант для EN→RU. Начинать с 8K контекста, не с архитектурного максимума 262144. Для большей точности квантования можно отдельно исследовать Q6/Q8, но только после измерения RAM и качества. [web:75][web:83]

```bash
cat > data/qwen-request.json <<'EOF'
{
  "model": "qwen-translate",
  "messages": [{"role": "user", "content": "Translate the following technical text from English into Russian. Output only the translation. Preserve UE, RFC 9110, numbers and clause references. Do not omit negation or change requirement strength.\n\nThe UE shall initiate the registration procedure when the timer expires. The server MUST NOT reuse the nonce. See clause 5.3.2 and RFC 9110."}],
  "temperature": 0,
  "max_tokens": 768,
  "stream": false
}
EOF
curl --fail --max-time 900 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @data/qwen-request.json > results/qwen.json
jq -er '.choices[0].message.content | select(length > 0)' results/qwen.json \
  > results/qwen.ru.txt
```

Здесь temperature=0 — наш пресет для сравнения переводов; разработчик Qwen рекомендует общий sampling-пресет 0.7 / top_p=0.8 / top_k=20. Не смешивать эти режимы в одном сравнительном отчёте. [web:75]

Альтернатива Ollama с сохранением точной идентичности checkpoint:

```bash
ollama run hf.co/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
```

Это использование HF GGUF, а не утверждение о наличии равнозначного краткого тега в каталоге Ollama. При несовместимости шаблона использовать выбранный llama.cpp, а не переименовывать базовую Qwen3. Импорт GGUF — документированная возможность Ollama. [web:73][web:83]

## 8. MADLAD400-3B: T5 без чата

Точное принятое имя — `google/madlad400-3b-mt`. Это encoder-decoder T5; язык результата задаётся префиксом `<2ru>`, а не инструкцией в system/user chat. Исходная карточка также содержит Candle GGUF: они не обязательно совместимы с llama.cpp. Выбрана отдельная конверсия под llama.cpp. [web:67][web:95][web:100]

```bash
llama-cli \
  --hf-repo notjjustnumbers/madlad400-3b-mt-Q4_K_M-GGUF \
  --hf-file madlad400-3b-mt-q4_k_m.gguf \
  -ngl 99 -c 2048 -n 256 --temp 0 --no-conversation \
  -p '<2ru> The server MUST NOT reuse the nonce.'
```

`--no-conversation` принципиален: нужен plain completion, без ChatML. Размер выбранной Q4_K_M-конверсии около 1.86 GB. Если Metal backend не поддержит конкретную операцию T5 в вашей сборке, повторить короткий тест с `-ngl 0` либо перейти на Transformers в §9. Не считать невозможность GPU-offload невозможностью CPU-инференса. [web:93][web:76]

Для первого теста подавать одно предложение, затем короткие абзацы. Не подавать целиком RFC или страницу PDF. Здесь не обещается поддержка MADLAD в Ollama: наличие GGUF не доказывает доступность encoder-decoder пути в конкретной версии Ollama.

## 9. Исключения: OPUS-MT, M2M100 и резерв MADLAD

### 9.1. Почему дополнительный runtime

OPUS — Marian, M2M100 — собственная encoder-decoder архитектура; официальные примеры используют seq2seq Transformers. В рамках исследования не подтверждён простой актуальный маршрут для этих двух через выбранную тройку. Поэтому ниже используется дополнительное изолированное окружение, без подмены моделей. CPU выбран как максимально простой базовый путь; MPS — отдельная последующая оптимизация, не обещание совместимости всех операций. [web:69][web:68]

```bash
cd "$HOME/local-translation"
/opt/homebrew/bin/python3.12 -m venv .venv-seq2seq
source .venv-seq2seq/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.6' 'transformers>=4.56,<5' sentencepiece sacremoses protobuf
```

Диапазон Transformers ограничен 4.x для изоляции от изменений major API. Это стартовый диапазон, не заявленный протестированный lock-файл. После успеха выполнить `pip freeze`.

Создать `scripts/translate_seq2seq.py`:

```python
import argparse
from pathlib import Path
import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
)

p = argparse.ArgumentParser()
p.add_argument('--family', choices=['opus', 'm2m100', 'madlad'], required=True)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
text = Path(a.input).read_text(encoding='utf-8').strip()
if not text:
    raise SystemExit('EMPTY_INPUT')
extra = {}
if a.family == 'opus':
    repo = 'Helsinki-NLP/opus-mt-en-ru'
    tokenizer = MarianTokenizer.from_pretrained(repo)
    model = MarianMTModel.from_pretrained(repo)
elif a.family == 'm2m100':
    repo = 'facebook/m2m100_418M'
    tokenizer = M2M100Tokenizer.from_pretrained(repo, src_lang='en')
    model = M2M100ForConditionalGeneration.from_pretrained(repo)
    extra['forced_bos_token_id'] = tokenizer.get_lang_id('ru')
else:
    repo = 'google/madlad400-3b-mt'
    tokenizer = T5Tokenizer.from_pretrained(repo)
    model = T5ForConditionalGeneration.from_pretrained(repo)
    text = '<2ru> ' + text
model = model.to('cpu').eval()
batch = tokenizer(text, return_tensors='pt', truncation=False)
if batch['input_ids'].shape[1] > 384:
    raise SystemExit('INPUT_TOO_LONG: split input; do not silently truncate')
with torch.inference_mode():
    output = model.generate(**batch, **extra,
                            max_new_tokens=384, do_sample=False)
answer = tokenizer.decode(output[0], skip_special_tokens=True)
if not answer.strip():
    raise SystemExit('EMPTY_OUTPUT')
Path(a.output).write_text(answer + '\n', encoding='utf-8')
print(answer)
```

Различия выбора языка следуют официальным примерам: OPUS уже фиксирует EN→RU; M2M100 требует source language и `forced_bos_token_id`; MADLAD использует `<2ru>`. Ограничение 384 токена — наш единый консервативный smoke-test guard, не универсальная характеристика моделей. [web:69][web:68][web:67]

```bash
python scripts/translate_seq2seq.py --family opus \
  --input data/source.en.txt --output results/opus.ru.txt
python scripts/translate_seq2seq.py --family m2m100 \
  --input data/source.en.txt --output results/m2m100.ru.txt
python scripts/translate_seq2seq.py --family madlad \
  --input data/source.en.txt --output results/madlad.ru.txt
```

Для возврата к MLX:

```bash
source .venv-mlx/bin/activate
```

## 10. Память, диск и диагностика

### 10.1. Практические стартовые ограничения

- Только одна тяжёлая модель в памяти; один активный запрос.
- TranslateGemma: короткий исходник, в пределах документированного входного бюджета; не использовать архитектурные 128K как целевой переводческий режим. [web:32][web:70]
- Остальные decoder-модели: начать с 4096–8192 контекста и 768 output tokens. Это настройки инструкции, а не максимумы моделей.
- 27B: начать с готового 4-bit, не конвертировать BF16 самостоятельно без необходимости.
- CPU seq2seq: сначала одно предложение, затем небольшой абзац.
- Не менять системные лимиты GPU-памяти через `sudo sysctl` ради первого запуска.

Фактическая вместимость зависит не только от весов: у runtime есть рабочие буферы, у decoder-моделей — KV cache; macOS и другие приложения также используют unified memory. Запас на диске не заменяет RAM и не гарантирует приемлемую работу через swap. [web:75][web:36]

Проверки:

```bash
memory_pressure
sysctl vm.swapusage
df -h "$HOME"
du -sh ~/.cache/huggingface ~/.ollama "$HOME/local-translation/models" 2>/dev/null
ollama ps
```

Завершать llama-server / MLX server через Ctrl+C в их окне. Для Ollama:

```bash
ollama stop translategemma:12b
```

Не выполнять массовые `killall` и не удалять cache автоматически.

### 10.2. Типовые неисправности

| Симптом | Проверка / действие |
|---|---|
| `command not found` | Проверить Homebrew PATH и активное venv; `which python`, `which llama-server`. |
| 401 / 403 | Проверить точный repo ID, вход HF и принятые условия Google. [web:32] |
| `unknown architecture` | Обновить runtime; проверить, что GGUF именно от нужной модели. Для Index особенно важна поддержка `qwen35`. [web:7] |
| MADLAD GGUF не загружается | Не брать старый Candle файл; использовать выбранную llama.cpp-конверсию или seq2seq. [web:95] |
| Ошибка шаблона TranslateGemma | Передавать структурированный `content`, source/target codes, без system prompt. [web:32] |
| Chimera выдаёт странный «перевод» | Проверить, переданы ли исходник и несколько полных переводов одного сегмента. [web:33] |
| Очень медленно / растёт swap | Остановить другие модели, уменьшить контекст, выбрать квант поменьше, проверить memory pressure. |
| Ответ обрывается | Проверить `finish_reason`, количество output tokens; уменьшить вход или увеличить лимит ответа в разумных пределах. |
| Сервер не запускается: port in use | Остановить предыдущий сервер или выбрать другой порт и изменить URL клиента. |
| Tower использует `[INST]` | Проверить ChatML из tokenizer v0.2; это не обычный Llama 2 chat prompt. [web:60] |

## 11. Контроль результата и воспроизводимость

Цель первого запуска — проверить работоспособность, а не объявить победителя по одному абзацу.

### 11.1. Критерии smoke-test

1. Процесс завершился без ошибки; HTTP-запрос вернул успешный статус.
2. Выход непустой, на русском, без `<think>` и служебных токенов.
3. Нет пропущенных предложений или оборванного окончания.
4. Сохранены UE, номера, отрицание, ссылка `RFC 9110`, `5.3.2`.
5. Смысл требования `shall` / `MUST NOT` не ослаблен.
6. У chat API `finish_reason` не равен `length`; если равен — результат неполный.

Это критерии приёмки пользователя/процесса, а не гарантии любой модели. Нельзя надёжно выбрать лучший технический переводчик только по общим WMT-бенчмаркам. MADLAD прямо описан как general-domain модель без гарантированной пригодности к узким доменам. [web:67]

Для сравнения собрать одинаковый набор 30–50 сегментов: нормы SHALL/MUST, исключения, таблицы, перечисления, определения, сетевые сокращения, ссылки на пункты. Хранить исходник, перевод, ID модели, квант, параметры и замечания эксперта. Для TranslateGemma не добавлять глоссарий как будто это исходный текст; официальный контракт предназначен для текста перевода. Для Index/Qwen отдельный эксперимент с терминологическими ограничениями допустим, но его нельзя смешивать с обычным переводом без отметки. [web:32][web:7]

### 11.2. Зафиксировать окружение

В каждом соответствующем venv:

```bash
python -m pip freeze > logs/requirements-mlx.lock.txt
# В .venv-seq2seq вместо этого:
# python -m pip freeze > logs/requirements-seq2seq.lock.txt
llama-cli --version > logs/llama-version.txt 2>&1
ollama --version > logs/ollama-version.txt 2>&1
sw_vers > logs/macos-version.txt
```

Не писать два разных окружения в один lock-файл. Для MLX snapshot/конверсий дополнительно сохранить точный commit исходного репозитория. Например, запрос метаданных не скачивает все веса:

```bash
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi
repos = [
 'mlx-community/translategemma-4b-it-4bit',
 'mlx-community/translategemma-12b-it-8bit',
 'mlx-community/translategemma-27b-it-4bit',
]
api = HfApi()
manifest = [{'repo_id': r, 'revision': api.model_info(r).sha} for r in repos]
Path('logs/model-revisions.json').write_text(json.dumps(manifest, indent=2))
PY
```

Для строго повторяемого запуска скачать конкретный revision и передавать локальный путь:

```bash
hf download mlx-community/translategemma-4b-it-4bit \
  --revision <COMMIT_SHA_FROM_MANIFEST> \
  --local-dir models/tg4-pinned
# Затем --model models/tg4-pinned в translate_mlx.py
```

Угловые скобки здесь обозначают обязательную подстановку, это не готовая shell-команда. GGUF фиксировать через repo revision, точное имя файла и SHA256 локального файла. Ollama — через `ollama list`, `ollama show` и сохранённый Modelfile/идентификатор артефакта; один только тег со временем может измениться.

Логи и JSON сохраняются локально и могут содержать исходные документы. Не коммитить их автоматически и не отправлять во внешние сервисы без разрешения.

## 12. Правила для ИИ-агента, выполняющего инструкцию

```yaml
run_policy:
  platform: macos-arm64
  workspace: ~/local-translation
  input_language: en
  output_language: ru
  max_parallel_models: 1
  max_parallel_requests: 1
  bind_address: 127.0.0.1
  initial_decoder_context: 4096
  initial_output_tokens: 768
  allow_model_substitution: false
  allow_remote_code: false
  allow_cloud_inference: false
  allow_deleting_models: false
  allow_system_memory_tuning: false
  require_confirmation_before:
    - installing_software
    - accepting_licenses
    - downloading_large_weights
    - changing_shell_profiles
    - expanding_runtime_scope_to_transformers
  stop_on:
    - wrong_cpu_architecture
    - license_or_auth_failure
    - unknown_model_architecture
    - missing_or_incompatible_chat_template
    - empty_output
    - truncated_output
    - sustained_high_memory_pressure
```

Порядок действий агента:

1. Прочитать окружение и свободный диск, ничего не менять.
2. Выбрать одну строку таблицы и точный маршрут.
3. Проверить repo, revision, существование нужного файла/кванта; не угадывать по похожему названию.
4. Показать план установки/скачивания и получить требуемое разрешение.
5. Установить только нужное окружение, выполнить smoke-test.
6. Сохранить вход, выход, версию, настройки, время и статус; токены и серийные номера не сохранять.
7. При несовместимости остановиться и сообщить точную ошибку; fallback выполнять только в пределах согласованного плана.
8. Завершить свой сервер, не останавливая чужие процессы.

Сам факт, что модель можно вызвать через HTTP API, не делает её подходящей tool-calling моделью. TranslateGemma и другие переводчики здесь являются инструментами перевода для внешнего агента; не следует поручать им выполнение shell-команд.

## 13. Рекомендуемый порядок самостоятельного освоения

1. TranslateGemma 4B MLX: проверить окружение и специальный шаблон.
2. TranslateGemma 12B 8-bit: базовый кандидат технического перевода.
3. TranslateGemma 27B 4-bit: проверить, даёт ли дополнительный размер пользу на ваших сегментах.
4. Index 2B/9B: альтернативы для предпросмотра и основного перевода.
5. Hunyuan-MT: независимый кандидат; затем Chimera для синтеза нескольких переводов.
6. Tower и Qwen: проверить задачи с дополнительными инструкциями отдельно от чистого перевода.
7. MADLAD, OPUS, M2M100: добавить как сравнительные baseline, не как заведомо лучшие/худшие модели.

Этот порядок — рекомендация по снижению сложности запуска, не рейтинг качества. Итоговый выбор делать по собственному EN→RU набору стандартов, с экспертной проверкой терминологии и нормативных формулировок.
