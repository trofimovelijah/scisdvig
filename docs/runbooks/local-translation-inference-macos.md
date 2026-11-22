# Локальный инференс переводческих моделей на MacBook Pro M5

- [Локальный инференс переводческих моделей на MacBook Pro M5](#локальный-инференс-переводческих-моделей-на-macbook-pro-m5)
  - [Краткий вывод](#краткий-вывод)
  - [Что выполнить на MacBook с нуля](#что-выполнить-на-macbook-с-нуля)
    - [Проверить среду и свободное место](#проверить-среду-и-свободное-место)
    - [Установить инструменты Apple и Homebrew](#установить-инструменты-apple-и-homebrew)
    - [Создать отдельный локальный стенд](#создать-отдельный-локальный-стенд)
    - [Установить только нужные runtime](#установить-только-нужные-runtime)
    - [Доступ к весам и лицензии](#доступ-к-весам-и-лицензии)
  - [Полная группировка моделей](#полная-группировка-моделей)
    - [Ресурсы MacBook: что известно и что ещё измерить](#ресурсы-macbook-что-известно-и-что-ещё-измерить)
  - [Группа A: TranslateGemma через MLX LM](#группа-a-translategemma-через-mlx-lm)
    - [Важный формат запроса](#важный-формат-запроса)
    - [Скрипт первого перевода](#скрипт-первого-перевода)
    - [Выполнить запуск каждой модели отдельно](#выполнить-запуск-каждой-модели-отдельно)
  - [Группа B: GGUF через llama.cpp](#группа-b-gguf-через-llamacpp)
    - [Запуск сервера: выбрать ОДНУ команду](#запуск-сервера-выбрать-одну-команду)
      - [Index](#index)
      - [Tencent](#tencent)
    - [Единый клиент с модельно-зависимыми шаблонами](#единый-клиент-с-модельно-зависимыми-шаблонами)
    - [Первые запросы](#первые-запросы)
    - [Отдельный сценарий Chimera](#отдельный-сценарий-chimera)
  - [Альтернатива: Ollama](#альтернатива-ollama)
    - [GGUF из HF](#gguf-из-hf)
    - [TranslateGemma в каталоге Ollama](#translategemma-в-каталоге-ollama)
    - [Если модель уже скачана как локальный GGUF](#если-модель-уже-скачана-как-локальный-gguf)
  - [Группа C: encoder-decoder — отдельное исключение](#группа-c-encoder-decoder--отдельное-исключение)
    - [Область применения и текущий статус](#область-применения-и-текущий-статус)
    - [MADLAD: пауза испытаний](#madlad-пауза-испытаний)
    - [Отдельное окружение](#отдельное-окружение)
    - [Скрипт для OPUS и M2M100](#скрипт-для-opus-и-m2m100)
    - [Первые запуски: сначала CPU](#первые-запуски-сначала-cpu)
    - [Проверка результата перед переходом на MPS](#проверка-результата-перед-переходом-на-mps)
    - [Проверка на MPS](#проверка-на-mps)
    - [Правила для coding-агента](#правила-для-coding-агента)
  - [7. Проверка, фиксация версий и локальность](#7-проверка-фиксация-версий-и-локальность)
    - [7.1. Минимальные критерии успешного запуска](#71-минимальные-критерии-успешного-запуска)
    - [7.2. Сохранить фактическую среду](#72-сохранить-фактическую-среду)
    - [7.3. Проверить работу без сети](#73-проверить-работу-без-сети)
    - [7.4. Типовые сбои и корректная реакция](#74-типовые-сбои-и-корректная-реакция)
  - [8. Контракт для ИИ-агента](#8-контракт-для-ии-агента)
  - [9. Итог и открытые решения](#9-итог-и-открытые-решения)

Назначение: ручной запуск перечисленных моделей, первые переводы и воспроизводимая основа для задач coding-агентам.
Статус: исследованная инструкция, НЕ отчёт об испытаниях. Команды не выполнялись на компьютере владельца. Успешная установка, совместимость конкретных версий, скорость и качество подлежат проверке.
Связанные документы: после фактических испытаний связать с существующими требованиями к переводу и ADR выбора runtime; названия и наличие таких документов здесь не предполагаются. Новые ADR пока не создаются.

## Краткий вывод

1. Для трёх `TranslateGemma` предлагается MLX LM с готовыми MLX-весами.
2. Для `Index 2B/9B`, `Hunyuan MT/Chimera`, `Tower` и `Qwen` предлагается llama.cpp + GGUF. Ollama — альтернативная оболочка запуска, а не обязательный дополнительный этап.
3. `MADLAD`, `OPUS-MT` и `M2M100` выделены отдельно: encoder-decoder модели нельзя автоматически приравнивать к chat LLM. Для них приведён отдельный путь Transformers/PyTorch, выходящий за заданные три runtime. Если список runtime является строгим ограничением, эти три модели остаются без принятого пути запуска, а не объявляются совместимыми без проверки.
4. На 64 GB разумно начинать с одной загруженной модели, короткого текста и ограниченного контекста. Для TranslateGemma 27B выбран 4-bit вариант. Это предложение по ресурсам, а не результат замера.
5. Примеры используют `EN→RU` как демонстрационное направление. Это не фиксация всех языковых пар проекта.

Фактическая основа группировки: готовые MLX-конверсии TranslateGemma, официальные GGUF Index, GGUF-конверсии остальных decoder-only моделей и архитектуры T5/Marian/M2M100. 

## Что выполнить на MacBook с нуля

1. Выполнять раздел по шагам, не копировать весь документ одной командой. 
2. Установка ПО и скачивание весов меняют локальную систему.
3. Агент не должен выполнять их без разрешения владельца. 
4. Интернет нужен для установки и загрузки артефактов, но не для последующей генерации по локальным весам.

### Проверить среду и свободное место

В Terminal:

```bash
uname -m
sw_vers
system_profiler SPHardwareDataType SPDisplaysDataType
sysctl -n hw.memsize
df -h "$HOME"
xcode-select -p
```

Ожидаются следующие значения: 
   - `arm64`,
   - фактическая конфигурация `M5` и 64 GB памяти,
   - не запускать Terminal/Python через `Rosetta`,
   - надпись `Pro` в названии ноутбука не определяет вариант чипа: M5, M5 Pro и M5 Max не следует смешивать в результатах замеров.

Для использования MLX-оптимизаций Neural Accelerators в GPU M5 Apple указывает macOS 26.2 или новее. Это требование именно к этому ускоренному пути, а не утверждение, что любой MLX вообще невозможно использовать на более ранней системе. Не нужно устанавливать beta-ОС ради этой инструкции.

Предлагаемый резерв SSD:
   - не менее `100 GB` для первых нескольких квантованных моделей и окружений,
   - около `200 GB` для последовательного исследования всего списка без скачивания всех форматов. 

Это организационный бюджет, не сумма точно проверенных файлов. Перед каждой загрузкой смотреть её фактический объём. В отдельных GGUF-репозиториях лежат десятки квантов: никогда не скачивать весь репозиторий без фильтра. Например, каталог Unsloth Qwen в совокупности занимает около 507 GB, а один `Q4_K_M` — около 18.6 GB.

### Установить инструменты Apple и Homebrew

Если Command Line Tools отсутствуют:

```bash
xcode-select --install
```

Дождаться окончания системной установки, затем повторить `xcode-select -p`. Apple включает Command Line Tools в требования к PyTorch/MPS.

Если Homebrew ещё нет, сначала просмотреть установщик, затем выполнить его:

```bash
curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh \
  -o /tmp/homebrew-install.sh
less /tmp/homebrew-install.sh
/bin/bash /tmp/homebrew-install.sh
```

Для стандартной arm64-установки подключить Homebrew к текущей оболочке:

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
brew --version
brew install python@3.12 git cmake
```

Постоянную настройку PATH сделать по командам `Next steps`, которые выдаст установщик; не добавлять одинаковые строки повторно. Стандартный префикс Homebrew на Apple Silicon — `/opt/homebrew`.

### Создать отдельный локальный стенд

Эта инструкция не изменяет существующий Git-репозиторий.

```bash
mkdir -p "$HOME/translation-lab"/{scripts,data,results,logs,models,env}
cd "$HOME/translation-lab"
cat > env/local.sh <<'EOF'
export HF_HOME="$HOME/translation-lab/models/hf-cache"
export TOKENIZERS_PARALLELISM=false
EOF
source env/local.sh
cat > data/source.en.txt <<'EOF'
The UE shall retain the session identifier for 30 seconds.
If the request is rejected, the server MUST NOT modify the configuration.
See RFC 2119 and 3GPP TS 23.501, clause 5.6.
EOF
```

Важно выполнять следующие аспекты:
   - `env/local.sh` подключать в каждом новом терминале,
   - `HF_HOME` содержит и кэш весов, и служебные данные HF: не добавлять его содержимое в Git,
   - токены не записывать в скрипты, Markdown, логи или аргументы команд.

### Установить только нужные runtime

Для MLX и лёгких клиентов GGUF:

```bash
cd "$HOME/translation-lab"
source env/local.sh
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade mlx-lm transformers huggingface_hub sentencepiece
python -c 'import platform, mlx.core as mx; print(platform.machine()); print(mx.default_device())'
```

MLX LM предоставляет Python API `load`/`generate` и работает с Apple Silicon. Наличие MLX-весов не означает совместимость произвольной архитектуры.

Для llama.cpp, независимо от Python-окружения:

```bash
brew install llama.cpp
llama-server --version
llama-server --help > logs/llama-server-help.txt
```

На macOS llama.cpp поддерживает GPU через Metal – при сборке из исходников Metal включён по-умолчанию. В первых запусках ниже задаётся `-ngl 99`, но фактический GPU-offload обязательно проверять в логе.

Если установленная версия не понимает `qwen35`, `Hunyuan` или флаги из инструкции: 
 - сначала сохранить ошибку и версию, 
 - затем обновить пакет с разрешения владельца. 

Сборка свежего исходного кода — отдельный диагностический шаг, не автоматическая смена среды:

```bash
cd "$HOME/translation-lab"
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build --config Release -j 4 --target llama-server llama-cli
./build/bin/llama-server --version
git rev-parse HEAD
```

После сборки использовать полный путь к выбранному бинарнику; не смешивать его с Homebrew-бинарником в одном протоколе испытаний. Команды CMake и Metal основаны на документации llama.cpp. 

Для Ollama — альтернативный, а не обязательный путь:

```bash
brew install ollama
ollama --version
```

Не запускать одновременно приложение Ollama, фоновую службу и `ollama serve` из терминала. В этой инструкции применяется только ручной сервер:

```bash
OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 \
ollama serve
```

Сервер оставить в отдельном терминале. `OLLAMA_NO_CLOUD` отключает облачные функции, но не загрузку локальных моделей из интернета. В macOS-приложении переменные задаются иначе — через `launchctl` с перезапуском приложения. Используется нативный запуск: Docker Desktop на macOS не даёт Ollama обычный GPU passthrough.

### Доступ к весам и лицензии

В HF при необходимости принять условия исходной Gemma-модели под своей учётной записью. Затем в окружении:

```bash
hf auth login
```

Токен вводить интерактивно. Для публичных ungated-конверсий вход может не понадобиться; 401/403 не обходить сменой прав или неизвестными зеркалами. Google явно требует согласия с условиями доступа к своим Gemma-весам.

До применения вне личного эксперимента проверить лицензию исходной модели и конкретной конверсии. У `TowerInstruct v0.2` указаны CC-BY-NC-4.0 и условия базовой `Llama 2`. Это существенное ограничение для коммерческого проекта – конверсия в GGUF не отменяет исходную лицензию.

## Полная группировка моделей

Обозначения: 

   - «основной» — предлагаемый путь этой инструкции; 
   - «альтернатива» — отдельный путь с собственными весами и проверкой,
   - «Не принят» не означает математическую невозможность портирования – это отсутствие проверенного здесь рецепта для заданного runtime.

| Модель | Точный исходный ID / выбранный артефакт | Основной путь | Альтернатива и особенности |
|---|---|---|---|
| translategemma-12b-it-8bit | `mlx-community/translategemma-12b-it-8bit`, исходная `google/translategemma-12b-it` | MLX, §3 | Ollama TranslateGemma, §5; другой формат и потенциально другой квант |
| translategemma-4b-it-4bit | `mlx-community/translategemma-4b-it-4bit` | MLX, §3 | Ollama `translategemma:4b` не является побитно теми же MLX-весами |
| Index-Translate-2B | `IndexTeam/Index-Translate-2B-GGUF` | llama.cpp Q4_K_M, §4 | Ollama HF GGUF; использовать thinking=false |
| Index-Translate-9B | `IndexTeam/Index-Translate-9B-GGUF` | llama.cpp Q4_K_M, §4 | Официальная GGUF-конверсия; для сравнения можно отдельно Q8_0 |
| Hunyuan-MT-7B | исходная `tencent/Hunyuan-MT-7B`; GGUF `olegshulyakov/Hunyuan-MT-7B-GGUF` | llama.cpp Q6_K, §4 | Сторонняя конверсия, также документирован запуск через Ollama |
| Hunyuan-MT-Chimera-7B | исходная `tencent/Hunyuan-MT-Chimera-7B`; GGUF `DevQuasar/tencent.Hunyuan-MT-Chimera-7B-GGUF` | llama.cpp Q4_K_M, §4 | Это объединение готовых переводов, а не обычный перевод одним входным текстом |
| MADLAD400-3B | `google/madlad400-3b-mt` | Исключение: Transformers/T5, §6 | Не включён в принятые рецепты MLX LM / llama.cpp / Ollama. В карточке есть GGUF для Candle: сам формат GGUF не доказывает работу в llama.cpp |
| OPUS-MT EN→RU | `Helsinki-NLP/opus-mt-en-ru` | Исключение: Transformers/Marian, §6 | Только EN→RU для этого ID; RU→EN требует другой модели |
| M2M100-418M | `facebook/m2m100_418M` | Исключение: Transformers/M2M100, §6 | Нужны src_lang и forced_bos_token_id; не chat template |
| translateGemma-27B-it | исходная `google/translategemma-27b-it`; выбранная `mlx-community/translategemma-27b-it-4bit` | MLX 4-bit, §3 | Разрядность не была задана: 4-bit предлагается как начальный вариант; Ollama 27b — альтернатива |
| TowerInstruct-7B-v0.2 | исходная `Unbabel/TowerInstruct-7B-v0.2`; GGUF `mradermacher/TowerInstruct-7B-v0.2-GGUF` | llama.cpp Q4_K_M, §4 | ChatML без system prompt; не подменять v0.2 на v0.1 |
| Qwen3-30B-A3B-Instruct-2507 | исходная `Qwen/Qwen3-30B-A3B-Instruct-2507`; GGUF `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | llama.cpp Q4_K_M, §4 | Возможны MLX-конверсии; здесь для уменьшения числа рецептов выбран GGUF. Не заменять модель на обычную `qwen3:30b`. |

Терминология инструментария:
   - MLX — вычислительный фреймворк, 
   - MLX LM — библиотека запуска LLM поверх MLX, 
   - llama.cpp — движок,
   - Ollama — средство управления моделями и API. 

В инструкции Ollama используется с GGUF; это не утверждение, что все версии Ollama всегда ограничены единственным внутренним backend. Сравнивать надо конкретные версии и конкретные артефакты, а не только названия runtime.

### Ресурсы MacBook: что известно и что ещё измерить

| Вариант | Ориентир объёма весов | Оценка для 64 GB |
|---|---|---|
| TranslateGemma 4B MLX 4-bit | Репозиторий около 2.22 GB | Начальный smoke-test |
| TranslateGemma 12B MLX 8-bit | Порядок 12–15 GB, инженерная оценка, не измеренный RSS | Кандидат на основной переводчик; качество на стандартах ещё не доказано |
| TranslateGemma 27B MLX 4-bit | Порядок 14–18 GB, инженерная оценка, не измеренный RSS | Ожидается размещение одной модели при коротком контексте; проверить пик памяти |
| Index 2B / 9B Q4_K_M | Около 1.31 / 5.78 GB | Файлы значительно меньше 64 GB; остаются кэши и рабочие буферы |
| Hunyuan MT Q6_K | Смотри фактический скачанный файл | Не экстраполировать память из имени 7B |
| Chimera Q4_K_M | Около 4.62 GB | Дополнительно нужен контекст с несколькими кандидатами |
| Tower Q4_K_M | Около 4.2 GB у выбранного конвертера | Начинать с коротких сегментов |
| Qwen Q4_K_M | Около 18.6 GB | Возможен как одиночный процесс при ограниченном контексте |
| MADLAD FP32 | Около 12 GB только параметры, оценка по 3B × 4 байта | Начать на CPU, затем отдельно проверить MPS |

Размер файла не равен памяти процесса. Общая память занята также macOS, GPU-буферами, активациями и KV-кэшем. Из 64 GB не следует доступность всех 64 GB для весов.

У Qwen 30.5B полных и 3.3B активных параметров: MoE сокращает вычисления на токен, но не превращает объём всех весов в объём 3B-модели. Не начинать с BF16 и максимального заявленного контекста.

Политика стенда, предлагаемая для начала: одна модель, один запрос, без изменения системных лимитов памяти; остановка при предупреждении OOM или заметном росте swap и Memory Pressure. Скорость в tok/s и время книги заранее не обещаются.

## Группа A: TranslateGemma через MLX LM

### Важный формат запроса

Официальный шаблон TranslateGemma требует `content` как список ровно из одного элемента, с `type`, `source_lang_code`, `target_lang_code` и `text`. Обычный пример чата с `content: "Hello"` на автосгенерированной странице не является надёжным переводческим примером. System role здесь не нужен. В карточке Google указано 2K входного контекста.

Существует расхождение: Ollama показывает 128K окна в метаданных семейства, Google описывает вход 2K. В этом стенде применяем консервативный бюджет: `prompt + резерв ответа ≤ 2048 токенов`, а не считаем `128K` подтверждённым режимом качественного перевода.

### Скрипт первого перевода

Создать `scripts/translate_mlx.py` со следующим содержимым:

```python
import argparse
import json
import sys
import time
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
p.add_argument('--source', default='en')
p.add_argument('--target', default='ru')
p.add_argument('--max-tokens', type=int, default=768)
a = p.parse_args()
text = Path(a.input).read_text(encoding='utf-8').strip()
if not text or not 1 <= a.max_tokens <= 1024:
    raise SystemExit('Empty input or invalid output budget')
t0 = time.perf_counter()
model, tokenizer = load(a.model)
messages = [{'role': 'user', 'content': [{
    'type': 'text', 'source_lang_code': a.source,
    'target_lang_code': a.target, 'text': text,
}]}]
tokens = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True
)
if len(tokens) + a.max_tokens > 2048:
    raise SystemExit('Conservative 2048-token budget exceeded; split the input')
t1 = time.perf_counter()
answer = generate(model, tokenizer, prompt=tokens,
                  max_tokens=a.max_tokens, sampler=make_sampler(temp=0.0),
                  verbose=False)
if not answer.strip():
    raise SystemExit('Empty generation')
Path(a.output).write_text(answer + '\n', encoding='utf-8')
print(json.dumps({'model': a.model, 'prompt_tokens': len(tokens),
                  'load_and_prompt_s': t1-t0,
                  'generation_s': time.perf_counter()-t1,
                  'output_budget': a.max_tokens,
                  'note': 'Check output for truncation; this API returns text'},
                 ensure_ascii=False), file=sys.stderr)
```

API загрузки/генерации и sampling основаны на MLX LM; структура сообщения — на карточке Google. Это составленный для стенда скрипт, а не проверенный на данном Mac пример.

### Выполнить запуск каждой модели отдельно

```bash
cd "$HOME/translation-lab"
source env/local.sh
source .venv-mlx/bin/activate
script -q -a logs/tg4.log /usr/bin/time -l python -u scripts/translate_mlx.py \
  --model mlx-community/translategemma-4b-it-4bit \
  --input data/source.en.txt --output results/tg4.ru.txt
cat results/tg4.ru.txt
```

Следующий запуск — только после завершения предыдущего процесса:

```bash
script -q -a logs/tg12.log /usr/bin/time -l python -u scripts/translate_mlx.py \
  --model mlx-community/translategemma-12b-it-8bit \
  --input data/source.en.txt --output results/tg12.ru.txt
```

```bash
script -q -a logs/tg27.log /usr/bin/time -l python -u scripts/translate_mlx.py \
  --model mlx-community/translategemma-27b-it-4bit \
  --input data/source.en.txt --output results/tg27.ru.txt
```

> [!NOTE]
> 
> Если при первом запросе возвращается сообщение "`warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.`", значит, что некоторые запросы к HF идут без токена, поэтому действуют лимиты на скачивание. Чтобы этого избежать, необходимо выполнить команду в терминале:

```bash
source env/local.sh
source .venv-mlx/bin/activate
unset HF_HUB_DISABLE_IMPLICIT_TOKEN

# Передать Python-процессу уже сохранённый токен, не выводя его:
export HF_TOKEN="$(python -c 'from huggingface_hub import get_token; print(get_token() or "", end="")')"

if [ -n "$HF_TOKEN" ]; then
  echo "Токен найден"
else
  echo "Токен не найден в текущем окружении"
fi

# если токен не найден, выполнить
hf auth login --force
```

Первый запуск включает скачивание весов (для оценки времени загрузки и генерации повторить после прогрева дискового кэша). Завершение отдельного Python-процесса освобождает принадлежащие ему модель и кэши. Пока не держать три модели в одном процессе.

Если шаблон выдаёт ошибку структуры сообщения, не заменять его произвольной строкой: 
   - сохранить `tokenizer_config.json`/шаблон, версию MLX LM и traceback,
   - отдельно проверить точный ID и происхождение конверсии.

Эти `MLX LM` рецепты — только text→text. Наличие у исходной TranslateGemma image-input не доказывает наличие vision-компонентов в конкретной MLX LM конверсии. Обработку PDF, OCR и сохранение вёрстки здесь не проверяем.

## Группа B: GGUF через llama.cpp

### Запуск сервера: выбрать ОДНУ команду

Краткая аннотация:
  - команды загружают выбранный квант, а не весь набор квантов,
  - запуск `llama-server` выполняется в первом терминале,
  - запрос выполняется из второго терминала, 
  - `--no-mmproj` отключает автоматическую загрузку vision-projector – стенд проверяет только текстовый перевод,
  - везде слушаем только loopback, порт 8080,
  - при смене модели необходимо остановить сервер через Ctrl+C.

#### Index

Официальные карточки Index приводят эти GGUF-репозитории, запуск llama.cpp, greedy decoding и `enable_thinking:false`. Для каждой из двух моделей указаны консервативные настройки и значения контекста и параллелизма.

Index 2B:

```bash
script -q logs/index2-server.log llama-server -hf IndexTeam/Index-Translate-2B-GGUF:Q4_K_M \
  -ngl 99 --no-mmproj -c 4096 --parallel 1 --host 127.0.0.1 --port 8080 \
  --alias index2 --jinja --chat-template-kwargs '{"enable_thinking":false}'
```

Index 9B:

```bash
script -q logs/index9-server.log llama-server -hf IndexTeam/Index-Translate-9B-GGUF:Q4_K_M \
  -ngl 99 --no-mmproj -c 4096 --parallel 1 --host 127.0.0.1 --port 8080 \
  --alias index9 --jinja --chat-template-kwargs '{"enable_thinking":false}'
```

#### Tencent

Сторонние GGUF-конверсии соответствующих Tencent-моделей. Их наличие подтверждено. Фактическое качество кванта и работа данной сборки llama.cpp на M5 необходимо проверить. 

Hunyuan MT:

```bash
hf download mradermacher/Hunyuan-MT-7B-GGUF \
  Hunyuan-MT-7B.Q6_K.gguf --local-dir models/hunyuan && \
script -q logs/hunyuan-server.log llama-server -m models/hunyuan/Hunyuan-MT-7B.Q6_K.gguf \
  -ngl 99 --no-mmproj -c 4096 --parallel 1 --host 127.0.0.1 --port 8080 --cors-origins localhost \
  --alias hunyuan
```

Chimera:

```bash
script -q logs/chimera-server.log llama-server -hf DevQuasar/tencent.Hunyuan-MT-Chimera-7B-GGUF:Q4_K_M \
  -ngl 99 --no-mmproj -c 8192 --parallel 1 --host 127.0.0.1 --port 8080 \
  --alias chimera
```


Tower:

```bash
script -q logs/tower-server.log llama-server -hf mradermacher/TowerInstruct-7B-v0.2-GGUF:Q4_K_M \
  -ngl 99 --no-mmproj -c 2048 --parallel 1 --host 127.0.0.1 --port 8080 \
  --alias tower
```

Квант опубликован; в оригинальной карточке Tower обучение ограничено `max_seq_length:2048`, используется ChatML. Потому первый тест ограничен 2048, независимо от того, какое большее окно может разрешить конфигурация. В карточке одновременно отмечено улучшение document-level поведения и сохранено предупреждение против использования как готового document-level переводчика: целые стандарты этим не считаются проверенными.

Qwen:

```bash
script -q logs/qwen-server.log llama-server -hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M \
  -ngl 99 --no-mmproj -c 4096 --parallel 1 --host 127.0.0.1 --port 8080 \
  --alias qwen2507
```

В выбранном репозитории есть ровно этот Q4_K_M; исходная модель — non-thinking Instruct-2507. Это универсальная instruct-модель, а не специализированная MT-модель. В качестве переводчика работает, в качестве верификатора требуется проверить.

Проверить готовность:

```bash
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8080/v1/models
```

### Единый клиент с модельно-зависимыми шаблонами

Чтобы не полагаться на автоматически выбранный `chat template` стороннего GGUF, клиент ниже берёт токенизатор исходной модели и передаёт готовые token IDs в нативный `/completion` llama.cpp. Шаблон применяется ровно один раз. Нативный API и параметры сервера описаны в документации llama.cpp.

Исходный токенизатор должен соответствовать именно выбранному GGUF. Проверка соответствия — обязательная часть первого испытания. Не использовать этот клиент с произвольным другим файлом модели.

Создать `scripts/translate_gguf.py`:

```python
import argparse
import json
import sys
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer

PROFILES = {
    'index2': ('IndexTeam/Index-Translate-2B', 4096),
    'index9': ('IndexTeam/Index-Translate-9B', 4096),
    'hunyuan': ('tencent/Hunyuan-MT-7B', 4096),
    'chimera': ('tencent/Hunyuan-MT-Chimera-7B', 8192),
    'tower': ('Unbabel/TowerInstruct-7B-v0.2', 2048),
    'qwen2507': ('Qwen/Qwen3-30B-A3B-Instruct-2507', 4096),
}
p = argparse.ArgumentParser()
p.add_argument('--profile', choices=PROFILES, required=True)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
p.add_argument('--candidates')
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
model_id, context = PROFILES[a.profile]
with urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=10) as response:
    model_list = json.load(response)
if a.profile not in {item.get('id') for item in model_list.get('data', [])}:
    raise SystemExit('Server alias does not match requested profile')
text = Path(a.input).read_text(encoding='utf-8').strip()
if not text:
    raise SystemExit('Empty source')
if a.profile.startswith('index'):
    content = '请将以下文本翻译为俄语，直接输出翻译结果，不要进行任何解释。\n\n' + text
elif a.profile == 'hunyuan':
    content = 'Translate the following segment into Russian, without additional explanation.\n\n' + text
elif a.profile == 'chimera':
    if not a.candidates:
        raise SystemExit('--candidates is required for Chimera')
    candidates = json.loads(Path(a.candidates).read_text(encoding='utf-8'))
    if not isinstance(candidates, list) or len(candidates) != 6 or not all(
        isinstance(x, str) and x.strip() for x in candidates
    ):
        raise SystemExit('Expected JSON array of six non-empty translations')
    content = (
        'Analyze the following multiple Russian translations of the English segment '
        'surrounded in triple backticks and generate a single refined Russian translation. '
        'Only output the refined translation, do not explain.\n\n'
        'The English segment:\n```' + text + '```\n\n'
        'The multiple Russian translations:\n' + '\n'.join(
            f'{i}. ```{candidate}```' for i, candidate in enumerate(candidates, 1)
        )
    )
elif a.profile == 'tower':
    content = 'Translate the following text from English into Russian.\nEnglish: ' + text + '\nRussian:'
else:
    content = ('Translate the following English technical text into Russian. '
               'Output only the translation. Preserve numbers, identifiers, references '
               'and placeholders exactly.\n\n' + text)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
kwargs = {'enable_thinking': False} if a.profile.startswith('index') else {}
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': content}], tokenize=True,
    add_generation_prompt=a.profile not in ('hunyuan', 'chimera'), **kwargs
)
if not isinstance(prompt, list) or not all(isinstance(x, int) for x in prompt):
    raise SystemExit('Expected flat token ID list')
limit = 768
if len(prompt) + limit > context:
    raise SystemExit('Context budget exceeded; split source/candidates')
body = {'prompt': prompt, 'n_predict': limit, 'stream': False,
        'temperature': 0.0, 'seed': a.seed, 'cache_prompt': False}
if a.profile in ('hunyuan', 'chimera'):
    body.update(temperature=0.7, top_p=0.6, top_k=20, min_p=0.0, repeat_penalty=1.05)
request = urllib.request.Request(
    'http://127.0.0.1:8080/completion',
    data=json.dumps(body).encode('utf-8'),
    headers={'Content-Type': 'application/json'}, method='POST'
)
with urllib.request.urlopen(request, timeout=1800) as response:
    result = json.load(response)
Path(a.output + '.json').write_text(
    json.dumps({'source_model': model_id, 'profile': a.profile,
                'prompt_tokens': len(prompt), 'response': result},
               ensure_ascii=False, indent=2), encoding='utf-8'
)
answer = result.get('content', '')
if not answer.strip():
    raise SystemExit('Empty output; inspect response JSON')
Path(a.output).write_text(answer + '\n', encoding='utf-8')
if result.get('stopped_limit') or result.get('tokens_predicted', 0) >= limit:
    raise SystemExit('Potentially truncated output; result saved for inspection')
print(answer)
```

Форматы запросов: 
  - Index — официальный краткий шаблон,
  - Hunyuan и Chimera — шаблоны Tencent, включая `add_generation_prompt=False`,
  - Tower — карточка Unbabel,
  - для Qwen сформулирован тестовый переводческий запрос. 
  
Sampling Hunyuan взят из карточки Tencent. Для остальных выбран `temperature=0` как стартовый режим сравнения – это не универсальная рекомендация авторов Qwen, которые приводят `temperature=0.7`.

Перед первым запуском установить профиль в соответствие серверу. Клиент проверяет alias сервера, но alias — имя, заданное оператором, а не криптографическая проверка весов. В частности, нельзя запустить сервер Index 2B и отправить токены Index 9B. Идентификатор alias из `/v1/models`, команда запуска, лог архитектуры и ID токенизатора должны согласоваться. При расхождении остановиться.

### Первые запросы

Во втором терминале для каждой из моделей выполнить проверку:

```bash
cd "$HOME/translation-lab"
source env/local.sh
source .venv-mlx/bin/activate
python scripts/translate_gguf.py --profile index2 \
  --input data/source.en.txt --output results/index2.ru.txt
```

После остановки сервера, выбора следующей модели и новой проверки `/health`:

```bash
python scripts/translate_gguf.py --profile index9 \
  --input data/source.en.txt --output results/index9.ru.txt

python scripts/translate_gguf.py --profile hunyuan \
  --input data/source.en.txt --output results/hunyuan.ru.txt

python scripts/translate_gguf.py --profile tower \
  --input data/source.en.txt --output results/tower.ru.txt

python scripts/translate_gguf.py --profile qwen2507 \
  --input data/source.en.txt --output results/qwen2507.ru.txt
```

> [!Note]
> 
> Для каждого профиля весов проверка срабатывает успешно.

### Отдельный сценарий Chimera

Назначение Chimera — объединять несколько переводов. В официальном примере шесть кандидатов; это не доказательство, что любое другое число невозможно, но для первого воспроизводимого теста оставляем шесть. Увеличение качества не гарантируется.

1. Запустить Hunyuan MT сервер.
2. Выполнить шесть запросов с разными seed:
```bash
(
set -e
export RUN_DIR
RUN_DIR=$(mktemp -d "$PWD/results/chimera-XXXXXX")
for seed in 41 42 43 44 45 46; do
  python scripts/translate_gguf.py --profile hunyuan --seed "$seed" \
    --input data/source.en.txt --output "$RUN_DIR/candidate-$seed.txt"
done
python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['RUN_DIR'])
values = [root.joinpath(f'candidate-{s}.txt').read_text(encoding='utf-8').strip()
          for s in range(41, 47)]
assert all(values), 'Missing/empty candidate'
Path('data/candidates.ru.json').write_text(
    json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
print('Unique candidates:', len(set(values)))
PY
)
```

> [!Note]
> Выполнена проверка на отсутствие ошибок и обрезаний во всех шести JSON-ответах; подоболочка останавливается при ошибке, а кандидаты каждого прогона создаются в новом каталоге.

3. Остановить Hunyuan сервер; запустить Chimera сервер.
4. Выполнить:

```bash
python scripts/translate_gguf.py --profile chimera \
  --input data/source.en.txt --candidates data/candidates.ru.json \
  --output results/chimera.ru.txt
```

Получается шесть генераций кандидатов плюс одна генерация объединения. Сравнивать итог нужно с лучшим кандидатом, учитывая суммарную задержку. Последовательный запуск не требует держать обе модели в памяти одновременно.

## Альтернатива: Ollama

### GGUF из HF

При запущенном локальном Ollama-сервере можно выбрать один артефакт:

```bash
ollama pull hf.co/IndexTeam/Index-Translate-2B-GGUF:Q4_K_M
ollama run hf.co/IndexTeam/Index-Translate-2B-GGUF:Q4_K_M
```

Остальные GGUF-идентификаторы использовать аналогично только после проверки поддержки архитектуры установленной версией Ollama. Для Index и Hunyuan соответствующие HF/Ollama-команды документированы; это не заменяет проверку шаблона. 

Для автоматизации важнее API, чем интерактивный чат. Ниже пример для Index; после ответа сохранить полный JSON и проверить отсутствие thinking/служебного вывода:

```bash
curl --fail-with-body http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"hf.co/IndexTeam/Index-Translate-2B-GGUF:Q4_K_M",
    "messages":[{"role":"user","content":"请将以下文本翻译为俄语，直接输出翻译结果，不要进行任何解释。\n\nThe UE shall retain the session identifier for 30 seconds."}],
    "think":false,
    "stream":false,
    "keep_alive":0,
    "options":{"temperature":0,"num_ctx":4096,"num_predict":768}
  }' > results/ollama-index2.json
ollama ps
```

Ollama-путь для Index нужно считать принятым.

В API Ollama `keep_alive:0` выгружает модель после запроса; `num_ctx` задаёт контекст; `ollama ps` показывает размещение модели. 

### TranslateGemma в каталоге Ollama

Выполнять при включённом `vpn`, иначе ошибка `TLS handshake timeout`:
```bash
ollama pull translategemma:4b
ollama show translategemma:4b
```

В семействе есть `4b`, `12b`, `27b`. Эти теги не фиксируют исходные пользовательские MLX 4-bit/8-bit артефакты; перед сравнением проверить quantization и digest. На странице Ollama ориентиры файлов: 4b — 3.3 GB, 12b — 8.1 GB, 27b — 17 GB.

Запрос по опубликованному Ollama Prompt Guide для `4b` выполняется успешно:

```bash
curl --fail-with-body http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"translategemma:4b",
    "stream":false,
    "keep_alive":0,
    "messages":[{"role":"user","content":"You are a professional English (en) to Russian (ru) translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Russian grammar, vocabulary, and cultural sensitivities.\nProduce only the Russian translation, without any additional explanations or commentary. Please translate the following English text into Russian:\n\nThe UE shall retain the session identifier for 30 seconds."}],
    "options":{"temperature":0,"num_ctx":2048,"num_predict":768}
  }' > results/ollama-tg4.json
```
```json
{
  "model":"translategemma:4b",
  "created_at":"2026-11-22T16:50:08.093981Z",
  "message":{
    "role":"assistant",
    "content":"UE должен хранить идентификатор сессии в течение 30 секунд."
  },
  "done":true,
  "done_reason":"stop",
  "total_duration":1087323250,
  "load_duration":843183292,
  "prompt_eval_count":89,
  "prompt_eval_cached_count":0,
  "prompt_eval_duration":68583000,
  "eval_count":16,
  "eval_duration":172760000
}
```

### Если модель уже скачана как локальный GGUF

Можно создать локальный Modelfile с `FROM /absolute/path/model.gguf`, затем `ollama create`. Но архитектура и шаблон всё равно должны поддерживаться. Не импортировать MLX-квантизованные safetensors как будто это эквивалентный GGUF: формат, схема квантования и поддержка runtime различаются. Официальный путь импорта GGUF описан Ollama.

Этот способ не является обязательным шагом инструкции: импорт и отдельные кэши могут занимать дополнительный SSD. Не скачивать одну модель во всех трёх форматах без задачи сравнения.

## Группа C: encoder-decoder — отдельное исключение

### Область применения и текущий статус

Этот раздел описывает дополнительный путь через Transformers/PyTorch,
не входящий в основной набор runtime: MLX, llama.cpp и Ollama.

OPUS использует Marian, M2M100 — свою encoder-decoder архитектуру.
Их интерфейс — подготовка входа и seq2seq `generate`, а не сообщения chat LLM.

MADLAD400-3B использует T5, но его испытания поставлены на паузу.
Модель не входит в активную последовательность запуска этого раздела.

> [!NOTE]
> Если допустимы только MLX, llama.cpp и Ollama, остановиться здесь.
> Для OPUS и M2M100 отметить
> `blocked: runtime outside approved set`.
> Для MADLAD сохранить статус `paused`.
>
> Следующие шаги требуют отдельного согласования Transformers/PyTorch.
> Наличие инструкции не означает согласование дополнительного runtime.

Результаты испытаний на 2026-11-22:

| Модель | Статус | Наблюдение |
|---|---|---|
| OPUS-MT EN→RU | Проверка результата не завершена | В изученном MPS-логе нет сообщений об ошибках. Полнота и смысл перевода отдельно не подтверждены |
| M2M100-418M | Проверка полноты не пройдена | На CPU и MPS пропущено первое предложение тестового исходника — требование о сохранении идентификатора сессии на 30 секунд |
| MADLAD400-3B | `paused` | На CPU и MPS получены английские повторы вместо русского перевода; генерация достигает лимита без EOS |

Успешное выполнение процесса не означает успешного перевода.
Наличие выходного файла также не является достаточным критерием.

### MADLAD: пауза испытаний

Объект паузы: `google/madlad400-3b-mt` в текущем локальном
Transformers/PyTorch-рецепте.

Основания:

- На CPU и MPS получена некорректная генерация:
  английский текст вместо русского перевода, повторы, пропуски и искажение смысла.
- Генерация достигала лимита 512 новых токенов без EOS.
- Fast- и Slow-токенизаторы на тестовом исходнике дали одинаковые ID;
  исходный текст и префикс `<2ru>` сохранялись.
- В испытании с отключённым кэшем наблюдаемый результат не улучшился.
- Корректно выполненный CPU-прогон с `eager` attention также не исправил результат.
- Первопричина не установлена. Не доказаны ни дефект самой модели,
  ни повреждение весов, ни конкретная несовместимость библиотек.

Промежуточный диагностический запуск, в котором MADLAD ошибочно
загружался через `M2M100ForConditionalGeneration`, не является
валидным испытанием MADLAD или `eager` attention.

Зафиксированное окружение:

| Компонент | Версия |
|---|---|
| Python | 3.12.16 |
| torch | 2.15.0 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| sentencepiece | 0.2.3 |
| huggingface-hub | 0.36.2 |
| safetensors | 0.8.0 |

Эти версии описывают исследованное окружение, а не рекомендуемый
или подтверждённо совместимый lockfile.

Правила на время паузы:

- Не запускать MADLAD в рамках штатного smoke-test.
- Не включать полученные результаты в сравнение качества или скорости
  как валидные переводы.
- Не продолжать подбор параметров и переустановку зависимостей
  без отдельной задачи владельца проекта.
- Сохранить имеющиеся логи, результаты и диагностические скрипты.
- Не удалять веса и окружение без отдельного решения.

Возобновление — только по отдельному решению владельца проекта.
Первая задача при возобновлении: минимальный воспроизводимый пример
с фиксацией revision модели, зависимостей, сведений загрузки весов
и полного результата генерации.

### Отдельное окружение

Для нового окружения:

```bash
cd "$HOME/translation-lab"
source env/local.sh

"$(brew --prefix python@3.12)/bin/python3.12" \
  -m venv .venv-seq2seq

source .venv-seq2seq/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  'torch>=2.6' \
  'transformers>=4.56,<5' \
  sentencepiece sacremoses huggingface_hub protobuf

python -m pip check

python -c \
  'import torch; print(torch.__version__); print("MPS:", torch.backends.mps.is_available())'
```

Если окружение уже существует, для обычного запуска достаточно:

```bash
cd "$HOME/translation-lab"
source env/local.sh
source .venv-seq2seq/bin/activate
```

Не переустанавливать зависимости существующего окружения
только ради повторного прогона.

Указанные диапазоны — стартовая конфигурация, не испытанный lockfile.
Точные версии фиксируются после проверки работоспособности и результата.

### Скрипт для OPUS и M2M100

Путь: `scripts/translate_seq2seq.py`.

Назначение: короткий EN→RU smoke-test через Transformers/PyTorch.
MADLAD намеренно отсутствует в допустимых профилях.

При замене существующего скрипта сохранить диагностические материалы
предыдущих испытаний.

```python
import argparse
import time
from pathlib import Path

import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["opus", "m2m100"],
        required=True,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
    )
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(
            "MPS requested but not available; no silent fallback"
        )

    text = Path(args.input).read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Empty input")

    started = time.perf_counter()

    if args.profile == "opus":
        model_id = "Helsinki-NLP/opus-mt-en-ru"
        tokenizer = MarianTokenizer.from_pretrained(model_id)
        model = MarianMTModel.from_pretrained(model_id)
    else:
        model_id = "facebook/m2m100_418M"
        tokenizer = M2M100Tokenizer.from_pretrained(model_id)
        tokenizer.src_lang = "en"
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_id
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
    )
    input_tokens = inputs["input_ids"].shape[-1]

    if input_tokens > 512:
        raise SystemExit(
            "Smoke-test policy: input >512 tokens; "
            "split without truncation"
        )

    model = model.to(
        device=args.device,
        dtype=torch.float32,
    ).eval()

    inputs = {
        key: value.to(args.device)
        for key, value in inputs.items()
    }

    max_new_tokens = 512
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    if args.profile == "m2m100":
        generation_kwargs["forced_bos_token_id"] = (
            tokenizer.get_lang_id("ru")
        )

    print(
        {
            "model": model_id,
            "model_class": type(model).__name__,
            "model_device": str(next(model.parameters()).device),
            "model_dtype": str(next(model.parameters()).dtype),
            "attention": getattr(
                model.config, "_attn_implementation", None
            ),
            "input_tokens": input_tokens,
            "generation_kwargs": generation_kwargs,
        },
        flush=True,
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **generation_kwargs,
        )

    answer = tokenizer.decode(
        output,
        skip_special_tokens=True,
    )
    if not answer.strip():
        raise SystemExit("Empty generation")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(answer + "\n", encoding="utf-8")

    last_token_id = int(output[0, -1])

    print(
        {
            "output_tokens_including_decoder_start": output.shape[-1],
            "last_token_id": last_token_id,
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "output_file": str(output_path),
            "total_s_including_load": time.perf_counter() - started,
        },
        flush=True,
    )

    if (
        output.shape[-1] >= max_new_tokens
        and last_token_id != tokenizer.eos_token_id
    ):
        raise SystemExit(
            "Potential output truncation; inspect saved output"
        )


if __name__ == "__main__":
    main()
```

Особенности:

- OPUS EN→RU не требует чат-промпта или языкового префикса.
- Для M2M100 задаются исходный язык `en` и целевой BOS для `ru`.
- Параметры генерации сохранены относительно исходного smoke-test:
  `max_new_tokens=512`, `do_sample=False`, `num_beams=1`.
- Известное предупреждение M2M100 об `early_stopping` не устранено
  изменением режима генерации: это отдельная диагностическая задача.

> [!TIP]
> Лимит 512 входных токенов — политика короткого smoke-test,
> а не заявление о максимальном контексте обеих архитектур.

### Первые запуски: сначала CPU

Следующие команды нужны для нового прогона или проверки после изменений,
а не для автоматического повторения уже выполненных испытаний.

```bash
source .venv-seq2seq/bin/activate
mkdir -p logs results

python -m py_compile scripts/translate_seq2seq.py
```

При ошибке синтаксической проверки остановиться.

```bash
for profile in opus m2m100; do
  /usr/bin/time -l python -u scripts/translate_seq2seq.py \
    --profile "$profile" \
    --input data/source.en.txt \
    --output "results/$profile.cpu.ru.txt" \
    --device cpu \
    > "logs/$profile.cpu.stdout.log" \
    2> "logs/$profile.cpu.log"

  rc=$?
  printf '\nexit_code=%s\n' "$rc" \
    >> "logs/$profile.cpu.log"
done
```

Команды перезаписывают одноимённые результаты и логи.
Перед повторным запуском сохранить материалы, нужные для сравнения.

CPU — контрольный путь, не обещание приемлемой скорости.

### Проверка результата перед переходом на MPS

Для исходника `data/source.en.txt`:

```text
The UE shall retain the session identifier for 30 seconds.
If the request is rejected, the server MUST NOT modify the configuration.
See RFC 2119 and 3GPP TS 23.501, clause 5.6.
```

Проверить:

1. Процесс завершился с `exit_code=0`.
2. Перевод существует и не пуст.
3. Передано требование о сохранении идентификатора сессии на 30 секунд.
4. При отклонении запроса именно сервер не должен менять конфигурацию.
5. Сохранены `RFC 2119`, `3GPP TS 23.501` и пункт `5.6`.
6. Нет повторов, обрыва текста и непереведённых предложений,
   кроме сохраняемых идентификаторов и обозначений.

Пункт 1 необходим, но недостаточен.

M2M100 в текущих испытаниях не прошёл пункт 3.
До устранения или отдельного решения владельца его результат
не считается принятым переводом.

### Проверка на MPS

Выполнять для профиля, прошедшего CPU-проверку, либо как отдельный
диагностический эксперимент с явной фиксацией этого статуса.

Пример для OPUS, только после проверки CPU-перевода:

```bash
source .venv-seq2seq/bin/activate
mkdir -p logs results

/usr/bin/time -l python -u scripts/translate_seq2seq.py \
  --profile opus \
  --input data/source.en.txt \
  --output results/opus.mps.ru.txt \
  --device mps \
  > logs/opus.mps.stdout.log \
  2> logs/opus.mps.log

rc=$?
printf '\nexit_code=%s\n' "$rc" >> logs/opus.mps.log
```

Для M2M100 аналогичный прогон выполняется только при соблюдении
условий выше, с заменой профиля и имён файлов.

При ошибке MPS сохранить журнал. Не включать незаметный переход
на CPU в рамках того же замера.

Скрипт явно переносит модель и входы на выбранное устройство.
Это не является доказательством исполнения каждой операции на GPU:
fallback отдельных операций отдельно не протоколируется.

FP32 — исходная конфигурация проверки, не утверждение
об оптимальном производственном режиме.

### Правила для coding-агента

- Активные профили этого раздела: `opus`, `m2m100`.
- Не возвращать `madlad` в рабочий скрипт и команды
  без отдельной задачи владельца.
- Не интерпретировать паузу MADLAD как доказательство
  непригодности модели во всех окружениях.
- Не менять зависимости и параметры генерации одновременно
  с исправлением документации.
- Сохранять отдельные логи и результаты каждого эксперимента.
- Не считать код завершения `0` автоматической приёмкой перевода.
- Не удалять ранее собранные диагностические материалы.

## 7. Проверка, фиксация версий и локальность

### 7.1. Минимальные критерии успешного запуска

| Требование | Решение | Задача | Проверка |
|---|---|---|---|
| Генерация локальна | Локальные веса, loopback API | Загрузить артефакты, затем тест без сети | При отключённом интернете повторный запрос завершается; в логах нет внешнего inference API |
| Нет тихой потери входа | `truncation=False`, явные бюджеты | Отправить заведомо слишком длинный текст | Клиент отказывает до генерации, а не обрезает исходник |
| Используется нужная модель | Полный ID + квант + revision/digest | Зафиксировать манифест | Совпадают source ID, GGUF provenance, токенизатор и сервер |
| Перевод завершён | Лимит ответа + контроль окончания | Проверить JSON/текст и исходные предложения | Нет остановки по лимиту и отсутствующих предложений |
| Технические инварианты сохранены | Короткий эталонный вход | Сверить числа и ссылки | `30`, `UE`, `RFC 2119`, `3GPP TS 23.501`, `5.6` не потеряны и не искажены |
| Нормативный смысл сохранён | Ручная проверка | Сверить shall / MUST NOT | Сохранены обязательность и отрицание; это не просто проверка наличия английских слов |
| Ресурсы приемлемы | Одна модель/запрос | Прогон после загрузки весов | Зафиксированы время, пик памяти, swap, отсутствие OOM |

Эти проверки подтверждают запуск и минимальную целостность, но не достаточное качество перевода стандартов. Для выбора основной модели нужен отдельный одинаковый набор фрагментов и ручная оценка терминологии, отрицаний, условий и нормативных требований. Роль «быстрый предпросмотр» для 4B и «основная» для 12B пока является предложением владельца, а не доказанным превосходством.

### 7.2. Сохранить фактическую среду

В каждом Python-окружении отдельно:

```bash
python -m pip freeze > results/requirements-current.lock.txt
```

Переименовать файлы по окружениям, например `requirements-mlx.lock.txt` и `requirements-seq2seq.lock.txt`, не перезаписывать один другим.

Для runtime и аппаратуры:

```bash
sw_vers > results/macos.txt
system_profiler SPHardwareDataType SPDisplaysDataType > results/hardware.txt
brew list --versions > results/brew-versions.txt
llama-server --version > results/llama-version.txt 2>&1
ollama --version > results/ollama-version.txt 2>&1
sysctl vm.swapusage
```

Перед публикацией hardware.txt удалить серийные номера/UUID. Эти команды сбора необязательного runtime выполнять только если он установлен. `pip freeze` фиксирует среду, но сам по себе не является полностью воспроизводимым lockfile с хешами и описанием платформы.

Первый bootstrap использует текущие доступные версии; это сознательное ограничение инструкции, не полностью детерминированная сборка. После smoke-test записать для каждого запуска:

```yaml
status: tested-or-failed
source_model: exact-owner/exact-model
artifact_repo: exact-owner/exact-conversion
artifact_revision: commit-sha-from-hugging-face
artifact_file: exact-file-or-shard-list
artifact_sha256: checksum-per-file
quantization: exact-value
runtime: exact-runtime
runtime_version: exact-version-or-git-commit
tokenizer_revision: commit-sha
os: exact-macos-version
hardware: exact-chip-and-memory
source_lang: en
target_lang: ru
context_tokens: actual-value
max_output_tokens: actual-value
sampling: exact-parameters
result_file: relative-path
```

Это схема полей, не готовый валидный манифест запуска: placeholders нельзя оставлять в отчёте со статусом tested. HF snapshot-кэш содержит каталоги commit SHA; у Ollama записывать digest через локальный API `/api/tags` и сведения `ollama show`.

Для повторного скачивания HF после фиксации SHA использовать:

```bash
hf download OWNER/REPO EXACT_FILE --revision EXACT_COMMIT_SHA \
  --local-dir models/pinned-example
shasum -a 256 models/pinned-example/EXACT_FILE
```

Для MLX snapshot скачивать все необходимые шардированные веса, конфигурации, токенизатор и шаблоны одной revision и передавать в `--model` локальный каталог. Для GGUF — только выбранный квант и все его шарды; запускать `llama-server -m /absolute/path/model.gguf ...`. Для клиента §4 отдельно сохранить токенизатор исходной модели на конкретной revision и заменить ID в профиле на локальный каталог. Не делать checksum терабайтного кэша целиком.

### 7.3. Проверить работу без сети

После первого успешного запуска и загрузки токенизаторов:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Повторить запрос с локальными весами и локальным токенизатором; для llama.cpp на offline-проверке использовать `-m`, а не рассчитывать на разрешение `-hf` через сеть. Для строгого теста отключить интернет на Mac, оставив loopback. Отсутствие cloud inference не равно отсутствию любых сетевых запросов при установке и проверке обновлений.

Остановить серверы Ctrl+C; в Ollama при необходимости:

```bash
ollama ps
ollama stop translategemma:4b
```

Не удалять файлы весов и кэши автоматически. В Ollama освобождение модели и управление `keep_alive` документированы отдельно. [web:195]

### 7.4. Типовые сбои и корректная реакция

| Симптом | Что проверить | Чего не делать |
|---|---|---|
| Ошибка архитектуры qwen35/hunyuan | Версия бинарника, происхождение GGUF, лог загрузки | Не переименовывать архитектуру в config вручную |
| Ошибка TranslateGemma template | Структура content и языковые коды, точная конверсия | Не заменять на универсальный system prompt |
| GGUF найден, но не загружается | Поддержка модели именно runtime, шардирование, checksum | Не считать любое расширение .gguf гарантией совместимости |
| Мусор/служебные токены | Соответствие токенизатора и весов, BOS/EOS и template | Не выдавать обрезку регулярным выражением за исправление |
| Ответ обрывается | stopped_limit, бюджет ответа, длина исходника | Не засчитывать частичный текст как успешный перевод |
| OOM / красный Memory Pressure | Другие процессы, размер кванта, контекст, параллелизм | Не повышать системные GPU/wired лимиты автоматически |
| MPS not implemented | Отдельный CPU-контроль и полный traceback | Не скрывать fallback при замерах |
| 401/403 HF | Учётная запись, принятие условий, токен | Не скачивать неизвестное зеркало для обхода условий |
| Порт занят | Какой сервер уже работает на 8080/11434 | Не завершать все Python-процессы командой killall |

## 8. Контракт для ИИ-агента

Цель первой задачи: подтвердить один маршрут TranslateGemma 4B/MLX на коротком EN→RU фрагменте. Не устанавливать всё и не запускать все модели одной задачей.

Контекст: этот runbook, MacBook Pro M5/64 GB, локальная обработка. Код проекта и базовая ветка в данном исследовании не изучались. Ничего в GitHub не изменялось.

Предлагаемая рабочая ветка после согласования базы: `chore/local-inference-smoke-test`. Если базовая ветка неизвестна, агент должен её уточнить, а не предполагать `main`.

Затрагиваемые файлы после разрешения: один скрипт `scripts/translate_mlx.py`, проверенные зависимости выбранного окружения, один отчёт о smoke-test и этот runbook при необходимости исправлений. Пути предварительно сверить со структурой репозитория; не создавать параллельную иерархию без нужды.

Ограничения:

- Перед установкой пакетов, загрузкой больших весов и записью в репозиторий получить соответствующее разрешение.
- Не переключаться на облачный API, другую модель, квант, runtime или зеркала без согласования.
- Не менять системные лимиты памяти, firewall, глобальный Python и настройки безопасности.
- Не включать trust_remote_code автоматически.
- Не добавлять LangGraph, UI, PDF/OCR, RAG или сервисную архитектуру в smoke-test.
- Не считать каждый узел агентом и не назначать моделям production-роли по факту успешной загрузки.
- Не коммитить веса, токены, реальные конфиденциальные тексты, машинные UUID и большие логи.

Критерии приёмки:

1. Зафиксированы фактические чип, память, macOS, runtime, библиотеки, revision модели и токенизатора.
2. Один короткий запрос завершён; непустой перевод сохранён.
3. Вход, параметры и результат позволяют повторить эксперимент.
4. Проверены ссылки, числа, отрицание и полнота всех предложений.
5. Выполнен повторный тест без сети.
6. Отдельно отмечены непроверенные утверждения, предупреждения и причины отказа.
7. Синтаксис кода проверен, но статическая проверка не выдана за реальный запуск модели.

Что не трогать: существующие графы LangGraph, обработку документов, хранение, UI, CI/CD, настройки GitHub и чужие файлы вне согласованного состава изменений.

Следующие маленькие задачи — отдельно: TranslateGemma 12B/MLX; Index 2B/llama.cpp; затем остальные профили; MPS для seq2seq; Ollama как отдельное сравнение. До этих замеров архитектурный выбор и роли моделей остаются предложениями.

## 9. Итог и открытые решения

Группировка выполнена по реальному интерфейсу запуска: MLX-текстовая генерация; decoder-only GGUF; отдельные seq2seq-модели. Chimera использует тот же тип runtime, что Hunyuan MT, но другой контракт входа и многошаговый сценарий.

Для заданных 64 GB 27B в 4-bit — обоснованный кандидат на одиночный тест, но не подтверждённая скорость и не гарантия выигрыша на 3GPP/RFC. Начальный порядок: 4B MLX → 12B MLX → Index 2B/9B GGUF → остальные модели по одной. Это предложение, не принятое ADR.

Открытые решения владельца:

1. Разрешён ли Transformers/PyTorch как отдельный runtime для MADLAD, OPUS и M2M100, или строго остаёмся в MLX/llama.cpp/Ollama?
2. Какова фактическая версия macOS? Это влияет на использование оптимизаций M5.
3. Есть ли коммерческое применение? Это существенно, в частности, для TowerInstruct.

Форматы входных документов, OCR, сохранение вёрстки, остальные языковые пары и целевая производительность не заданы и этой инструкцией не фиксируются.
