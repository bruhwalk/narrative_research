# End-to-End Inference

Папка содержит переносимый pipeline:

1. собирает/приклеивает `viral_final` для исторического корпуса;
2. строит Temporal RAG индекс по историческому корпусу новостей;
3. принимает на вход таблицу новых/размечаемых новостей;
4. для каждой новости достает 30-дневный контекст как в `temporal retrieval_goldenset.ipynb`;
5. передает новость + контекст в LLM;
6. сохраняет таблицу с `llm_context`, сырым ответом модели и распарсенными полями разметки.

## Что На Входе

Минимальная входная таблица должна иметь:

| column | meaning |
|---|---|
| `message` | текст новости |
| `date` | дата/время публикации |
| `topic` | тема, опционально |

Имена колонок можно переопределить аргументами `--text-col`, `--date-col`, `--topic-col`.

## Установка

```bash
cd /path/to/narrative_research
python -m venv .venv
source .venv/bin/activate
pip install -r end_to_end_inference/requirements.txt
```

Для Ollama Cloud:

```bash
export OLLAMA_HOST=https://ollama.com
export OLLAMA_API_KEY=...
```

Для локальной Ollama обычно достаточно запущенного сервера `ollama serve`.

## 1. Построить Индекс

Индекс строится один раз. По умолчанию используется корпус:

`temporal_retrieve_virality_signals/data/cleaned_news.csv`

На этапе построения индекса pipeline автоматически приклеивает `viral_final` из:

`temporal_retrieve_virality_signals/data/dataset_tg_economic.parquet`

Если в этой таблице уже есть `viral_final`, он используется напрямую. Если нужно пересчитать его из engagement-метрик (`views_o*`, `forwards_o*`, `reactions_o*`, `subscribers`), добавь `--recompute-virality`.

Склейка идет сначала по `message_id`. Если id в корпусе и engagement-таблице из разных систем, pipeline добивает пропуски точным матчем по нормализованному тексту `message`.

```bash
python -m end_to_end_inference build-index \
  --corpus temporal_retrieve_virality_signals/data/cleaned_news.csv \
  --corpus-text-col message \
  --corpus-date-col date_day \
  --corpus-channel-col channel_name \
  --corpus-id-col message_id \
  --index-dir end_to_end_inference/indexes/e5_large \
  --encoder-name intfloat/multilingual-e5-large \
  --device cpu
```

На машине с CUDA можно заменить `--device cpu` на `--device cuda`.

Пересчитать `viral_final` при сборке индекса:

```bash
python -m end_to_end_inference build-index \
  --corpus temporal_retrieve_virality_signals/data/cleaned_news.csv \
  --virality-source temporal_retrieve_virality_signals/data/dataset_tg_economic.parquet \
  --recompute-virality \
  --index-dir end_to_end_inference/indexes/e5_large
```

Отключить virality полностью:

```bash
python -m end_to_end_inference build-index \
  --corpus temporal_retrieve_virality_signals/data/cleaned_news.csv \
  --no-compute-virality
```

## 2. Прогнать Таблицу Новостей

Только получить контекст:

```bash
python -m end_to_end_inference infer \
  --input path/to/news.xlsx \
  --output end_to_end_inference/outputs/news_with_context.xlsx \
  --index-dir end_to_end_inference/indexes/e5_large \
  --text-col message \
  --date-col date \
  --topic-col topic \
  --context-only
```

Получить контекст и LLM-разметку:

```bash
python -m end_to_end_inference infer \
  --input path/to/news.xlsx \
  --output end_to_end_inference/outputs/news_inference.xlsx \
  --index-dir end_to_end_inference/indexes/e5_large \
  --text-col message \
  --date-col date \
  --topic-col topic \
  --model qwen3-vl:235b-instruct-cloud \
  --ollama-host https://ollama.com \
  --continue-on-error
```

Для короткого теста:

```bash
python -m end_to_end_inference infer \
  --input path/to/news.csv \
  --output end_to_end_inference/outputs/test.csv \
  --max-rows 5 \
  --context-only
```

## Опциональный Judge

Ноутбук использовал LLM judge для фильтрации кандидатов. В CLI он выключен по умолчанию, чтобы pipeline запускался на обычном ПК. Если нужен judge через Ollama:

```bash
python -m end_to_end_inference infer \
  --input path/to/news.csv \
  --output end_to_end_inference/outputs/news_inference.csv \
  --judge-model qwen2.5:14b-instruct \
  --model qwen3-vl:235b-instruct-cloud
```

Без `--judge-model` все кандидаты после semantic filter получают `judge_relevance=1`.

## Логика Retrieval

Код повторяет ключевой подход из `temporal retrieval_goldenset.ipynb`:

- dense retrieval через `SentenceTransformer` с E5-префиксами `query:` и `passage:`;
- BM25 по русской токенизации;
- RRF fusion;
- ограничение по времени: документы не позже опорной даты и внутри окна `--max-window-days`, по умолчанию 30;
- фильтр `dense_score >= tau_sem_hi OR (dense_score >= tau_sem_lo AND bm25_score >= tau_bm25_lo)`;
- virality boost по готовому или рассчитанному `viral_final`;
- дневная агрегация: упоминания, каналы, суммарный `viral_final`, IQR-выбросы;
- примеры самых релевантных сообщений добавляются в `llm_context`.

## Как Считается Virality

Если `--recompute-virality` включен или в source-таблице нет `viral_final`, score считается по формуле из virality notebook:

```text
viral_final =
  0.45 * viral_static
  + 0.20 * viral_dynamic
  + 0.35 * percentile(IsolationForest anomaly score)
```

`viral_static` строится из относительных просмотров/форвардов/реакций и CTR внутри канала. `viral_dynamic` использует раннюю динамику, acceleration/decay и peakiness. Все тяжелохвостые признаки логарифмируются и переводятся в percentile внутри `id_channel`, чтобы большие каналы не съедали маленькие.

## Выходные Колонки

Pipeline добавляет:

| column | meaning |
|---|---|
| `llm_context` | temporal context для новости |
| `retrieved_candidates_n` | сколько кандидатов осталось после semantic filter |
| `kept_candidates_n` | сколько кандидатов осталось после judge или сколько передано без judge |
| `llm_raw_response` | сырой ответ модели |
| `economic_effect` | -2..2 |
| `information_resonance` | 1..3 |
| `topic_agreement` | 1..3 |
| `economic_narrative` | `Да` / `Нет` |
| `narrative_strength` | 1..3 |
| `comment` | короткое объяснение |
| `inference_error` | ошибка по строке, если включен `--continue-on-error` |

## Артефакты

`end_to_end_inference/indexes/` и `end_to_end_inference/outputs/` не коммитятся. Индекс может быть тяжелым и должен строиться локально на той машине, где запускается pipeline.
