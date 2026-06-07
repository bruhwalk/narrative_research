# Graph Features Experiment

Экспериментальная ветка исследования: построение графа связей между Telegram-сообщениями и проверка, помогают ли графовые признаки при детекции экономических нарративов.

**Статус: попытка, не вошедшая в основной пайплайн.** Графовые сигналы не улучшили качество ни в ML-бейзлайне, ни в LLM-разметке. Основной проект пошёл через Temporal RAG (`temporal_retrieve_virality_signals/`) и контекстную разметку (`annotation/`, `benchmarks/`).

---

## Зачем этот этап

Гипотеза: структура распространения новостей (схожие сообщения, каналы-источники, соседство в семантическом графе) даёт дополнительный сигнал о вирусности и нарративности поста.

Проверяли два направления:

1. **ML-бейзлайн** — LightGBM на табличных признаках реакций + текстовые эмбеддинги.
2. **LLM + graph ablation** — те же графовые фичи подаются в промпт LLM-аннотатора; сравниваются 5 конфигураций (без графа → постепенное добавление фич).

---

## Связь с основным workflow

| Этап проекта | Папка | Роль |
|---|---|---|
| Сбор корпуса | `hf_datasets/` | Исходные новости и Golden Set |
| Temporal RAG | `temporal_retrieve_virality_signals/` | **Рабочий** контекст для LLM (текст + диффузия) |
| Разметка моделями | `annotation/` | Бенчмарк разметки с/без контекста |
| Метрики | `benchmarks/` | F1, F2, kappa против Golden Set |
| **Граф (этот этап)** | `narratives_graph/` | Отдельная гипотеза, **не дала прироста** |

В `benchmarks/` вариант `llm_context_features` (структурные признаки без текста) тоже уступает `llm_context` (F1 0.694 vs 0.775). Графовый эксперимент подтверждает тот же вывод на другом наборе признаков.

---

## Результаты (кратко)

### LLM ablation (`07_llm_graph_ablation.ipynb`)

Модель `gpt-oss:20b-cloud`, ~500 примеров Golden Set:

| Конфигурация | Narrative F1 |
|---|---:|
| Без графа | **0.531** |
| + `msg_sim_degree` + `msg_clustering` + `ch_pagerank` + `nbr_hist_viral_share` | 0.524 |
| + `msg_sim_degree` + `msg_clustering` + `ch_pagerank` | 0.487 |
| + `msg_sim_degree` | 0.472 |
| + `msg_sim_degree` + `msg_clustering` | 0.458 |

**Вывод:** добавление графовых признаков в промпт не улучшает F1; лучший результат — без графа.

### Графовые фичи как отдельный предиктор

LogReg только на графовых признаках: AUC ≈ 0.56 — слабый сигнал, близко к случайному.

---

## Схема графа

```
Channel ──published──▶ Message ──similar_to──▶ Message
   (id_channel)          (uuid)      (cosine ≥ θ,
                                      |Δt| ≤ window)
```

**Узлы:** каналы (`id_channel`), сообщения (`id`).

**Рёбра:**
- `channel → message` — публикация;
- `message → message` — семантическая близость (BoW → SVD → cosine), с временным окном.

**Признаки (без GNN):**

| Признак | Смысл |
|---|---|
| `msg_sim_degree` | Число семантически похожих соседей |
| `msg_clustering` | Локальная кластеризация в графе сообщений |
| `ch_out_degree` | Сколько сообщений опубликовал канал |
| `ch_pagerank` | PageRank канала в двудольном графе |
| `nbr_hist_viral_share` | Доля «исторически вирусных» соседей |

Схема подробнее генерируется в `04_graph_build_and_features.ipynb` → `artifacts/graph_schema.md`.

---

## Структура папки

| Путь | Описание |
|---|---|
| `notebooks/01_reactions_eda.ipynb` | EDA таблицы реакций, проверка ключей |
| `notebooks/02_messages_eda_and_targets.ipynb` | Слияние сообщений с реакциями, построение таргета вирусности |
| `notebooks/03_baseline_virality_model.ipynb` | Бейзлайн LightGBM без графа (time-based split) |
| `notebooks/04_graph_build_and_features.ipynb` | Построение графа и извлечение признаков |
| `notebooks/05_graph_visualization.ipynb` | Визуализация структуры графа |
| `notebooks/06_graph_features_sanity_check.ipynb` | Распределения и sanity-check фич |
| `notebooks/07_llm_graph_ablation.ipynb` | LLM-разметка с ablation по графовым признакам |
| `data/` | Входные CSV/XLSX (тяжёлые файлы — локально, см. `.gitignore`) |
| `artifacts/` | Промежуточные рёбра, `graph_features.csv`, схема графа |
| `requirements.txt` | Зависимости для ноутбуков |

**Не включено:** `normality_check_graph_features.ipynb` — дублировал `06`, с битыми путями; удалён при интеграции.

---

## Порядок запуска

```text
01 → 02 → 03        # данные + бейзлайн
02 → 04 → 06        # граф и проверка фич
04 → 05             # визуализация (опционально)
04 + data/markup → 07   # LLM ablation
```

### Ожидаемые входные данные (`data/`)

| Файл | Откуда |
|---|---|
| `reactions_all_tg_1610.csv` | Сырые реакции |
| `reactions_all_tg_1610_unique.csv` | Дедуплицированные реакции |
| `all_tg_1610.xlsx` | Сообщения каналов |
| `messages_with_reactions.csv` | Результат `02` |
| `message_targets.csv` | Таргеты вирусности |
| `train_ids.csv`, `valid_ids.csv`, `test_ids.csv` | Time-based split |
| `markup_dataset_filled_20260220.xlsx` | Датасет для LLM-ablation |
| `golden_set_v1.csv` | Экспорт Golden Set для метрик в `07` (из `hf_datasets/economic-narratives-golden-set`) |

### Артефакты (`artifacts/`)

| Файл | Описание |
|---|---|
| `edges_msg_msg.csv` | Рёбра message–message |
| `edges_ch_msg.csv` | Рёбра channel–message |
| `graph_features.csv` | Таблица графовых признаков по `id` |
| `graph_schema.md` | Документация схемы |

---

## Установка

```bash
cd narratives_graph
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Для `07_llm_graph_ablation.ipynb` нужен `OLLAMA_API_KEY` (cloud Ollama). Ключ задаётся только через переменную окружения:

```bash
export OLLAMA_API_KEY="..."
```

---

## Примечания

- Ноутбуки рассчитаны на запуск из `notebooks/`; пути к `data/` и `artifacts/` поднимаются на уровень выше автоматически.
- Тяжёлые CSV (миллионы строк) не коммитятся — положите их локально в `data/`.
- Этот этап сохранён как воспроизводимый архив гипотезы, а не как production-компонент.
