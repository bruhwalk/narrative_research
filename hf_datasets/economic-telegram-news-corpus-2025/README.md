---
license: cc-by-nc-4.0
language:
  - ru
size_categories:
  - 10K<n<100K
tags:
  - economics
  - telegram
  - narratives
  - russian
  - news
  - social-media
  - virality
  - nlp
  - text-classification
task_categories:
  - text-classification
pretty_name: Economic Telegram News Corpus 2025
---

# Economic Telegram News Corpus 2025

A corpus of **31,292 Russian-language economic news posts** collected from 7 major Telegram channels, spanning January 2024 to September 2025. The dataset supports research on economic narrative detection, topic classification, and information diffusion in social media.

## Associated Paper

> **Going Viral: LLM-Based Modeling of Economic Narratives**

## Dataset Description

The raw collection contains 123,273 posts. The economic corpus was constructed by:
1. Removing duplicates and near-duplicates
2. Excluding non-news content
3. Selecting posts assigned to economic topics via an LLM-based classifier (~90% accuracy on the Golden Set)

### Virality Score

Each post includes a composite virality score (`viral_final`) computed over a 3-day window after publication:

```
viral_final = 0.45 * viral_static + 0.20 * viral_dynamic + 0.35 * viral_ml
```

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | string (UUID) | Unique message identifier |
| `id_channel` | int | Channel ID (1–7) |
| `message` | string | Full post text (Russian) |
| `viral_final` | float | Composite virality score [0, 1] |
| `is_economic` | bool | Economic post flag |
| `economic_topic` | string | LLM-assigned economic topic (9 categories) |
| `topic_confidence` | float | Topic classification confidence |
| `channel_name` | string | Telegram channel name |
| `channel_w` | float | Channel weight |
| `message_vector` | string | PostgreSQL tsvector (full-text search index) |
| `subscribers` | int | Channel subscriber count |
| `date` | datetime | Exact publication timestamp (UTC) |
| `date_day` | datetime | Publication date (day-level, UTC) |

## Channels (7)

| Channel | Posts |
|---------|-------|
| Forbes Russia | — |
| Блумберг | — |
| РИА Новости | — |
| Экономика | — |
| Раньше всех. Ну почти | — |
| Банки, деньги, два офшора | — |
| Сигналы РЦБ | — |

## Topics (9)

| Topic | Count |
|-------|-------|
| Государственная экономическая политика | 9,040 |
| Корпоративные финансы | 4,572 |
| Макроэкономика | 3,978 |
| Санкции и геополитика | 3,582 |
| Рынки капитала | 3,169 |
| Сырьевые рынки | 2,118 |
| Международная торговля | 1,825 |
| Другое | 1,608 |
| Валютный рынок | 1,400 |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("bruhwalkk/economic-telegram-news-corpus-2025")
```

## Citation

If you use this dataset, please cite the associated paper:

```bibtex
@article{economic_narratives_2025,
  title={Going Viral: LLM-Based Modeling of Economic Narratives},
  journal={Записки научных семинаров ПОМИ},
  year={2025}
}
```

## License

CC-BY-NC-4.0
