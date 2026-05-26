---
license: cc-by-nc-4.0
language:
  - ru
size_categories:
  - n<1K
tags:
  - economics
  - telegram
  - narratives
  - russian
  - news
  - social-media
  - annotation
  - golden-set
  - temporal-rag
  - llm-classification
  - virality
task_categories:
  - text-classification
pretty_name: Economic Narratives Golden Set
---

# Economic Narratives Golden Set

A manually annotated dataset of **500 Russian-language Telegram posts** labeled for economic narrative presence, with LLM-generated temporal contexts. This is the evaluation benchmark from the paper on LLM-based economic narrative detection.

## Associated Paper

> **Going Viral: LLM-Based Modeling of Economic Narratives**

## Dataset Description

The Golden Set was sampled from the [Economic Telegram News Corpus](https://huggingface.co/datasets/bruhwalkk/economic-telegram-news-corpus-2025) to ensure coverage across virality levels and all 9 economic topics (~55–57 posts per topic).

### Annotation Protocol

- **2 independent annotators** per post (initial overlap = 2)
- Disagreements resolved via majority voting (up to 5 annotations total)
- **Inter-annotator agreement**: Cohen's κ = 0.794 (strong)
- **Narrative prevalence**: 34.2% (171/500)

### Key Statistics

| Metric | Value |
|--------|-------|
| Total messages | 500 |
| Narrative prevalence | 34.2% (171) |
| Mean viral_final | 0.583 |
| Viral (narrative) | 0.624 |
| Viral (non-narrative) | 0.562 |
| ROC-AUC (viral → narrative) | 0.666 |
| Cohen's κ | 0.794 |
| Narratives among negative-effect posts | 54% |

### Temporal RAG Context

Each post is enriched with a 30-day temporal context generated via a leakage-safe Temporal RAG pipeline (FAISS + BM25 hybrid retrieval with temporal RRF fusion). Adding this context improves recall from 0.719 to 0.912 and F2 from 0.697 to 0.852.

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | string (UUID) | Unique message identifier |
| `message` | string | Full post text (Russian) |
| `channel_id` | int | Channel ID |
| `anchor_date` | datetime | Publication timestamp |
| `llm_context` | string | 30-day temporal text context (Temporal RAG digest) |
| `llm_features_json` | string (JSON) | Structured diffusion features |
| `topic_llm` | string | LLM-assigned economic topic |
| `economic_narrative` | int (0/1) | **Target label** — expert narrative annotation |
| `narrative_strength` | float [0–1] | Narrative strength (expert annotation) |
| `information_resonance` | float [0–1] | Potential mass-audience salience |
| `economic_effect` | float [-1, +1] | Economic effect direction and magnitude |
| `viral_engagement` | float | Virality engagement score |
| `out_of_coverage` | bool | Whether the post is outside temporal coverage |
| `mentions_today` | int | Same-topic mentions on publication day |
| `mentions_sum_window` | int | Total mentions in 30-day window |
| `days_nonzero_window` | int | Days with non-zero mentions in window |
| `spread_level` | string | Spread level category |
| `trend_slope_all` | float | Mention trend slope |
| `llm_context_features` | string | Context + structural diffusion features |
| `llm_context_features_full` | string | Full context combining timeline + features |

## Label Distribution

| Label | Count | % |
|-------|-------|---|
| Not a narrative (0) | 329 | 65.8% |
| Narrative (1) | 171 | 34.2% |

## Annotation Dimensions

- **Economic effect**: strong negative / weak negative / neutral / weak positive / strong positive
- **Information resonance**: low / medium / high (potential mass-audience salience)
- **Economic narrative**: yes / no (whether the post can become a viral explanatory story)
- **Narrative strength**: low / medium / high (influence potential on broad audience)

### Narrative Checklist (from annotation guidelines)

A post is assessed from the perspective of an average Russian reader considering:
- A clear trigger event
- Emotional framing (fear / uncertainty / anger / optimism)
- Simplification / generalization ("prices will surge", "everyone will get poorer")
- Explicit causal links to economy / markets / everyday life
- References to authorities / experts and broad public discussion

## Usage

```python
from datasets import load_dataset

ds = load_dataset("bruhwalkk/economic-narratives-golden-set")
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
