# Annotation

This folder contains the annotation layer for the economic narrative detection experiments. It stores the scripts used to produce model annotations and the resulting checked workbooks.

## Relation To The Research Workflow

Annotation is the first experimental layer after the Golden Set is prepared. Each model receives the same post text, topic, and narrative rubric, then returns a structured label set. These outputs are used by the benchmark scripts and by the LLM-as-Judge comparison.

The annotation rubric follows the same core dimensions used in the Golden Set:

| Field | Range | Meaning |
|---|---:|---|
| `economic_narrative` | yes/no | Whether the post can function as a viral economic narrative. |
| `narrative_strength` | 1-3 | Strength of emotional framing, simplification, and action potential. |
| `economic_effect` | -2..2 | Direction and magnitude of expected economic impact. |
| `information_resonance` | 1-3 | Potential for mass public salience. |
| `topic_agreement` | 1-3 | Whether the supplied topic matches the post. |

## Contents

| Path | Description |
|---|---|
| `scripts/multi_model_annotation.py` | Baseline no-context multi-model annotation script. |
| `scripts/remarkup_qwen3_context.py` | Context-aware re-annotation script for `qwen3-vl:235b-instruct-cloud`. |
| `combined_markup/LLm markup.xlsx` | Combined no-context markup with all model slots. |
| `markup_by_model/` | One XLSX workbook per model slot extracted from the combined markup. |
| `context_markup/` | Context-enriched re-markup files for the three context variants. |

## Model-Specific Markup Files

| File | Model |
|---|---|
| `markup_by_model/01_gpt_oss_20b_cloud.xlsx` | `gpt-oss:20b-cloud` |
| `markup_by_model/02_gpt_oss_120b_cloud.xlsx` | `gpt-oss:120b-cloud` |
| `markup_by_model/03_qwen3_vl_235b_instruct_cloud.xlsx` | `qwen3-vl:235b-instruct-cloud` |
| `markup_by_model/04_gemma3_27b_cloud.xlsx` | `gemma3:27b-cloud` |
| `markup_by_model/05_qwen3_vl_8b.xlsx` | `qwen3-vl:8b` |
| `markup_by_model/06_wedlm_8b_instruct.xlsx` | `WeDLM-8B-Instruct` |

## Context Markup Files

| File | Context Variant |
|---|---|
| `context_markup/goldenset_remarked_llm_context_20260302_185955.xlsx` | `llm_context` |
| `context_markup/goldenset_remarked_llm_context_features_20260302_190459.xlsx` | `llm_context_features` |
| `context_markup/goldenset_remarked_llm_context_features_full_20260302_190838.xlsx` | `llm_context_features_full` |

## Implementation Notes

`multi_model_annotation.py` reads a spreadsheet with `message` and `topic`, applies a shared narrative prompt to each model, parses a JSON object from the response, validates field ranges, and writes model outputs into numbered slots.

`remarkup_qwen3_context.py` keeps the same output schema but adds one of three Temporal RAG-derived context blocks:

| Context Field | Meaning |
|---|---|
| `llm_context` | 30-day temporal text context. |
| `llm_context_features` | Structured diffusion and spread features. |
| `llm_context_features_full` | Timeline plus structural features. |

All included XLSX files were checked for common credential patterns before packaging.

