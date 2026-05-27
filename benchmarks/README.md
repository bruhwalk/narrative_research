# Benchmarks

This folder contains the benchmark layer for evaluating economic narrative annotation quality. Annotation files are stored separately in `annotation/`; this folder scores those outputs against the human Golden Set and stores the resulting plots.

## Relation To The Research Workflow

The benchmark corresponds to two experiment types:

1. zero-shot narrative classification;
2. temporal context augmentation via 30-day Temporal RAG context.

The metric choice follows the monitoring-oriented setup of the research: F1 and Cohen's kappa measure balanced quality and agreement with expert labels, while F2 gives additional weight to recall because missed emerging narratives are costly.

## Contents

| Path | Description |
|---|---|
| `scripts/compute_context_metrics.py` | Computes F1, F2, precision, recall, and Cohen's kappa for context variants. |
| `scripts/run_all_contexts.py` | Runs the context-aware annotation workflow across all context variants. |
| `figures/baseline_metrics_plots/` | Baseline metric plots from the local workflow. |
| `figures/context_metrics_plots/` | Plots for `llm_context`, `llm_context_features`, and `llm_context_features_full`. |

## Context Variants

| Variant | Interpretation |
|---|---|
| `without_context` | Model sees only target post text and topic. |
| `llm_context` | Model sees a compact 30-day timeline of related prior posts. |
| `llm_context_features` | Model sees aggregated temporal and diffusion features only. |
| `llm_context_features_full` | Model sees both timeline and structural features. |

## Main Binary Narrative Result

| Context | Samples | F1 | F2 | Precision | Recall | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| `without_context` | 500 | 0.667 | 0.697 | 0.621 | 0.719 | 0.473 |
| `llm_context` | 100 | 0.775 | 0.852 | 0.674 | 0.912 | 0.631 |
| `llm_context_features` | 100 | 0.694 | 0.718 | 0.658 | 0.735 | 0.523 |
| `llm_context_features_full` | 100 | 0.722 | 0.747 | 0.684 | 0.765 | 0.567 |

## Figure Inventory

| Folder | Meaning |
|---|---|
| `figures/baseline_metrics_plots/` | Baseline plots for F1, precision, recall, kappa, and combined metrics. |
| `figures/context_metrics_plots/llm_context/` | Metric plots for the temporal text context condition. |
| `figures/context_metrics_plots/llm_context_features/` | Metric plots for structured feature-only context. |
| `figures/context_metrics_plots/llm_context_features_full/` | Metric plots for combined text and feature context. |
| `figures/context_metrics_plots/summary/` | Cross-context comparison plots. |

