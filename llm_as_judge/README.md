# LLM-as-Judge

This folder contains the robustness and qualitative-comparison layer for the economic narrative annotation experiments. It is separate from `benchmarks/` because the judge workflow does not compute standard classification metrics directly. Instead, independent LLM evaluators compare human and model annotations under a shared rubric.

## Relation To The Research Workflow

The LLM-as-Judge workflow measures cross-model robustness using:

- rubric-based scoring;
- pairwise comparisons;
- win rates;
- ELO-style rankings;
- plots with and without the human reference.

This layer evaluates whether model rankings remain stable across heterogeneous judge models and whether LLM annotators approach the human Golden Set reference.

## Contents

| Path | Description |
|---|---|
| `scripts/llm_as_judge.py` | Runs judge-model evaluation of human and LLM annotation outputs with ELO tracking. |
| `scripts/analyze_llm_judge.py` | Builds post-hoc judge analytics and visualizations. |
| `figures/llm_judge_analytics/` | Judge plots with the human reference included. |
| `figures/llm_judge_analytics_no_human/` | Judge plots for LLM-only comparisons. |

## Judge Rubric

| Criterion | Meaning |
|---|---|
| `narrative` | Whether the yes/no economic narrative decision matches the expected interpretation. |
| `strength` | Agreement on narrative strength. |
| `effect` | Agreement on direction and magnitude of economic effect. |
| `resonance` | Agreement on public salience. |
| `topic` | Whether the topic assignment is appropriate. |

## Implementation Notes

`llm_as_judge.py` sends the original message, topic, human Golden Set labels, and six LLM outputs to a judge model. The judge returns per-classifier scores, a ranking, and a best-response explanation. The script updates an ELO tracker after each item and exports judge-specific result workbooks.

`analyze_llm_judge.py` aggregates judge result files, creates plots with and without the human reference, and summarizes ranking stability.

## Figure Inventory

| Folder | Meaning |
|---|---|
| `figures/llm_judge_analytics/` | ELO, win-rate, judge-specific, and reason-analysis plots with human annotations included. |
| `figures/llm_judge_analytics_no_human/` | The same style of plots after removing the human reference, useful for comparing LLMs directly. |

