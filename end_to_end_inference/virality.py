"""Virality scoring for index-time corpus preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .io_utils import read_table


def _text_key(value: object) -> str:
    return " ".join(str(value).lower().split())


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _row_max(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[cols].max(axis=1)


def _row_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[cols].mean(axis=1)


def _rank_pct_by_channel(df: pd.DataFrame, col: str, channel_col: str) -> pd.Series:
    if channel_col in df.columns:
        return df.groupby(channel_col)[col].rank(pct=True)
    return df[col].rank(pct=True)


def compute_viral_final(
    engagement_df: pd.DataFrame,
    *,
    id_col: str = "message_id",
    channel_col: str = "id_channel",
    existing_policy: str = "use",
) -> pd.DataFrame:
    """Compute a 0..1 viral_final score from engagement snapshots.

    The formula follows the virality notebook:
    0.45 * static percentiles + 0.20 * dynamic percentiles + 0.35 * ML anomaly percentile.

    If `viral_final` already exists and `existing_policy="use"`, it is returned as-is.
    Set `existing_policy="recompute"` to force recalculation.
    """
    if id_col not in engagement_df.columns:
        raise KeyError(f"Virality source must contain '{id_col}'")

    if existing_policy == "use" and "viral_final" in engagement_df.columns:
        out = engagement_df[[id_col, "viral_final"]].copy()
        out["viral_final"] = pd.to_numeric(out["viral_final"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        return out.groupby(id_col, as_index=False)["viral_final"].max()

    feat = engagement_df.copy()
    eps = 1e-9

    if channel_col not in feat.columns:
        feat[channel_col] = "unknown"

    for base in ["views", "forwards", "reactions", "replies"]:
        for k in range(4):
            col = f"{base}_o{k}"
            if col in feat.columns:
                feat[col] = pd.to_numeric(feat[col], errors="coerce")

    feat["subscribers"] = _num(feat, "subscribers", 0.0).fillna(0.0).clip(lower=0.0)

    for k in range(4):
        view_col = f"views_o{k}"
        feat[f"has_o{k}"] = feat[view_col].notna().astype(int) if view_col in feat.columns else 0

    for base in ["views", "forwards", "reactions", "replies"]:
        for k in range(4):
            delta_col = f"delta_{base}_o{k}"
            cur_col = f"{base}_o{k}"
            prev_col = f"{base}_o{k - 1}"
            if delta_col in feat.columns:
                feat[delta_col] = pd.to_numeric(feat[delta_col], errors="coerce")
                continue
            if cur_col not in feat.columns:
                feat[delta_col] = np.nan
                continue
            if k == 0 or prev_col not in feat.columns:
                feat[delta_col] = _num(feat, cur_col)
            else:
                feat[delta_col] = (_num(feat, cur_col) - _num(feat, prev_col)).clip(lower=0.0)

    feat["has_forwards"] = feat.get("has_forwards", feat[[f"forwards_o{k}" for k in range(4) if f"forwards_o{k}" in feat.columns]].notna().any(axis=1)).astype(bool)
    feat["has_reactions"] = feat.get("has_reactions", feat[[f"reactions_o{k}" for k in range(4) if f"reactions_o{k}" in feat.columns]].notna().any(axis=1)).astype(bool)
    feat["has_replies"] = feat.get("has_replies", feat[[f"replies_o{k}" for k in range(4) if f"replies_o{k}" in feat.columns]].notna().any(axis=1)).astype(bool)

    subs = feat["subscribers"].astype(float) + eps
    for k in range(4):
        feat[f"delta_views_per_sub_o{k}"] = _num(feat, f"delta_views_o{k}") / subs
        feat[f"views_per_sub_o{k}"] = _num(feat, f"views_o{k}") / subs
        feat[f"forwards_per_sub_o{k}"] = _num(feat, f"delta_forwards_o{k}") / subs
        feat[f"reactions_per_sub_o{k}"] = _num(feat, f"delta_reactions_o{k}") / subs

        delta_views = _num(feat, f"delta_views_o{k}")
        feat[f"ctr_forwards_o{k}"] = _num(feat, f"delta_forwards_o{k}") / (delta_views + eps)
        feat[f"ctr_reactions_o{k}"] = _num(feat, f"delta_reactions_o{k}") / (delta_views + eps)

    delta_view_cols = [f"delta_views_o{k}" for k in range(4) if f"delta_views_o{k}" in feat.columns]
    feat["peak_delta_views_0_3"] = _row_max(feat, delta_view_cols)
    feat["mean_delta_views_0_3"] = _row_mean(feat, delta_view_cols)
    feat["peakiness_0_3"] = feat["peak_delta_views_0_3"] / (feat["mean_delta_views_0_3"] + eps)

    if "views_o1" in feat.columns and "views_o3" in feat.columns:
        feat["early_share_1_over_3"] = _num(feat, "views_o1") / (_num(feat, "views_o3") + eps)

    feat["acc_01"] = _num(feat, "delta_views_o1") / (_num(feat, "delta_views_o0") + eps)
    feat["acc_12"] = _num(feat, "delta_views_o2") / (_num(feat, "delta_views_o1") + eps)
    feat["acc_23"] = _num(feat, "delta_views_o3") / (_num(feat, "delta_views_o2") + eps)
    feat["decay_o1"] = _num(feat, "delta_views_o1") / (_num(feat, "delta_views_o0") + eps)

    feat["best_ctr_forwards_0_3"] = _row_max(feat, [f"ctr_forwards_o{k}" for k in range(4)])
    feat["best_ctr_reactions_0_3"] = _row_max(feat, [f"ctr_reactions_o{k}" for k in range(4)])
    feat.loc[~feat["has_reactions"], "best_ctr_reactions_0_3"] = np.nan

    static_feats = [
        "views_per_sub_o0",
        "forwards_per_sub_o0",
        "ctr_forwards_o0",
        "best_ctr_forwards_0_3",
        "peakiness_0_3",
        "reactions_per_sub_o0",
    ]
    static_feats = [c for c in static_feats if c in feat.columns]
    sc_static = pd.DataFrame(index=feat.index)
    for col in static_feats:
        value = feat[col].copy()
        if "ctr_" not in col:
            value = np.log1p(value.clip(lower=0.0))
        sc_static[col + "_pct"] = _rank_pct_by_channel(pd.DataFrame({col: value, channel_col: feat[channel_col]}), col, channel_col)
    if "reactions_per_sub_o0_pct" in sc_static.columns:
        sc_static.loc[~feat["has_reactions"], "reactions_per_sub_o0_pct"] = 0.5
    viral_static = sc_static.fillna(0.5).mean(axis=1) if len(sc_static.columns) else pd.Series(0.5, index=feat.index)

    dyn_feats = [
        "delta_views_per_sub_o0",
        "delta_views_per_sub_o1",
        "decay_o1",
        "acc_01",
        "early_share_1_over_3",
        "peakiness_0_3",
        "best_ctr_forwards_0_3",
        "forwards_per_sub_o1",
    ]
    dyn_feats = [c for c in dyn_feats if c in feat.columns]
    sc_dyn = pd.DataFrame(index=feat.index)
    for col in dyn_feats:
        value = feat[col].copy()
        if col != "decay_o1":
            value = np.log1p(value.clip(lower=0.0))
        sc_dyn[col + "_pct"] = _rank_pct_by_channel(pd.DataFrame({col: value, channel_col: feat[channel_col]}), col, channel_col)
    viral_dynamic = sc_dyn.fillna(0.5).mean(axis=1) if len(sc_dyn.columns) else pd.Series(0.5, index=feat.index)

    viral_ml_pct = _compute_ml_virality_pct(feat, channel_col=channel_col)
    feat["viral_final"] = (0.45 * viral_static + 0.20 * viral_dynamic + 0.35 * viral_ml_pct).clip(0.0, 1.0)

    out = feat[[id_col, "viral_final"]].copy()
    out["viral_final"] = pd.to_numeric(out["viral_final"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out.groupby(id_col, as_index=False)["viral_final"].max()


def _compute_ml_virality_pct(feat: pd.DataFrame, *, channel_col: str) -> pd.Series:
    ml_feats = [
        "delta_views_per_sub_o0",
        "delta_views_per_sub_o1",
        "forwards_per_sub_o0",
        "ctr_forwards_o0",
        "peakiness_0_3",
        "acc_01",
        "best_ctr_forwards_0_3",
        "has_o1",
        "has_o2",
        "has_o3",
        "has_reactions",
        "has_replies",
        "early_share_1_over_3",
        "best_ctr_reactions_0_3",
    ]
    ml_feats = [col for col in ml_feats if col in feat.columns]
    if not ml_feats or len(feat) < 20:
        return pd.Series(0.5, index=feat.index)

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except Exception:
        score = pd.DataFrame(index=feat.index)
        for col in ml_feats:
            value = pd.to_numeric(feat[col], errors="coerce")
            if "ctr_" not in col:
                value = np.log1p(value.clip(lower=0.0))
            score[col] = _rank_pct_by_channel(pd.DataFrame({col: value, channel_col: feat[channel_col]}), col, channel_col)
        return score.fillna(0.5).mean(axis=1)

    x = feat[[channel_col] + ml_feats].copy()
    for col in ml_feats:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        if "ctr_" not in col:
            x[col] = np.log1p(x[col].clip(lower=0.0))
        x[col] = _rank_pct_by_channel(x, col, channel_col)
        x[col + "_isna"] = x[col].isna().astype(int)
        x[col] = x[col].fillna(0.5)

    model_cols = ml_feats + [col + "_isna" for col in ml_feats]
    xs = StandardScaler().fit_transform(x[model_cols])
    contamination = min(0.10, max(0.01, 25 / max(len(feat), 1)))
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(xs)
    raw = -model.decision_function(xs)
    return pd.Series(raw, index=feat.index).rank(pct=True)


def attach_viral_final(
    corpus: pd.DataFrame,
    *,
    virality_source: str | Path,
    corpus_id_col: str = "message_id",
    source_id_col: str = "message_id",
    source_channel_col: str = "id_channel",
    corpus_text_col: str = "message",
    source_text_col: str = "message",
    recompute: bool = False,
    require: bool = False,
    sheet: str | int | None = 0,
) -> pd.DataFrame:
    """Attach or compute viral_final from an engagement source table.

    Primary join is by message id. If ids come from different systems, remaining
    missing rows are filled by exact normalized message-text match.
    """
    path = Path(virality_source)
    if not path.exists():
        if require:
            raise FileNotFoundError(f"Virality source not found: {path}")
        out = corpus.copy()
        if "viral_final" not in out.columns:
            out["viral_final"] = 0.0
        return out

    source = read_table(path, sheet=sheet)
    policy = "recompute" if recompute else "use"
    scores = compute_viral_final(
        source,
        id_col=source_id_col,
        channel_col=source_channel_col,
        existing_policy=policy,
    )

    out = corpus.copy()
    left_key = corpus_id_col
    if left_key not in out.columns:
        raise KeyError(f"Corpus must contain '{left_key}' to attach virality")

    out[left_key] = out[left_key].astype(str)
    scores[source_id_col] = scores[source_id_col].astype(str)
    out = out.drop(columns=["viral_final"], errors="ignore").merge(
        scores.rename(columns={source_id_col: left_key}),
        on=left_key,
        how="left",
    )

    missing_mask = out["viral_final"].isna()
    if missing_mask.any() and corpus_text_col in out.columns and source_text_col in source.columns:
        source_text_scores = (
            source[[source_id_col, source_text_col]]
            .copy()
            .assign(**{source_id_col: source[source_id_col].astype(str)})
            .merge(scores.assign(**{source_id_col: scores[source_id_col].astype(str)}), on=source_id_col, how="left")
        )
        source_text_scores["_text_key"] = source_text_scores[source_text_col].map(_text_key)
        source_text_scores = (
            source_text_scores[source_text_scores["_text_key"].ne("")]
            .groupby("_text_key", as_index=False)["viral_final"]
            .max()
        )

        text_scores = out.loc[missing_mask, [corpus_text_col]].copy()
        text_scores["_row_index"] = text_scores.index
        text_scores["_text_key"] = text_scores[corpus_text_col].map(_text_key)
        text_scores = text_scores.merge(source_text_scores, on="_text_key", how="left")
        fill_values = text_scores.set_index("_row_index")["viral_final"]
        out.loc[fill_values.index, "viral_final"] = out.loc[fill_values.index, "viral_final"].fillna(fill_values)

    out["viral_final"] = pd.to_numeric(out["viral_final"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out
