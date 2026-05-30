"""Input/output helpers for portable table-based inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def read_table(path: str | Path, *, sheet: str | int | None = 0) -> pd.DataFrame:
    """Read CSV, TSV, XLSX, Parquet, JSON, or JSONL into a DataFrame."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".tsv":
        return pd.read_csv(p, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if suffix == ".json":
        return pd.read_json(p)

    raise ValueError(f"Unsupported table format: {p.suffix}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame using the format implied by the file extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df.to_csv(p, index=False)
        return
    if suffix == ".tsv":
        df.to_csv(p, index=False, sep="\t")
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(p, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(p, index=False)
        return
    if suffix in {".jsonl", ".ndjson"}:
        df.to_json(p, orient="records", lines=True, force_ascii=False)
        return
    if suffix == ".json":
        df.to_json(p, orient="records", force_ascii=False, indent=2)
        return

    raise ValueError(f"Unsupported output format: {p.suffix}")


def _require_col(df: pd.DataFrame, col: str, role: str) -> None:
    if col not in df.columns:
        raise KeyError(f"Missing {role} column '{col}'. Available columns: {list(df.columns)}")


def prepare_input_news(
    df: pd.DataFrame,
    *,
    text_col: str,
    date_col: str,
    topic_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize an arbitrary input news table for inference."""
    _require_col(df, text_col, "text")
    _require_col(df, date_col, "date")

    out = df.copy()
    out["_input_row_id"] = range(len(out))
    out["_message"] = out[text_col].fillna("").astype(str)
    out["_anchor_date"] = pd.to_datetime(out[date_col], errors="coerce", utc=True).dt.date.astype(str)
    out["_topic"] = out[topic_col].fillna("").astype(str) if topic_col and topic_col in out.columns else ""
    out["_source_id"] = out[id_col].fillna("").astype(str) if id_col and id_col in out.columns else out["_input_row_id"].astype(str)

    bad_dates = out["_anchor_date"].eq("NaT")
    if bad_dates.any():
        sample = out.loc[bad_dates, [text_col, date_col]].head(3).to_dict("records")
        raise ValueError(f"Failed to parse {bad_dates.sum()} input dates. Examples: {sample}")

    empty_text = out["_message"].str.strip().eq("")
    if empty_text.any():
        raise ValueError(f"Found {empty_text.sum()} empty input messages")

    return out


def prepare_corpus(
    df: pd.DataFrame,
    *,
    text_col: str,
    date_col: str,
    channel_col: Optional[str] = None,
    id_col: Optional[str] = None,
    viral_col: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a historical news corpus for temporal retrieval."""
    _require_col(df, text_col, "corpus text")
    _require_col(df, date_col, "corpus date")

    out = pd.DataFrame()
    out["message"] = df[text_col].fillna("").astype(str)
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    out["date_day"] = out["date"].dt.normalize()

    if id_col and id_col in df.columns:
        out["message_id"] = df[id_col].fillna("").astype(str)
    else:
        out["message_id"] = range(len(out))

    if channel_col and channel_col in df.columns:
        out["channel_name"] = df[channel_col].fillna("").astype(str)
    elif "channel_name" in df.columns:
        out["channel_name"] = df["channel_name"].fillna("").astype(str)
    elif "channel" in df.columns:
        out["channel_name"] = df["channel"].fillna("").astype(str)
    else:
        out["channel_name"] = ""

    if viral_col and viral_col in df.columns:
        out["viral_final"] = pd.to_numeric(df[viral_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    elif "viral_final" in df.columns:
        out["viral_final"] = pd.to_numeric(df["viral_final"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    elif "viral_score" in df.columns:
        out["viral_final"] = pd.to_numeric(df["viral_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        out["viral_final"] = 0.0

    out = out[out["message"].str.strip().ne("")].copy()
    out = out[out["date_day"].notna()].copy()
    out.reset_index(drop=True, inplace=True)
    if len(out) == 0:
        raise ValueError("Prepared corpus is empty after dropping empty texts and invalid dates")

    return out

