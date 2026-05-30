"""Temporal retrieval and context construction.

This module intentionally mirrors the important logic from
`temporal retrieval_goldenset.ipynb` in a script-friendly form:

1. dense E5 retrieval;
2. BM25 retrieval;
3. RRF fusion with a temporal window;
4. semantic thresholding;
5. optional LLM judge;
6. daily context construction for downstream annotation.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .llm_client import parse_relevance
from .prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt


def tokenize_ru(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^0-9a-zа-яё\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def slugify_encoder_name(name: str) -> str:
    value = str(name).strip().lower().replace("/", "_")
    value = re.sub(r"[^0-9a-z._-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


@dataclass
class IndexBundle:
    df: pd.DataFrame
    encoder: object
    faiss_index: object
    bm25: object
    encoder_name: str
    device: str


def _paths(index_dir: str | Path, encoder_name: str) -> dict[str, Path]:
    root = Path(index_dir)
    tag = slugify_encoder_name(encoder_name)
    return {
        "root": root,
        "corpus": root / "corpus.parquet",
        "faiss": root / f"faiss__{tag}.index",
        "bm25": root / f"bm25_corpus_tok__{tag}.pkl",
        "meta": root / "metadata.json",
    }


def build_index(
    corpus_df: pd.DataFrame,
    *,
    index_dir: str | Path,
    encoder_name: str = "intfloat/multilingual-e5-large",
    device: str = "cpu",
    batch_size: int = 64,
) -> None:
    """Build and persist FAISS + BM25 artifacts."""
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    paths = _paths(index_dir, encoder_name)
    paths["root"].mkdir(parents=True, exist_ok=True)

    corpus = corpus_df.copy()
    corpus["message"] = corpus["message"].fillna("").astype(str)
    corpus["date_day"] = pd.to_datetime(corpus["date_day"], errors="coerce", utc=True).dt.normalize()
    corpus.to_parquet(paths["corpus"], index=False)

    corpus_tok = [tokenize_ru(text) for text in corpus["message"].tolist()]
    BM25Okapi(corpus_tok)

    encoder = SentenceTransformer(encoder_name, device=device)
    doc_inputs = ["passage: " + text for text in corpus["message"].tolist()]
    embeddings = encoder.encode(
        doc_inputs,
        batch_size=int(batch_size),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)

    faiss.write_index(index, str(paths["faiss"]))
    with open(paths["bm25"], "wb") as fh:
        pickle.dump(corpus_tok, fh)

    meta = {
        "encoder_name": encoder_name,
        "device": device,
        "rows": int(len(corpus)),
        "embedding_dim": int(embeddings.shape[1]),
    }
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(
    *,
    index_dir: str | Path,
    encoder_name: str = "intfloat/multilingual-e5-large",
    device: str = "cpu",
) -> IndexBundle:
    """Load persisted retrieval artifacts."""
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    paths = _paths(index_dir, encoder_name)
    missing = [str(p) for key, p in paths.items() if key != "root" and not p.exists()]
    if missing:
        raise FileNotFoundError("Missing index artifacts: " + ", ".join(missing))

    df = pd.read_parquet(paths["corpus"])
    df["date_day"] = pd.to_datetime(df["date_day"], errors="coerce", utc=True).dt.normalize()
    faiss_index = faiss.read_index(str(paths["faiss"]))
    with open(paths["bm25"], "rb") as fh:
        corpus_tok = pickle.load(fh)
    bm25 = BM25Okapi(corpus_tok)
    encoder = SentenceTransformer(encoder_name, device=device)
    return IndexBundle(df=df, encoder=encoder, faiss_index=faiss_index, bm25=bm25, encoder_name=encoder_name, device=device)


def _topk_indices_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    if k == len(scores):
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
    return idx.astype(int)


def dense_candidates_faiss(index, encoder, query: str, top_n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    query_vector = encoder.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False)
    query_vector = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32))
    scores, idx = index.search(query_vector, int(top_n))
    return idx[0].astype(int), scores[0].astype(np.float32)


def _time_arrays(df: pd.DataFrame, rowpos: np.ndarray, anchor_date, date_col: str) -> tuple[pd.Series, np.ndarray]:
    anchor = pd.to_datetime(anchor_date, utc=True).normalize()
    dates = pd.to_datetime(df.loc[rowpos, date_col], errors="coerce", utc=True).dt.normalize()
    age = (anchor - dates).dt.days.to_numpy(dtype=np.float32)
    age = np.where(np.isfinite(age), age, 1e9).astype(np.float32)
    age = np.where(age < 0, 1e9, age).astype(np.float32)
    return dates, age


def _time_rank_from_age(age_days: np.ndarray) -> np.ndarray:
    order = np.argsort(age_days, kind="stable")
    rank = np.empty_like(order, dtype=np.int32)
    rank[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return rank


def hybrid_retrieve_rrf(
    df: pd.DataFrame,
    index,
    encoder,
    bm25,
    tokenize_fn: Callable[[str], list[str]],
    query: str,
    *,
    k: Optional[int] = None,
    topN_each: int = 10000,
    k_rrf: int = 60,
    w_dense: float = 1.0,
    w_bm25: float = 1.0,
    anchor_date: str | pd.Timestamp | None = None,
    date_col: str = "date_day",
    max_window_days: Optional[int] = 30,
    w_time: float = 0.0,
    w_channel: Optional[float] = None,
    channel_w_col: str = "channel_w",
    w_viral: Optional[float] = None,
    viral_col: str = "viral_final",
) -> pd.DataFrame:
    """Wide hybrid retrieval with scores needed by the semantic filter."""
    if anchor_date is not None:
        anchor = pd.to_datetime(anchor_date, utc=True).normalize()
        all_dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.normalize()
        allowed = all_dates <= anchor
        if max_window_days is not None:
            age_all = (anchor - all_dates).dt.days
            allowed &= (age_all >= 0) & (age_all <= int(max_window_days))
        allowed_np = allowed.to_numpy(dtype=bool)
    else:
        allowed_np = None

    dense_idx, dense_scores_raw = dense_candidates_faiss(index, encoder, query, top_n=int(topN_each))
    if allowed_np is not None and len(dense_idx) > 0:
        keep = allowed_np[dense_idx]
        dense_idx = dense_idx[keep]
        dense_scores_raw = dense_scores_raw[keep]

    dense_rank = {int(rowpos): rank for rank, rowpos in enumerate(dense_idx, start=1)}
    dense_score = {int(rowpos): float(score) for rowpos, score in zip(dense_idx, dense_scores_raw)}

    bm_scores = None
    bm_rank: dict[int, int] = {}
    if bm25 is not None:
        bm_scores = bm25.get_scores(tokenize_fn(query)).astype(np.float32)
        if allowed_np is not None:
            bm_scores[~allowed_np] = -np.inf
        bm_idx = _topk_indices_from_scores(bm_scores, int(topN_each))
        bm_rank = {int(rowpos): rank for rank, rowpos in enumerate(bm_idx, start=1)}

    union = np.array(sorted(set(dense_rank) | set(bm_rank)), dtype=int)
    if len(union) == 0:
        return df.iloc[[]].copy().reset_index(drop=True)

    rrf = np.zeros(len(union), dtype=np.float32)
    for pos, rowpos in enumerate(union):
        if int(rowpos) in dense_rank:
            rrf[pos] += float(w_dense) / (float(k_rrf) + dense_rank[int(rowpos)])
        if int(rowpos) in bm_rank:
            rrf[pos] += float(w_bm25) / (float(k_rrf) + bm_rank[int(rowpos)])

    rank_time = None
    if anchor_date is not None and w_time and len(union) > 0:
        _, age = _time_arrays(df, union, anchor_date, date_col)
        rank_time = _time_rank_from_age(age)
        rrf = rrf + (float(w_time) / (float(k_rrf) + rank_time.astype(np.float32)))

    order = np.argsort(-rrf)
    union = union[order]
    rrf = rrf[order]
    if rank_time is not None:
        rank_time = rank_time[order]

    out = df.iloc[union].copy()
    out["_rowpos"] = union
    out["score_rrf"] = rrf
    out["rank_dense"] = out["_rowpos"].map(lambda rp: dense_rank.get(int(rp), np.nan))
    out["rank_bm25"] = out["_rowpos"].map(lambda rp: bm_rank.get(int(rp), np.nan))
    out["dense_score"] = out["_rowpos"].map(lambda rp: dense_score.get(int(rp), np.nan))
    if bm_scores is None:
        out["bm25_score"] = np.nan
    else:
        out["bm25_score"] = out["_rowpos"].map(
            lambda rp: float(bm_scores[int(rp)]) if np.isfinite(bm_scores[int(rp)]) else -np.inf
        )

    if anchor_date is not None:
        doc_day, age = _time_arrays(df, union, anchor_date, date_col)
        out["doc_day"] = doc_day.dt.tz_localize(None)
        out["age_days"] = age
        if rank_time is not None:
            out["rank_time"] = rank_time

    if channel_w_col in out.columns:
        if w_channel is None:
            w_channel = 0.10 * float(np.std(out["score_rrf"].to_numpy(dtype=np.float32)) or 1.0)
        out["score_rrf"] = out["score_rrf"] + float(w_channel) * out[channel_w_col].astype(np.float32)

    if viral_col in out.columns:
        viral = np.clip(out[viral_col].astype(np.float32).to_numpy(), 0.0, 1.0) ** 5.0
        if w_viral is None:
            w_viral = 0.5 * float(np.std(out["score_rrf"].to_numpy(np.float32)) or 1.0)
        out["score_rrf"] = out["score_rrf"] + float(w_viral) * viral

    out = out.sort_values("score_rrf", ascending=False)
    if k is not None:
        out = out.head(int(k))
    return out.reset_index(drop=True)


def semantic_filter_reasonable(
    cand: pd.DataFrame,
    *,
    tau_sem_hi: float = 0.8,
    tau_sem_lo: float = 0.5,
    tau_bm25_lo: float = 1.0,
    dense_col: str = "dense_score",
    bm25_col: str = "bm25_score",
) -> pd.DataFrame:
    """Keep dense_hi OR dense_lo+bm25_lo candidates."""
    if cand is None or len(cand) == 0:
        return cand
    if dense_col not in cand.columns:
        raise KeyError(f"'{dense_col}' not found in candidates")

    dense = cand[dense_col].astype(np.float32).to_numpy()
    if bm25_col in cand.columns:
        bm25_values = cand[bm25_col].astype(np.float32).to_numpy()
    else:
        bm25_values = np.full(len(cand), -np.inf, dtype=np.float32)

    dense = np.where(np.isfinite(dense), dense, -np.inf).astype(np.float32)
    bm25_values = np.where(np.isfinite(bm25_values), bm25_values, -np.inf).astype(np.float32)
    keep = (dense >= float(tau_sem_hi)) | ((dense >= float(tau_sem_lo)) & (bm25_values >= float(tau_bm25_lo)))
    return cand.loc[keep].copy().reset_index(drop=True)


def retrieve_all_candidates(
    bundle: IndexBundle,
    query: str,
    *,
    anchor_date: str,
    max_window_days: int = 30,
    topN_each: int = 10000,
    tau_sem_hi: float = 0.8,
    tau_sem_lo: float = 0.5,
    tau_bm25_lo: float = 1.0,
) -> pd.DataFrame:
    cand = hybrid_retrieve_rrf(
        df=bundle.df,
        index=bundle.faiss_index,
        encoder=bundle.encoder,
        bm25=bundle.bm25,
        tokenize_fn=tokenize_ru,
        query=query,
        k=None,
        topN_each=int(topN_each),
        anchor_date=anchor_date,
        max_window_days=int(max_window_days),
        w_time=0.0,
    )
    return semantic_filter_reasonable(
        cand,
        tau_sem_hi=tau_sem_hi,
        tau_sem_lo=tau_sem_lo,
        tau_bm25_lo=tau_bm25_lo,
    )


def judge_filter_candidates(
    cand: pd.DataFrame,
    *,
    query: str,
    judge_generate: Callable[[str, Optional[str]], str],
    keep_threshold: int = 1,
    doc_max_chars: int = 1200,
) -> pd.DataFrame:
    """Filter candidates through a simple JSON relevance judge."""
    if cand is None or len(cand) == 0:
        return cand

    relevances: list[int] = []
    for _, row in cand.iterrows():
        prompt = build_judge_prompt(
            query=query,
            candidate_text=str(row.get("message", ""))[: int(doc_max_chars)],
            channel=str(row.get("channel_name", "")),
            date_day=str(row.get("date_day", ""))[:10],
        )
        raw = judge_generate(prompt, JUDGE_SYSTEM_PROMPT)
        relevances.append(parse_relevance(raw))

    out = cand.copy()
    out["judge_relevance"] = relevances
    return out[out["judge_relevance"] >= int(keep_threshold)].copy().reset_index(drop=True)


def build_daily_context(
    cand: pd.DataFrame,
    *,
    message: str,
    anchor_date: str,
    max_window_days: int = 30,
    keep_judge: int = 1,
    iqr_k: float = 1.5,
    max_example_docs: int = 8,
    example_chars: int = 420,
) -> str:
    """Build the compact context block used by the downstream annotation model."""
    if cand is None or len(cand) == 0:
        return (
            f"Опорная дата: {anchor_date}.\n"
            f"Запрос (текст новости): {message}\n\n"
            f"За окно {max_window_days} дней релевантных сообщений не найдено."
        )

    dfp = cand.copy()
    dfp["date_day"] = pd.to_datetime(dfp["date_day"], errors="coerce", utc=True).dt.floor("D")
    dfp["viral_final"] = pd.to_numeric(dfp.get("viral_final", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "judge_relevance" in dfp.columns:
        dfp["judge_relevance"] = pd.to_numeric(dfp["judge_relevance"], errors="coerce").fillna(0).astype(int)
    else:
        dfp["judge_relevance"] = 1

    kept = dfp[dfp["judge_relevance"] >= int(keep_judge)].copy()
    if len(kept) == 0:
        return (
            f"Опорная дата: {anchor_date}.\n"
            f"Запрос (текст новости): {message}\n\n"
            f"Кандидаты найдены (N={len(dfp)}), но после фильтра judge_relevance >= {keep_judge} ничего не осталось."
        )

    daily = (
        kept.groupby("date_day", as_index=False)
        .agg(mentions=("message", "size"), viral_sum=("viral_final", "sum"))
        .sort_values("date_day")
    )
    q1 = daily["viral_sum"].quantile(0.25)
    q3 = daily["viral_sum"].quantile(0.75)
    iqr = q3 - q1
    hi = q3 + float(iqr_k) * iqr
    lo = q1 - float(iqr_k) * iqr

    outliers = daily[(daily["viral_sum"] > hi) | (daily["viral_sum"] < lo)][["date_day", "viral_sum"]].copy()
    outliers = outliers.sort_values("viral_sum", ascending=False)
    outlier_dates = [value.date().isoformat() for value in outliers["date_day"].tolist()]

    by_day_channel = (
        kept.groupby(["date_day", "channel_name"], as_index=False)
        .agg(mentions=("message", "size"), viral_sum=("viral_final", "sum"))
    )
    by_day_channel["date"] = pd.to_datetime(by_day_channel["date_day"], utc=True).dt.date
    daily_ctx = (
        by_day_channel.groupby("date", as_index=False)
        .agg(
            channels_n=("channel_name", "nunique"),
            mentions=("mentions", "sum"),
            viral_sum=("viral_sum", "sum"),
            channels=("channel_name", lambda values: sorted({str(v) for v in values if str(v).strip()})),
        )
        .sort_values("date")
    )
    daily_ctx["viral_sum"] = daily_ctx["viral_sum"].round(3)

    lines = []
    for row in daily_ctx.itertuples(index=False):
        channels = ", ".join(row.channels) if row.channels else "канал не указан"
        lines.append(
            f"Дата: {row.date}. "
            f"Писали {int(row.channels_n)} канал(а/ов): {channels}. "
            f"Упоминаний: {int(row.mentions)}. "
            f"Суммарный виральный скор: {row.viral_sum} "
            "(сумма viral_final по найденным новостям дня, каждая новость 0..1)."
        )

    examples = kept.sort_values("score_rrf", ascending=False).head(int(max_example_docs))
    example_lines = []
    for idx, row in enumerate(examples.itertuples(index=False), start=1):
        date_day = str(getattr(row, "date_day", ""))[:10]
        channel = str(getattr(row, "channel_name", ""))
        viral = float(getattr(row, "viral_final", 0.0))
        relevance = int(getattr(row, "judge_relevance", 1))
        text = re.sub(r"\s+", " ", str(getattr(row, "message", ""))).strip()[: int(example_chars)]
        example_lines.append(
            f"{idx}. {date_day}; канал: {channel or 'не указан'}; "
            f"judge_relevance={relevance}; viral_final={viral:.3f}; текст: {text}"
        )

    outlier_block = (
        f"Выделяющиеся даты по суммарной виральности за окно {max_window_days} дней: "
        f"{', '.join(outlier_dates) if outlier_dates else 'нет (по IQR)'}. "
        f"Пороги IQR: hi={hi:.3f}, lo={lo:.3f} (k={iqr_k})."
    )

    return (
        f"Опорная дата: {anchor_date}.\n"
        f"Запрос (текст новости): {message}\n"
        f"Окно: {max_window_days} дней.\n"
        f"Кандидатов после фильтра: {len(kept)}.\n\n"
        f"{outlier_block}\n\n"
        "Дневная динамика:\n"
        + "\n\n".join(lines)
        + "\n\nПримеры самых релевантных найденных сообщений:\n"
        + "\n".join(example_lines)
    )


def build_context_for_row(
    bundle: IndexBundle,
    *,
    message: str,
    anchor_date: str,
    max_window_days: int = 30,
    topN_each: int = 10000,
    tau_sem_hi: float = 0.8,
    tau_sem_lo: float = 0.5,
    tau_bm25_lo: float = 1.0,
    judge_generate: Optional[Callable[[str, Optional[str]], str]] = None,
    judge_keep_threshold: int = 1,
    keep_judge: int = 1,
    iqr_k: float = 1.5,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Retrieve, optionally judge, and build one row's LLM context."""
    raw = retrieve_all_candidates(
        bundle,
        query=message,
        anchor_date=anchor_date,
        max_window_days=max_window_days,
        topN_each=topN_each,
        tau_sem_hi=tau_sem_hi,
        tau_sem_lo=tau_sem_lo,
        tau_bm25_lo=tau_bm25_lo,
    )
    if judge_generate is not None and raw is not None and len(raw) > 0:
        kept = judge_filter_candidates(
            raw,
            query=message,
            judge_generate=judge_generate,
            keep_threshold=judge_keep_threshold,
        )
    else:
        kept = raw.copy() if raw is not None else raw
        if kept is not None and len(kept) > 0 and "judge_relevance" not in kept.columns:
            kept["judge_relevance"] = 1

    context = build_daily_context(
        kept,
        message=message,
        anchor_date=anchor_date,
        max_window_days=max_window_days,
        keep_judge=keep_judge,
        iqr_k=iqr_k,
    )
    return context, raw, kept

