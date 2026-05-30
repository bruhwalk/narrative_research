"""Tiny retrieval HTTP service over precomputed notebook embeddings.

The service keeps the document embeddings and FAISS index in memory. If a local
SentenceTransformer encoder is available, arbitrary input news are encoded as
E5-style queries (`query: ...`) and searched against document vectors that were
created with the matching `passage: ...` prefix.
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .llm_client import OllamaChatClient, OllamaConfig, parse_json_object, validate_annotation_payload
from .prompts import NARRATIVE_SYSTEM_PROMPT, build_annotation_prompt
from .retrieval import build_daily_context, tokenize_ru


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_DIR = ROOT / "temporal_retrieve_virality_signals" / "notebooks" / "embeddings"
DEFAULT_CORPUS = ROOT / "temporal_retrieve_virality_signals" / "data" / "dataset_tg_economic.parquet"

EMBEDDING_FILES = {
    "e5-small": "emb_e5_small_fp16.npy",
    "e5-base": "emb_e5_base_fp16.npy",
    "e5-large": "emb_e5_large_fp16.npy",
    "minilm": "emb_minilm_fp16.npy",
    "gte": "emb_gte_fp16.npy",
}

DEFAULT_ENCODERS = {
    "e5-small": "intfloat/multilingual-e5-small",
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-large": "intfloat/multilingual-e5-large",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "gte": "Alibaba-NLP/gte-multilingual-base",
}

QUERY_PREFIXES = {
    "e5-small": "query: ",
    "e5-base": "query: ",
    "e5-large": "query: ",
    "minilm": "",
    "gte": "",
}


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _load_query_encoder(
    *,
    embedding_key: str,
    encoder_path: Optional[str],
    encoder_name: Optional[str],
    device: str,
) -> tuple[Any | None, str | None]:
    model_ref = encoder_path or encoder_name or DEFAULT_ENCODERS.get(embedding_key)
    if not model_ref:
        return None, None

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")

    return SentenceTransformer(model_ref, device=device), model_ref


class PrecomputedRetrievalIndex:
    def __init__(
        self,
        *,
        embeddings_dir: str | Path = DEFAULT_EMBEDDINGS_DIR,
        corpus_path: str | Path = DEFAULT_CORPUS,
        embedding_key: str = "e5-small",
        encoder_path: Optional[str] = None,
        encoder_name: Optional[str] = None,
        query_prefix: Optional[str] = None,
        device: str = "cpu",
        load_query_encoder: bool = True,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.corpus_path = Path(corpus_path)
        self.embedding_key = embedding_key
        self.query_prefix = QUERY_PREFIXES.get(embedding_key, "") if query_prefix is None else query_prefix
        self.query_encoder: Any | None = None
        self.query_encoder_ref: str | None = None

        emb_file = EMBEDDING_FILES.get(embedding_key, embedding_key)
        emb_path = self.embeddings_dir / emb_file
        rowmap_path = self.embeddings_dir / "rowmap.csv"
        if not emb_path.exists():
            raise FileNotFoundError(emb_path)
        if not rowmap_path.exists():
            raise FileNotFoundError(rowmap_path)

        rowmap = pd.read_csv(rowmap_path)
        corpus = pd.read_parquet(
            self.corpus_path,
            columns=["message_id", "message", "date", "id_channel", "viral_final", "topic"],
        )
        df = rowmap.merge(corpus, on="message_id", how="left", suffixes=("_rowmap", ""))
        if df["message"].isna().any():
            missing = int(df["message"].isna().sum())
            raise ValueError(f"Failed to join {missing} rowmap rows to corpus by message_id")

        df["message"] = df["message"].fillna("").astype(str)
        df["date_day"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.normalize()
        df["channel_name"] = df["id_channel"].astype(str)
        df["viral_final"] = pd.to_numeric(df["viral_final"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        df["topic"] = df["topic"].fillna("").astype(str)
        self.df = df.reset_index(drop=True)

        emb = np.load(emb_path, mmap_mode=None).astype(np.float32)
        if emb.shape[0] != len(self.df):
            raise ValueError(f"Embedding rows {emb.shape[0]} != rowmap rows {len(self.df)}")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        self.embeddings = np.ascontiguousarray(emb, dtype=np.float32)

        self.faiss_index = faiss.IndexFlatIP(int(self.embeddings.shape[1]))
        self.faiss_index.add(self.embeddings)

        self.bm25 = BM25Okapi([tokenize_ru(text) for text in self.df["message"].tolist()])
        self.id_to_row = {
            str(message_id): int(i)
            for i, message_id in enumerate(self.df["message_id"].astype(str).tolist())
        }

        if load_query_encoder:
            self.query_encoder, self.query_encoder_ref = _load_query_encoder(
                embedding_key=embedding_key,
                encoder_path=encoder_path,
                encoder_name=encoder_name,
                device=device,
            )

    def encode_query(self, message: str) -> np.ndarray:
        if self.query_encoder is None:
            raise RuntimeError("query encoder is not loaded")
        text = self.query_prefix + _clean_text(message)
        vector = self.query_encoder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vector.astype(np.float32))

    def retrieve(
        self,
        *,
        message: str = "",
        message_id: Optional[str] = None,
        anchor_date: str,
        window_days: int = 30,
        top_k: int = 30,
        topN_each: int = 300,
    ) -> pd.DataFrame:
        anchor = pd.to_datetime(anchor_date, utc=True).normalize()
        dates = pd.to_datetime(self.df["date_day"], errors="coerce", utc=True).dt.normalize()
        age = (anchor - dates).dt.days
        allowed = (dates <= anchor) & (age >= 0) & (age <= int(window_days))
        allowed_np = allowed.to_numpy(dtype=bool)

        dense_rank: dict[int, int] = {}
        dense_scores: dict[int, float] = {}
        if message and self.query_encoder is not None:
            q = self.encode_query(message)
            scores, idx = self.faiss_index.search(q, int(min(topN_each, len(self.df))))
            keep = allowed_np[idx[0]]
            dense_idx = idx[0][keep]
            dense_sc = scores[0][keep]
            dense_rank = {int(rowpos): rank for rank, rowpos in enumerate(dense_idx, start=1)}
            dense_scores = {int(rowpos): float(score) for rowpos, score in zip(dense_idx, dense_sc)}
        elif message_id is not None and str(message_id) in self.id_to_row:
            row = self.id_to_row[str(message_id)]
            q = self.embeddings[row : row + 1]
            scores, idx = self.faiss_index.search(q, int(min(topN_each, len(self.df))))
            keep = allowed_np[idx[0]]
            dense_idx = idx[0][keep]
            dense_sc = scores[0][keep]
            dense_rank = {int(rowpos): rank for rank, rowpos in enumerate(dense_idx, start=1)}
            dense_scores = {int(rowpos): float(score) for rowpos, score in zip(dense_idx, dense_sc)}

        bm_rank: dict[int, int] = {}
        bm_scores = np.full(len(self.df), -np.inf, dtype=np.float32)
        if message:
            bm_scores = self.bm25.get_scores(tokenize_ru(message)).astype(np.float32)
            bm_scores[~allowed_np] = -np.inf
            if np.isfinite(bm_scores).any():
                k = min(int(topN_each), len(bm_scores))
                bm_idx = np.argpartition(-bm_scores, k - 1)[:k]
                bm_idx = bm_idx[np.argsort(-bm_scores[bm_idx])]
                bm_rank = {int(rowpos): rank for rank, rowpos in enumerate(bm_idx, start=1)}

        union = np.array(sorted(set(dense_rank) | set(bm_rank)), dtype=int)
        if len(union) == 0:
            return self.df.iloc[[]].copy()

        k_rrf = 60.0
        score = np.zeros(len(union), dtype=np.float32)
        for pos, rowpos in enumerate(union):
            if int(rowpos) in dense_rank:
                score[pos] += 1.0 / (k_rrf + dense_rank[int(rowpos)])
            if int(rowpos) in bm_rank:
                score[pos] += 1.0 / (k_rrf + bm_rank[int(rowpos)])

        out = self.df.iloc[union].copy()
        out["_rowpos"] = union
        out["score_rrf"] = score
        out["rank_dense"] = out["_rowpos"].map(lambda rp: dense_rank.get(int(rp), np.nan))
        out["rank_bm25"] = out["_rowpos"].map(lambda rp: bm_rank.get(int(rp), np.nan))
        out["dense_score"] = out["_rowpos"].map(lambda rp: dense_scores.get(int(rp), np.nan))
        out["bm25_score"] = out["_rowpos"].map(lambda rp: float(bm_scores[int(rp)]) if np.isfinite(bm_scores[int(rp)]) else np.nan)
        out["age_days"] = (anchor - pd.to_datetime(out["date_day"], utc=True)).dt.days
        out["judge_relevance"] = 1

        if "viral_final" in out.columns:
            boost = 0.05 * (out["viral_final"].astype(float).to_numpy() ** 5)
            out["score_rrf"] = out["score_rrf"] + boost

        return out.sort_values("score_rrf", ascending=False).head(int(top_k)).reset_index(drop=True)

    def retrieve_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = _clean_text(payload.get("message", ""))
        message_id = payload.get("message_id")
        anchor_date = str(payload.get("anchor_date") or "")[:10]
        if not anchor_date:
            raise ValueError("anchor_date is required")

        top_k = int(payload.get("top_k", 30))
        window_days = int(payload.get("window_days", 30))
        topN_each = int(payload.get("topN_each", 300))
        cand = self.retrieve(
            message=message,
            message_id=str(message_id) if message_id is not None else None,
            anchor_date=anchor_date,
            window_days=window_days,
            top_k=top_k,
            topN_each=topN_each,
        )
        query_text = message
        if not query_text and message_id is not None and str(message_id) in self.id_to_row:
            query_text = str(self.df.iloc[self.id_to_row[str(message_id)]]["message"])

        context = build_daily_context(
            cand,
            message=query_text,
            anchor_date=anchor_date,
            max_window_days=window_days,
            keep_judge=1,
        )
        candidate_cols = [
            "message_id",
            "date",
            "topic",
            "channel_name",
            "viral_final",
            "score_rrf",
            "rank_dense",
            "rank_bm25",
            "dense_score",
            "bm25_score",
            "message",
        ]
        candidates = cand[[c for c in candidate_cols if c in cand.columns]].copy()
        if message and self.query_encoder is not None:
            mode = "dense_by_query_encoder+bm25"
        elif message_id is not None and str(message_id) in self.id_to_row:
            mode = "dense_by_message_id+bm25"
        elif message:
            mode = "bm25_only"
        else:
            mode = "empty_query"
        return {
            "embedding_key": self.embedding_key,
            "query_encoder": self.query_encoder_ref,
            "query_prefix": self.query_prefix,
            "mode": mode,
            "count": int(len(candidates)),
            "context": context,
            "candidates": json.loads(candidates.to_json(orient="records", force_ascii=False)),
        }

    def annotate_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = _clean_text(payload.get("model") or payload.get("llm_model") or "")
        if not model:
            raise ValueError("model or llm_model is required for /annotate")

        retrieval = self.retrieve_response(payload)
        message = _clean_text(payload.get("message", ""))
        message_id = payload.get("message_id")
        if not message and message_id is not None and str(message_id) in self.id_to_row:
            message = str(self.df.iloc[self.id_to_row[str(message_id)]]["message"])
        if not message:
            raise ValueError("message or known message_id is required for /annotate")

        topic = _clean_text(payload.get("topic", ""))
        prompt = build_annotation_prompt(message, topic, retrieval["context"])
        client = OllamaChatClient(
            OllamaConfig(
                model=model,
                host=_clean_text(payload.get("ollama_host", "")) or None,
                api_key=_clean_text(payload.get("ollama_api_key", "")) or None,
                timeout_s=int(payload.get("timeout_s", 120)),
                temperature=float(payload.get("temperature", 0.2)),
                num_predict=int(payload.get("num_predict", 2048)),
            )
        )
        raw_response = client.generate(prompt, system=NARRATIVE_SYSTEM_PROMPT)
        parsed = validate_annotation_payload(parse_json_object(raw_response))
        return {
            "model": model,
            "annotation": parsed,
            "llm_raw_response": raw_response,
            "retrieval": retrieval,
        }


class RetrievalHandler(BaseHTTPRequestHandler):
    index: PrecomputedRetrievalIndex

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "rows": int(len(self.index.df)),
                    "embedding_key": self.index.embedding_key,
                    "dim": int(self.index.embeddings.shape[1]),
                    "query_encoder_ready": self.index.query_encoder is not None,
                    "query_encoder": self.index.query_encoder_ref,
                    "query_prefix": self.index.query_prefix,
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in {"/retrieve", "/annotate"}:
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
            if self.path == "/retrieve":
                self._send_json(200, self.index.retrieve_response(payload))
            else:
                self._send_json(200, self.index.annotate_response(payload))
        except Exception as exc:
            self._send_json(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny retrieval service over precomputed embeddings")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--embeddings-dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--embedding-key", default="e5-small", choices=sorted(EMBEDDING_FILES))
    parser.add_argument("--encoder-path", default=None, help="Local SentenceTransformer directory for query encoding")
    parser.add_argument("--encoder-name", default=None, help="Hugging Face/SentenceTransformer model name")
    parser.add_argument("--query-prefix", default=None, help="Override query prefix; E5 defaults to 'query: '")
    parser.add_argument("--device", default="cpu", help="SentenceTransformer device, e.g. cpu, cuda, mps")
    parser.add_argument("--no-query-encoder", action="store_true", help="Disable arbitrary text dense retrieval")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    index = PrecomputedRetrievalIndex(
        embeddings_dir=args.embeddings_dir,
        corpus_path=args.corpus,
        embedding_key=args.embedding_key,
        encoder_path=args.encoder_path,
        encoder_name=args.encoder_name,
        query_prefix=args.query_prefix,
        device=args.device,
        load_query_encoder=not args.no_query_encoder,
    )
    RetrievalHandler.index = index
    server = HTTPServer((args.host, args.port), RetrievalHandler)
    print(
        f"Serving retrieval on http://{args.host}:{args.port} "
        f"rows={len(index.df)} embedding={args.embedding_key} dim={index.embeddings.shape[1]} "
        f"query_encoder={index.query_encoder_ref or 'disabled'}"
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
