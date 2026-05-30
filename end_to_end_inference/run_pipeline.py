"""CLI for end-to-end temporal context retrieval and narrative inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .io_utils import prepare_corpus, prepare_input_news, read_table, write_table
from .llm_client import OllamaChatClient, OllamaConfig, parse_json_object, validate_annotation_payload
from .prompts import NARRATIVE_SYSTEM_PROMPT, build_annotation_prompt
from .retrieval import build_context_for_row, build_index, load_index
from .virality import attach_viral_final


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "temporal_retrieve_virality_signals" / "data" / "cleaned_news.csv"
DEFAULT_VIRALITY_SOURCE = PROJECT_ROOT / "temporal_retrieve_virality_signals" / "data" / "dataset_tg_economic.parquet"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "end_to_end_inference" / "indexes" / "e5_large"


def _add_shared_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directory with FAISS/BM25 artifacts")
    parser.add_argument("--encoder-name", default="intfloat/multilingual-e5-large", help="SentenceTransformer encoder")
    parser.add_argument("--device", default="cpu", help="Encoder device: cpu, cuda, mps")


def _add_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Historical news corpus path")
    parser.add_argument("--corpus-sheet", default=0, help="Sheet name/index for XLSX corpus")
    parser.add_argument("--corpus-text-col", default="message")
    parser.add_argument("--corpus-date-col", default="date_day")
    parser.add_argument("--corpus-channel-col", default="channel_name")
    parser.add_argument("--corpus-id-col", default="message_id")
    parser.add_argument("--corpus-viral-col", default="", help="Optional virality score column in corpus")
    parser.add_argument(
        "--virality-source",
        default=str(DEFAULT_VIRALITY_SOURCE),
        help="Engagement table used to attach/compute viral_final during index build",
    )
    parser.add_argument("--virality-sheet", default=0, help="Sheet name/index for XLSX virality source")
    parser.add_argument("--virality-id-col", default="message_id")
    parser.add_argument("--virality-channel-col", default="id_channel")
    parser.add_argument("--virality-text-col", default="message")
    parser.add_argument("--no-compute-virality", action="store_true", help="Do not attach/compute viral_final at build-index")
    parser.add_argument("--recompute-virality", action="store_true", help="Recompute viral_final even if source already has it")
    parser.add_argument("--require-virality", action="store_true", help="Fail if virality source is missing")


def cmd_build_index(args: argparse.Namespace) -> int:
    corpus_sheet: str | int | None = args.corpus_sheet
    if isinstance(corpus_sheet, str) and corpus_sheet.isdigit():
        corpus_sheet = int(corpus_sheet)

    print(f"Reading corpus: {args.corpus}")
    raw = read_table(args.corpus, sheet=corpus_sheet)
    corpus = prepare_corpus(
        raw,
        text_col=args.corpus_text_col,
        date_col=args.corpus_date_col,
        channel_col=args.corpus_channel_col or None,
        id_col=args.corpus_id_col or None,
        viral_col=args.corpus_viral_col or None,
    )
    if not args.no_compute_virality:
        virality_sheet: str | int | None = args.virality_sheet
        if isinstance(virality_sheet, str) and virality_sheet.isdigit():
            virality_sheet = int(virality_sheet)
        before_nonzero = int((pd.to_numeric(corpus["viral_final"], errors="coerce").fillna(0.0) > 0).sum())
        corpus = attach_viral_final(
            corpus,
            virality_source=args.virality_source,
            corpus_id_col="message_id",
            source_id_col=args.virality_id_col,
            source_channel_col=args.virality_channel_col,
            corpus_text_col="message",
            source_text_col=args.virality_text_col,
            recompute=args.recompute_virality,
            require=args.require_virality,
            sheet=virality_sheet,
        )
        after_nonzero = int((pd.to_numeric(corpus["viral_final"], errors="coerce").fillna(0.0) > 0).sum())
        print(
            "Virality attached: "
            f"nonzero viral_final rows {before_nonzero} -> {after_nonzero}; "
            f"mean={corpus['viral_final'].mean():.4f}, max={corpus['viral_final'].max():.4f}"
        )
    print(f"Prepared corpus rows: {len(corpus)}")
    print(f"Building index: {args.index_dir}")
    build_index(
        corpus,
        index_dir=args.index_dir,
        encoder_name=args.encoder_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    print("Index build complete")
    return 0


def _make_ollama(args: argparse.Namespace, *, model: str) -> OllamaChatClient:
    return OllamaChatClient(
        OllamaConfig(
            model=model,
            host=args.ollama_host or None,
            api_key=args.ollama_api_key or None,
            timeout_s=args.ollama_timeout_s,
            temperature=args.temperature,
            num_predict=args.num_predict,
        )
    )


def _annotation_columns() -> list[str]:
    return [
        "economic_effect",
        "information_resonance",
        "topic_agreement",
        "economic_narrative",
        "narrative_strength",
        "comment",
    ]


def _safe_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


def cmd_infer(args: argparse.Namespace) -> int:
    input_sheet: str | int | None = args.input_sheet
    if isinstance(input_sheet, str) and input_sheet.isdigit():
        input_sheet = int(input_sheet)

    index_paths = list(Path(args.index_dir).glob("faiss__*.index"))
    if not index_paths and args.build_index_if_missing:
        build_args = argparse.Namespace(**vars(args))
        cmd_build_index(build_args)

    print(f"Loading index: {args.index_dir}")
    bundle = load_index(index_dir=args.index_dir, encoder_name=args.encoder_name, device=args.device)

    print(f"Reading input: {args.input}")
    input_df = read_table(args.input, sheet=input_sheet)
    news = prepare_input_news(
        input_df,
        text_col=args.text_col,
        date_col=args.date_col,
        topic_col=args.topic_col or None,
        id_col=args.id_col or None,
    )

    if args.max_rows > 0:
        news = news.head(args.max_rows).copy()

    llm: Optional[OllamaChatClient] = None
    if not args.context_only:
        llm = _make_ollama(args, model=args.model)

    judge_llm: Optional[OllamaChatClient] = None
    if args.judge_model:
        judge_llm = _make_ollama(args, model=args.judge_model)

    def judge_generate(prompt: str, system: Optional[str]) -> str:
        if judge_llm is None:
            raise RuntimeError("judge_model is not configured")
        return judge_llm.generate(prompt, system=system)

    output = news.copy()
    output["llm_context"] = ""
    output["retrieved_candidates_n"] = 0
    output["kept_candidates_n"] = 0
    output["llm_raw_response"] = ""
    output["inference_error"] = ""
    for col in _annotation_columns():
        output[col] = ""

    total = len(output)
    for pos, idx in enumerate(output.index, start=1):
        message = str(output.at[idx, "_message"])
        topic = str(output.at[idx, "_topic"])
        anchor_date = str(output.at[idx, "_anchor_date"])
        row_label = str(output.at[idx, "_source_id"])
        t0 = time.perf_counter()

        try:
            context, raw_candidates, kept_candidates = build_context_for_row(
                bundle,
                message=message,
                anchor_date=anchor_date,
                max_window_days=args.max_window_days,
                topN_each=args.topN_each,
                tau_sem_hi=args.tau_sem_hi,
                tau_sem_lo=args.tau_sem_lo,
                tau_bm25_lo=args.tau_bm25_lo,
                judge_generate=judge_generate if judge_llm is not None else None,
                judge_keep_threshold=args.judge_keep_threshold,
                keep_judge=args.keep_judge,
                iqr_k=args.iqr_k,
            )
            output.at[idx, "llm_context"] = context
            output.at[idx, "retrieved_candidates_n"] = 0 if raw_candidates is None else len(raw_candidates)
            output.at[idx, "kept_candidates_n"] = 0 if kept_candidates is None else len(kept_candidates)

            if llm is not None:
                prompt = build_annotation_prompt(message=message, topic=topic, llm_context=context)
                raw = llm.generate(prompt, system=NARRATIVE_SYSTEM_PROMPT)
                output.at[idx, "llm_raw_response"] = raw
                payload = validate_annotation_payload(parse_json_object(raw))
                for col, value in payload.items():
                    output.at[idx, col] = value

            elapsed = time.perf_counter() - t0
            print(
                f"[{pos}/{total}] row={row_label} ok "
                f"retrieved={output.at[idx, 'retrieved_candidates_n']} "
                f"kept={output.at[idx, 'kept_candidates_n']} "
                f"{elapsed:.1f}s"
            )
        except Exception as exc:
            output.at[idx, "inference_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{pos}/{total}] row={row_label} error: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise

    hidden_cols = ["_input_row_id", "_message", "_anchor_date", "_topic", "_source_id"]
    result = output.drop(columns=[col for col in hidden_cols if col in output.columns])
    write_table(result, args.output)
    print(f"Saved: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end Temporal RAG context retrieval and narrative inference"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-index", help="Build FAISS + BM25 index over historical news")
    _add_shared_index_args(build)
    _add_corpus_args(build)
    build.add_argument("--batch-size", type=int, default=64)
    build.set_defaults(func=cmd_build_index)

    infer = sub.add_parser("infer", help="Run context retrieval and optional LLM annotation")
    _add_shared_index_args(infer)
    _add_corpus_args(infer)
    infer.add_argument("--input", required=True, help="Input news table")
    infer.add_argument("--output", required=True, help="Output table path")
    infer.add_argument("--input-sheet", default=0, help="Sheet name/index for XLSX input")
    infer.add_argument("--text-col", default="message")
    infer.add_argument("--date-col", default="date")
    infer.add_argument("--topic-col", default="topic")
    infer.add_argument("--id-col", default="")
    infer.add_argument("--max-rows", type=int, default=0)
    infer.add_argument("--build-index-if-missing", action="store_true")
    infer.add_argument("--batch-size", type=int, default=64)

    infer.add_argument("--max-window-days", type=int, default=30)
    infer.add_argument("--topN-each", dest="topN_each", type=int, default=10000)
    infer.add_argument("--tau-sem-hi", type=float, default=0.8)
    infer.add_argument("--tau-sem-lo", type=float, default=0.5)
    infer.add_argument("--tau-bm25-lo", type=float, default=1.0)
    infer.add_argument("--iqr-k", type=float, default=1.5)

    infer.add_argument("--context-only", action="store_true", help="Only add llm_context, skip final LLM annotation")
    infer.add_argument("--model", default="qwen3-vl:235b-instruct-cloud", help="Ollama model for final annotation")
    infer.add_argument("--judge-model", default="", help="Optional Ollama model for relevance judge")
    infer.add_argument("--judge-keep-threshold", type=int, default=1)
    infer.add_argument("--keep-judge", type=int, default=1)
    infer.add_argument("--ollama-host", default="", help="Ollama host, e.g. https://ollama.com or http://localhost:11434")
    infer.add_argument("--ollama-api-key", default="", help="Ollama API key; can also use OLLAMA_API_KEY")
    infer.add_argument("--ollama-timeout-s", type=int, default=120)
    infer.add_argument("--temperature", type=float, default=0.2)
    infer.add_argument("--num-predict", type=int, default=2048)
    infer.add_argument("--continue-on-error", action="store_true")
    infer.set_defaults(func=cmd_infer)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
