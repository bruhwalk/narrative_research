"""Download and save a SentenceTransformer encoder for offline service runs."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "models" / "multilingual-e5-small"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache a SentenceTransformer model locally")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from sentence_transformers import SentenceTransformer

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model)
    model.save(str(output))
    print(f"Saved {args.model} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
