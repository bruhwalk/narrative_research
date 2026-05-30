"""Small LLM client layer used by the portable inference CLI."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


def strip_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model response."""
    text = strip_text(text)
    if not text:
        raise ValueError("Empty model response")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:500]}")

    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj


def _to_int(value: Any, lo: int, hi: int, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be int, got bool")
    try:
        iv = int(float(value))
    except Exception as exc:
        raise ValueError(f"{field} must be int in [{lo}, {hi}], got {value!r}") from exc
    if iv < lo or iv > hi:
        raise ValueError(f"{field} out of range [{lo}, {hi}]: {iv}")
    return iv


def validate_annotation_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the expected narrative annotation JSON."""
    required = {
        "economic_effect",
        "information_resonance",
        "topic_agreement",
        "economic_narrative",
        "narrative_strength",
        "comment",
    }
    missing = sorted(required - set(obj))
    if missing:
        raise ValueError("Missing keys: " + ", ".join(missing))

    narrative = strip_text(obj.get("economic_narrative"))
    narrative_low = narrative.lower()
    if narrative_low in {"yes", "true", "1", "да"}:
        narrative = "Да"
    elif narrative_low in {"no", "false", "0", "нет"}:
        narrative = "Нет"
    else:
        raise ValueError(f"economic_narrative must be Да/Нет, got {narrative!r}")

    return {
        "economic_effect": _to_int(obj.get("economic_effect"), -2, 2, "economic_effect"),
        "information_resonance": _to_int(obj.get("information_resonance"), 1, 3, "information_resonance"),
        "topic_agreement": _to_int(obj.get("topic_agreement"), 1, 3, "topic_agreement"),
        "economic_narrative": narrative,
        "narrative_strength": _to_int(obj.get("narrative_strength"), 1, 3, "narrative_strength"),
        "comment": strip_text(obj.get("comment")),
    }


def parse_relevance(text: str) -> int:
    """Parse judge relevance JSON."""
    try:
        obj = parse_json_object(text)
        val = int(obj.get("relevance", 0))
        return val if val in (0, 1, 2) else 0
    except Exception:
        match = re.search(r"relevance\"\s*:\s*([012])", strip_text(text))
        return int(match.group(1)) if match else 0


@dataclass
class OllamaConfig:
    model: str
    host: Optional[str] = None
    api_key: Optional[str] = None
    timeout_s: int = 120
    temperature: float = 0.2
    num_predict: int = 2048


class OllamaChatClient:
    """Ollama local/cloud chat client."""

    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        from ollama import Client

        host = self.cfg.host or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_CLOUD_HOST")
        api_key = self.cfg.api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_API_KEY_1")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        model = self.cfg.model
        if host and model.endswith("-cloud"):
            model = model[:-6]

        client = Client(host=host, headers=headers, timeout=self.cfg.timeout_s) if host else Client(headers=headers, timeout=self.cfg.timeout_s)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": self.cfg.temperature, "num_predict": self.cfg.num_predict},
        )
        message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        if isinstance(message, dict):
            return strip_text(message.get("content"))
        if hasattr(message, "content"):
            return strip_text(message.content)
        return strip_text(response)
