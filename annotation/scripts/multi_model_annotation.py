import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _strip(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and pd.isna(s):
        return ""
    return str(s).strip()


def _json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model response")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:2000]}")

    raw = text[start : end + 1]

    raw = re.sub(r"\bNaN\b", "null", raw)
    raw = re.sub(r"\bNone\b", "null", raw)

    return json.loads(raw)


def _clamp_int(v: Any, lo: int, hi: int) -> int:
    try:
        iv = int(v)
    except Exception:
        raise ValueError(f"Expected int in [{lo},{hi}], got: {v!r}")
    return max(lo, min(hi, iv))


def _normalize_yes_no(v: Any) -> str:
    s = _strip(v).lower()
    if s in {"да", "yes", "true", "1"}:
        return "Да"
    if s in {"нет", "no", "false", "0"}:
        return "Нет"
    raise ValueError(f"Expected Да/Нет, got: {v!r}")


def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)

    out["economic_effect"] = _clamp_int(out.get("economic_effect"), -2, 2)
    out["information_resonance"] = _clamp_int(out.get("information_resonance"), 1, 3)
    out["topic_agreement"] = _clamp_int(out.get("topic_agreement"), 1, 3)
    out["narrative_strength"] = _clamp_int(out.get("narrative_strength"), 1, 3)
    out["economic_narrative"] = _normalize_yes_no(out.get("economic_narrative"))
    out["comment"] = _strip(out.get("comment", ""))

    return out


def build_prompt(message: str, topic: str) -> str:
    return (
        "Ты — эксперт по экономическим новостям и общественному восприятию в России. "
        "Твоя задача — определить, является ли короткая новость экономическим нарративом для широкой российской аудитории.\n\n"
        "Критерии нарратива в вашем понимании:\n\n"
        "1. Релевантность для России: Новость должна иметь прямой или косвенный экономический эффект для жителей России. "
        "Мировые новости без последствий для России — не нарратив.\n\n"
        "2. Широкий общественный резонанс: Фокус на обычных гражданах, а не на узких группах. "
        "Нарратив — это яркая новость, которая может вызвать сильный отклик в массах, желание делиться ею "
        "и влиять на импульсивные решения (например, срочные покупки, вывод средств).\n\n"
        "3. Яркость и сила события: В основе нарратива лежит сильное событие (резкий рост цен, важное политическое заявление, "
        "масштабные санкции), а не рутинная информация.\n\n"
        "Проанализируй новость по следующему плану (рассуждения держи в уме, в ответ не выноси):\n\n"
        "[1] Триггер и релевантность:\n"
        "В чём суть новости? Есть ли чёткое триггерное событие (заявление, решение, кризис)?\n"
        "Имеет ли это событие прямые последствия для экономического положения или настроений широких слоёв населения России?\n\n"
        "[2] Эмоциональный заряд и упрощение:\n"
        "Какие эмоции может вызвать текст у обычного человека? (тревога, страх, гнев, оптимизм).\n"
        "Сводится ли основная мысль к простым, обобщающим формулировкам? (Например: «Цены на всё вырастут», «Рубль обвалится», «Наступит дефицит»).\n\n"
        "[3] Логика воздействия и аудитория:\n"
        "Пытается ли новость объяснить, как именно это событие повлияет на жизнь людей, а не просто констатирует факт?\n"
        "Направлена ли новость на массовую аудиторию, а не на профессионалов?\n\n"
        "[4] Источники и резонансный потенциал:\n"
        "Кто является источником или героем новости? (Правительство, ЦБ, известный политик, эксперты в СМИ). "
        "Усиливает ли это авторитетность и потенциальное распространение?\n"
        "Может ли эта новость стать «вирусной» историей для обсуждения в соцсетях и бытовых разговорах?\n\n"
        "Пояснения к полям:\n\n"
        "economic_narrative (Да/Нет): Итоговое решение. Да — если есть триггерное событие, релевантное для широкой российской аудитории, "
        "и новость обладает потенциалом вызвать эмоциональный отклик и массовое обсуждение.\n\n"
        "narrative_strength (1-3): Сила нарративных свойств. Оценивай, насколько текст эмоционален, упрощён и побуждает к действию. "
        "1 — констатация факта; 3 — прямое предупреждение о катастрофических последствиях для всех. "
        "Не нарративные новости в большинстве своем должны получать оценку 1, при этом нарративы тоже могут изредка получать такую оценку.\n\n"
        "economic_effect (-2..2): Влияние на экономическое положение обычного человека в России. Примеры: крах банков: -2, рост в отдельном регионе: -1, отмена санкций: 2.\n\n"
        "topic_agreement (1-3): Твоя оценка правильности выбранной темы (topic). Считай, что тема выбрана из фиксированного списка.\n\n"
        "information_resonance (1-3): Потенциал широкого и эмоционального восприятия в обществе. Высокий резонанс — темы, затрагивающие каждого "
        "(цены на еду, бензин, рубль, важные политические решения для всего населения).\n\n"
        "ВАЖНО: Ответь строго одним JSON-объектом БЕЗ пояснений, БЕЗ markdown и БЕЗ текста вокруг.\n"
        "Схема JSON (ключи строго такие):\n"
        "{\n"
        "  \"economic_effect\": -2,\n"
        "  \"information_resonance\": 1,\n"
        "  \"topic_agreement\": 1,\n"
        "  \"economic_narrative\": \"Да\",\n"
        "  \"narrative_strength\": 1,\n"
        "  \"comment\": \"...\"\n"
        "}\n\n"
        f"Тема (topic): {topic}\n"
        f"Новость (message): {message}\n"
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str  # cloud_ollama | local_ollama | wedlm


def _ollama_client(host: Optional[str]) -> Any:
    from ollama import Client

    headers = None
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}

    if host:
        return Client(host=host, headers=headers)
    return Client(headers=headers) if headers else Client()


def infer_ollama(model: str, prompt: str, host: Optional[str]) -> str:
    client = _ollama_client(host)
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    msg = resp.get("message") or {}
    return _strip(msg.get("content"))


def infer_wedlm(prompt: str) -> str:
    from transformers import AutoTokenizer
    from wedlm import LLM, SamplingParams

    model_name = "tencent/WeDLM-8B-Instruct"
    llm = LLM(model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], SamplingParams(temperature=0.2, max_tokens=512))
    return _strip(outputs[0].get("text"))


def run_inference(
    model_spec: ModelSpec, prompt: str, cloud_host: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
    try:
        t0 = None
        if model_spec.provider in {"local_ollama", "wedlm"}:
            t0 = time.perf_counter()

        if model_spec.provider == "cloud_ollama":
            text = infer_ollama(model_spec.name, prompt, host=cloud_host)
        elif model_spec.provider == "local_ollama":
            text = infer_ollama(model_spec.name, prompt, host=None)
        elif model_spec.provider == "wedlm":
            text = infer_wedlm(prompt)
        else:
            raise ValueError(f"Unknown provider: {model_spec.provider}")

        elapsed_s = None
        if t0 is not None:
            elapsed_s = time.perf_counter() - t0

        payload = _json_from_text(text)
        payload = _validate_payload(payload)
        return payload, None, elapsed_s
    except Exception as e:
        elapsed_s = None
        try:
            if "t0" in locals() and t0 is not None:
                elapsed_s = time.perf_counter() - t0
        except Exception:
            elapsed_s = None

        return None, f"{type(e).__name__}: {e}", elapsed_s


def col(name: str, idx: int) -> str:
    return f"{name} ({idx})"


def ensure_columns(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    required_base = ["message", "topic"]
    for c in required_base:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    metric_cols = [
        "LLm",
        "Экономический эффект",
        "Информационный резонанс",
        "Правильность определения темы",
        "Экономический нарратив",
        "Сила нарратива",
        "Комментарий",
    ]

    for i in range(1, n + 1):
        for m in metric_cols:
            cn = col(m, i)
            if cn not in df.columns:
                df[cn] = ""

    return df


def should_skip_row(df: pd.DataFrame, row_idx: int, slot: int) -> bool:
    llm_cell = _strip(df.at[row_idx, col("LLm", slot)])
    return bool(llm_cell)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Датасет для разметки.xlsx")
    parser.add_argument("--output", default="")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(base_dir, input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    if args.output:
        out_path = args.output
        if not os.path.isabs(out_path):
            out_path = os.path.join(base_dir, out_path)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(os.path.basename(input_path))
        out_path = os.path.join(base_dir, f"{name}_filled_{stamp}{ext}")

    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(out_path)

    cloud_host = os.getenv("OLLAMA_CLOUD_HOST")
    if cloud_host:
        cloud_host = cloud_host.strip()

    models: List[ModelSpec] = [
        ModelSpec("gpt-oss:20b-cloud", "cloud_ollama"),
        ModelSpec("gpt-oss:120b-cloud", "cloud_ollama"),
        ModelSpec("qwen3-vl:235b-instruct-cloud", "cloud_ollama"),
        ModelSpec("qwen3:8b", "local_ollama"),
        ModelSpec("tencent/WeDLM-8B-Instruct", "wedlm"),
    ]

    df = pd.read_excel(input_path, sheet_name=args.sheet)
    df = ensure_columns(df, n=len(models))

    total = len(df)
    start = max(0, min(total, args.start_row))
    end = total if args.max_rows <= 0 else min(total, start + args.max_rows)

    for r in range(start, end):
        message = _strip(df.at[r, "message"])
        topic = _strip(df.at[r, "topic"])

        if not message:
            continue

        prompt = build_prompt(message=message, topic=topic)

        for slot, model_spec in enumerate(models, start=1):
            if should_skip_row(df, r, slot):
                continue

            payload, err, elapsed_s = run_inference(model_spec, prompt, cloud_host=cloud_host)

            df.at[r, col("LLm", slot)] = model_spec.name
            if payload is not None:
                df.at[r, col("Экономический эффект", slot)] = payload["economic_effect"]
                df.at[r, col("Информационный резонанс", slot)] = payload["information_resonance"]
                df.at[r, col("Правильность определения темы", slot)] = payload["topic_agreement"]
                df.at[r, col("Экономический нарратив", slot)] = payload["economic_narrative"]
                df.at[r, col("Сила нарратива", slot)] = payload["narrative_strength"]
                df.at[r, col("Комментарий", slot)] = payload["comment"]
            else:
                df.at[r, col("Комментарий", slot)] = err or "Unknown error"

            time_part = ""
            if elapsed_s is not None:
                time_part = f" | {elapsed_s:.2f}s"
            print(
                f"Row {r+1}/{total} | slot {slot} | {model_spec.name} | {'OK' if payload else 'ERR'}{time_part}"
            )

    df.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
