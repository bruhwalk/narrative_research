"""Prompts for context-aware economic narrative annotation."""

from __future__ import annotations


NARRATIVE_SYSTEM_PROMPT = (
    "Ты - эксперт по экономическим новостям и общественному восприятию в России. "
    "Твоя задача - определить, является ли короткая новость экономическим нарративом "
    "для широкой российской аудитории. Ответ должен быть строго валидным JSON."
)


def build_annotation_prompt(message: str, topic: str, llm_context: str) -> str:
    """Build the final context-aware annotation prompt."""
    topic_block = topic.strip() if topic else "не задана"
    context_block = llm_context.strip() if llm_context else "Контекст не найден."

    return (
        "Критерии нарратива:\n\n"
        "1. Релевантность для России: новость должна иметь прямой или косвенный "
        "экономический эффект для жителей России.\n\n"
        "2. Широкий общественный резонанс: фокус на обычных гражданах, а не на узких "
        "профессиональных группах.\n\n"
        "3. Яркость и сила события: резкий рост цен, важное политическое заявление, "
        "масштабные санкции, кризис или похожий сильный инфоповод.\n\n"
        "4. Временная динамика: используй предоставленный Temporal RAG контекст. "
        "Если тема активно появлялась в последние 30 дней, это усиливает нарративный "
        "и вирусный потенциал. Не выдумывай факты вне текста новости и контекста.\n\n"
        "Проанализируй в уме:\n"
        "- есть ли триггерное событие;\n"
        "- есть ли экономический эффект для широкой аудитории в России;\n"
        "- есть ли эмоциональный заряд и простая распространяемая формулировка;\n"
        "- поддерживает ли Temporal RAG контекст резонанс и свежесть темы.\n\n"
        "Поля ответа:\n"
        "- economic_effect: целое число от -2 до 2;\n"
        "- information_resonance: целое число от 1 до 3;\n"
        "- topic_agreement: целое число от 1 до 3;\n"
        "- economic_narrative: строка \"Да\" или \"Нет\";\n"
        "- narrative_strength: целое число от 1 до 3;\n"
        "- comment: короткое объяснение на русском, 1-3 предложения.\n\n"
        "Верни строго один JSON-объект без markdown и без текста вокруг:\n"
        "{\n"
        "  \"economic_effect\": -2,\n"
        "  \"information_resonance\": 1,\n"
        "  \"topic_agreement\": 1,\n"
        "  \"economic_narrative\": \"Да\",\n"
        "  \"narrative_strength\": 1,\n"
        "  \"comment\": \"...\"\n"
        "}\n\n"
        f"Тема (topic): {topic_block}\n\n"
        f"Новость (message):\n{message}\n\n"
        "Temporal RAG контекст:\n"
        f"{context_block}\n"
    )


JUDGE_SYSTEM_PROMPT = """Ты - строгий эксперт по информационному поиску по новостям.

Твоя задача: оценить релевантность кандидатной новости запросу.

Шкала:
2 - кандидат явно про тот же инфоповод/факт/событие.
1 - кандидат связан по теме, но это немного другой инфоповод.
0 - кандидат нерелевантен.

Используй только текст кандидата. Верни строго JSON:
{"relevance": 0|1|2}
"""


def build_judge_prompt(query: str, candidate_text: str, channel: str, date_day: str) -> str:
    """Build a relevance-judge prompt."""
    return (
        f"ЗАПРОС:\n{query}\n\n"
        "КАНДИДАТ:\n"
        f"channel={channel}\n"
        f"date={date_day}\n"
        f"text:\n{candidate_text}\n"
    )

