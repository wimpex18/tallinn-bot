"""Per-user communication style analysis and profiling.

Analyses messages to build a style profile per user (formality, slang, profanity,
message length, emoji usage, etc.) and generates a natural-language summary
that gets injected into the Perplexity system prompt so the bot mirrors each
user's tone.
"""

import re
import logging

import httpx

from config import (
    PERPLEXITY_API_KEY,
    STYLE_MIN_MESSAGES,
    STYLE_SUMMARY_TTL,
)

logger = logging.getLogger(__name__)


# ── Signal extraction (pure, no I/O) ────────────────────────────────

def analyze_message_style(text: str) -> dict:
    """Extract communication-style signals from a single message."""
    signals = {}
    signals["uses_emoji"] = bool(re.search(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F900-\U0001F9FF\U00002702-\U000027B0]', text,
    ))
    signals["uses_caps"] = text.isupper() and len(text) > 3
    signals["uses_profanity"] = bool(re.search(
        r'\b(бля|хуй|пизд|сук|нах|ебан|дерьм|блин)\w*', text.lower(),
    ))
    signals["uses_slang"] = bool(re.search(
        r'\b(чел|кста|норм|имхо|лол|кек|хз|ваще|ок|пон|рофл|изи|го)\b', text.lower(),
    ))
    signals["msg_length"] = len(text)
    signals["uses_parenthesis_smileys"] = bool(re.search(r'[)(]{2,}', text))
    return signals


# ── Redis-backed profile (read/write) ───────────────────────────────

async def update_style_counters(redis_client, user_id: int, text: str) -> None:
    """Incrementally update style counters in Redis from a single message."""
    if not redis_client:
        return
    signals = analyze_message_style(text)
    key = f"user:{user_id}:style"
    try:
        pipe = redis_client.pipeline()
        pipe.hincrby(key, "msg_count", 1)
        if signals["uses_emoji"]:
            pipe.hincrby(key, "emoji_count", 1)
        if signals["uses_profanity"]:
            pipe.hincrby(key, "profanity_count", 1)
        if signals["uses_slang"]:
            pipe.hincrby(key, "slang_count", 1)
        if signals["uses_caps"]:
            pipe.hincrby(key, "caps_count", 1)
        pipe.hincrbyfloat(key, "total_msg_length", signals["msg_length"])
        await pipe.execute()
    except Exception as e:
        logger.error(f"Failed to update style counters for user {user_id}: {e}")


async def get_style_summary(redis_client, user_id: int) -> str | None:
    """Build a natural-language style instruction from Redis counters.

    Returns None if too few messages or style is neutral.
    """
    if not redis_client:
        return None

    # Check cached LLM-generated summary first
    cached = await redis_client.get(f"user:{user_id}:style_summary")
    if cached:
        return cached

    data = await redis_client.hgetall(f"user:{user_id}:style")
    if not data:
        return None
    msg_count = int(data.get("msg_count", 0))
    if msg_count < STYLE_MIN_MESSAGES:
        return None

    traits = []
    profanity_rate = int(data.get("profanity_count", 0)) / msg_count
    slang_rate = int(data.get("slang_count", 0)) / msg_count
    emoji_rate = int(data.get("emoji_count", 0)) / msg_count
    avg_len = float(data.get("total_msg_length", 0)) / msg_count

    if profanity_rate > 0.3:
        traits.append("часто матерится — можно отвечать грубовато и с юмором")
    elif profanity_rate > 0.1:
        traits.append("иногда использует мат — допустим неформальный тон")

    if slang_rate > 0.4:
        traits.append("использует сленг и сокращения")

    if emoji_rate > 0.3:
        traits.append("использует эмодзи — можно добавлять в ответ")
    elif emoji_rate < 0.05 and msg_count >= 10:
        traits.append("не использует эмодзи — не добавляй их")

    if avg_len < 30:
        traits.append("пишет коротко — отвечай так же кратко")
    elif avg_len > 150:
        traits.append("пишет развёрнуто — можно отвечать подробнее")

    if not traits:
        return None

    return "Стиль этого пользователя: " + "; ".join(traits) + "."


async def generate_style_summary_llm(
    redis_client, http_client: httpx.AsyncClient,
    user_id: int, user_name: str,
) -> str | None:
    """Use LLM to create a richer style description from recent messages.

    Called periodically by JobQueue (not per-message).
    """
    if not redis_client:
        return None
    recent = await redis_client.lrange(f"user:{user_id}:recent_msgs", 0, 19)
    if len(recent) < STYLE_MIN_MESSAGES:
        return None

    messages_text = "\n".join(f"- {msg}" for msg in recent)
    prompt = (
        f"Проанализируй стиль общения пользователя {user_name} по этим сообщениям:\n\n"
        f"{messages_text}\n\n"
        "Опиши кратко (1-2 предложения) стиль: формальность, мат, сленг, длину "
        "сообщений, настроение, юмор. Формат — прямая инструкция боту, как отвечать "
        "этому пользователю. Если стиль нейтральный — ответь НЕТ."
    )

    try:
        client = http_client or httpx.AsyncClient(timeout=10.0)
        resp = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.2,
            },
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        if "НЕТ" in result.upper() or len(result) < 10:
            return None
        # Cache for 24 h
        await redis_client.set(
            f"user:{user_id}:style_summary", result, ex=STYLE_SUMMARY_TTL,
        )
        logger.info(f"Generated style summary for user {user_id}: {result[:60]}...")
        return result
    except Exception as e:
        logger.error(f"LLM style generation failed for user {user_id}: {e}")
        return None
