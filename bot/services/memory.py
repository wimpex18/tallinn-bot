"""Redis-backed persistent memory for user and group facts."""

import time
import re
import logging

import httpx

from config import PERPLEXITY_API_KEY

logger = logging.getLogger(__name__)

# Initialized in main.py post_init
redis_client = None
http_client: httpx.AsyncClient = None


async def save_user_fact(user_id: int, fact: str) -> None:
    """Save a fact about a user (sorted set, newest kept, max 20)."""
    if not redis_client:
        return
    try:
        key = f"user:{user_id}:facts"
        await redis_client.zadd(key, {fact: time.time()})
        count = await redis_client.zcard(key)
        if count > 20:
            await redis_client.zremrangebyrank(key, 0, -(20 + 1))
    except Exception as e:
        logger.error(f"Failed to save user fact: {e}")


async def get_user_facts(user_id: int) -> list[str]:
    """Get all facts about a user (ordered oldest→newest)."""
    if not redis_client:
        return []
    try:
        return await redis_client.zrange(f"user:{user_id}:facts", 0, -1)
    except Exception as e:
        logger.error(f"Failed to get user facts: {e}")
        return []


async def save_group_fact(chat_id: int, fact: str) -> None:
    """Save a fact about the group (sorted set, newest kept, max 30)."""
    if not redis_client:
        return
    try:
        key = f"group:{chat_id}:facts"
        await redis_client.zadd(key, {fact: time.time()})
        count = await redis_client.zcard(key)
        if count > 30:
            await redis_client.zremrangebyrank(key, 0, -(30 + 1))
    except Exception as e:
        logger.error(f"Failed to save group fact: {e}")


async def get_group_facts(chat_id: int) -> list[str]:
    """Get all facts about the group (ordered oldest→newest)."""
    if not redis_client:
        return []
    try:
        return await redis_client.zrange(f"group:{chat_id}:facts", 0, -1)
    except Exception as e:
        logger.error(f"Failed to get group facts: {e}")
        return []


async def save_user_interaction(user_id: int, user_name: str, username: str) -> None:
    """Save info about a user who interacted with the bot."""
    if not redis_client or not user_name:
        return
    try:
        from datetime import datetime
        key = f"user:{user_id}:profile"
        await redis_client.hset(key, mapping={
            "name": user_name,
            "username": username or "",
            "last_seen": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Failed to save user interaction: {e}")


def extract_facts_from_response(question: str, answer: str, user_name: str) -> list[str]:
    """Extract memorable facts from a conversation using regex patterns."""
    facts = []
    patterns = [
        (r"люблю\s+(\w+)", "любит {}"),
        (r"нравится\s+(\w+)", "нравится {}"),
        (r"не люблю\s+(\w+)", "не любит {}"),
        (r"не ем\s+(\w+)", "не ест {}"),
        (r"работаю\s+(.+?)(?:\.|$)", "работает {}"),
        (r"живу\s+(.+?)(?:\.|$)", "живёт {}"),
    ]
    for pattern, fact_template in patterns:
        match = re.search(pattern, question.lower())
        if match:
            fact = fact_template.format(match.group(1))
            if user_name:
                fact = f"{user_name} {fact}"
            facts.append(fact)
    return facts


async def smart_extract_facts(
    question: str, answer: str, user_name: str, chat_context: str = None,
) -> list[str]:
    """Use LLM to extract important facts from conversation."""
    if not question or len(question) < 10:
        return []

    context_part = f"Контекст чата: {chat_context}" if chat_context else ""
    prompt = f"""Извлеки важные факты о пользователе из этого диалога.
Пользователь: {user_name or 'unknown'}

Вопрос: {question}
Ответ: {answer}

{context_part}

Выдай ТОЛЬКО факты о человеке (интересы, предпочтения, планы, работа, и т.д.)
Формат: один факт на строку, коротко (3-7 слов)
Если фактов нет - напиши "НЕТ"
Максимум 3 факта."""

    try:
        client = http_client or httpx.AsyncClient(timeout=10.0)
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.1,
            },
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()

        if "НЕТ" in result.upper() or len(result) < 5:
            return []

        facts = []
        for line in result.split("\n"):
            line = line.strip().lstrip("-•").strip()
            if 3 < len(line) < 100:
                if user_name and not line.startswith(user_name):
                    line = f"{user_name}: {line}"
                facts.append(line)
        return facts[:3]
    except Exception as e:
        logger.error(f"Failed to extract facts: {e}")
        return []
