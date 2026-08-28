"""Redis-backed persistent memory for user and group facts.

Every write path refreshes a whole-key TTL (REDIS_KEY_TTL_DAYS) instead of
relying on a periodic full-keyspace scan to notice staleness — Redis expires
untouched keys on its own, so there is no separate cleanup job to run.
"""

import json
import logging
import random
import re
import time
from datetime import UTC

from config import MISTRAL_MODEL, REDIS_KEY_TTL_DAYS, STYLE_RECENT_MESSAGES_KEPT

logger = logging.getLogger(__name__)

# Initialized in main.py post_init
redis_client = None

_KEY_TTL_SECONDS = REDIS_KEY_TTL_DAYS * 86400


async def save_user_fact(user_id: int, fact: str) -> None:
    """Save a fact about a user (sorted set, newest kept, max 20)."""
    if not redis_client:
        return
    try:
        key = f"user:{user_id}:facts"
        pipe = redis_client.pipeline()
        pipe.zadd(key, {fact: time.time()})
        pipe.expire(key, _KEY_TTL_SECONDS)
        await pipe.execute()
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
        pipe = redis_client.pipeline()
        pipe.zadd(key, {fact: time.time()})
        pipe.expire(key, _KEY_TTL_SECONDS)
        await pipe.execute()
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


# ── Saved quotes (per group, on-demand only — never posted unprompted) ──

_MAX_QUOTES_KEPT = 100


async def save_quote(chat_id: int, quote: str) -> None:
    """Save a memorable quote for the group (sorted set, newest kept, max 100)."""
    if not redis_client or not quote:
        return
    try:
        key = f"group:{chat_id}:quotes"
        pipe = redis_client.pipeline()
        pipe.zadd(key, {quote: time.time()})
        pipe.expire(key, _KEY_TTL_SECONDS)
        await pipe.execute()
        count = await redis_client.zcard(key)
        if count > _MAX_QUOTES_KEPT:
            await redis_client.zremrangebyrank(key, 0, -(_MAX_QUOTES_KEPT + 1))
    except Exception as e:
        logger.error(f"Failed to save quote: {e}")


async def get_random_quote(chat_id: int) -> str | None:
    """Get one random saved quote for the group, or None if none are saved."""
    if not redis_client:
        return None
    try:
        key = f"group:{chat_id}:quotes"
        count = await redis_client.zcard(key)
        if not count:
            return None
        idx = random.randint(0, count - 1)
        result = await redis_client.zrange(key, idx, idx)
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Failed to get random quote: {e}")
        return None


async def get_all_quotes(chat_id: int, limit: int = 10) -> list[str]:
    """Get saved quotes for the group, newest first, capped at `limit`."""
    if not redis_client:
        return []
    try:
        return await redis_client.zrevrange(f"group:{chat_id}:quotes", 0, limit - 1)
    except Exception as e:
        logger.error(f"Failed to get quotes: {e}")
        return []


async def save_user_interaction(user_id: int, user_name: str, username: str) -> None:
    """Save info about a user who interacted with the bot."""
    if not redis_client or not user_name:
        return
    try:
        from datetime import datetime
        key = f"user:{user_id}:profile"
        pipe = redis_client.pipeline()
        pipe.hset(key, mapping={
            "name": user_name,
            "username": username or "",
            "last_seen": datetime.now(UTC).isoformat(),
        })
        pipe.expire(key, _KEY_TTL_SECONDS)
        await pipe.execute()
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
    """Use LLM to extract important facts from conversation.

    Returns structured JSON output for reliable parsing instead of fragile
    line-by-line text parsing.
    """
    if not question or len(question) < 10:
        return []

    from bot.services import ai as ai_service
    if not ai_service.mistral_client:
        return []

    context_part = f"Контекст чата: {chat_context}" if chat_context else ""
    prompt = f"""Извлеки важные факты о пользователе из этого диалога.
Пользователь: {user_name or 'unknown'}

Вопрос: {question}
Ответ: {answer}

{context_part}

Выдай ТОЛЬКО факты о человеке (интересы, предпочтения, планы, работа, и т.д.)
Каждый факт — 3-7 слов. Максимум 3 факта.
Отвечай ТОЛЬКО валидным JSON: {{"facts": ["факт 1", "факт 2"]}} или {{"facts": []}} если фактов нет."""

    try:
        response = await ai_service.mistral_client.chat.complete_async(
            model=MISTRAL_MODEL,
            max_tokens=150,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip() if response.choices else ""

        try:
            data = json.loads(raw)
            raw_facts = data.get("facts", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"Failed to parse JSON facts, skipping: {raw[:80]}")
            return []

        facts = []
        for fact in raw_facts:
            if isinstance(fact, str) and 3 < len(fact) < 100:
                if user_name and not fact.startswith(user_name):
                    fact = f"{user_name}: {fact}"
                facts.append(fact)
        return facts[:3]
    except Exception as e:
        logger.error(f"Failed to extract facts: {e}")
        return []


# ── Recent messages buffer (for proactive memory + style) ────────────

async def store_recent_message(
    chat_id: int, user_id: int, user_name: str, text: str,
    thread_id: int | None = None,
) -> None:
    """Push a message into per-chat-thread and per-user recent-message lists in Redis.

    Key format: chat:{chat_id}:{thread_id}:recent_msgs
    thread_id=0 for non-topic (regular) group chats.
    This mirrors the in-memory context key so the Redis fallback after
    a restart restores the correct per-topic history.
    """
    if not redis_client:
        return
    try:
        entry = f"{user_name}: {text[:300]}"
        chat_key = f"chat:{chat_id}:{thread_id or 0}:recent_msgs"
        user_key = f"user:{user_id}:recent_msgs"
        pipe = redis_client.pipeline()
        # Per-chat-thread buffer (for proactive memory + restart recovery)
        pipe.lpush(chat_key, entry)
        pipe.ltrim(chat_key, 0, 29)  # keep 30
        pipe.expire(chat_key, _KEY_TTL_SECONDS)
        # Per-user buffer (for style analysis — not thread-scoped)
        pipe.lpush(user_key, text[:300])
        pipe.ltrim(user_key, 0, STYLE_RECENT_MESSAGES_KEPT - 1)
        pipe.expire(user_key, _KEY_TTL_SECONDS)
        await pipe.execute()
    except Exception as e:
        logger.error(f"Failed to store recent message: {e}")


async def get_recent_chat_messages(
    chat_id: int, count: int = 20, thread_id: int | None = None,
) -> list[str]:
    """Get the last N messages from a chat thread (newest first)."""
    if not redis_client:
        return []
    try:
        key = f"chat:{chat_id}:{thread_id or 0}:recent_msgs"
        return await redis_client.lrange(key, 0, count - 1)
    except Exception as e:
        logger.error(f"Failed to get recent chat messages: {e}")
        return []


# ── Proactive fact extraction from conversation ──────────────────────

async def extract_facts_from_conversation(
    chat_id: int, messages: list[str],
) -> list[str]:
    """Use LLM to extract facts about users from a batch of group messages.

    Called by the proactive memory job, not per-message.
    """
    if not messages or len(messages) < 3:
        return []

    from bot.services import ai as ai_service
    if not ai_service.mistral_client:
        return []

    conversation = "\n".join(reversed(messages))  # oldest first
    prompt = f"""Проанализируй эти сообщения из группового чата:

{conversation}

Извлеки важные факты о людях: интересы, предпочтения, планы, работа, настроение,
отношения. Каждый факт в формате "Имя: факт", 3-7 слов. Максимум 5 фактов.
Отвечай ТОЛЬКО валидным JSON: {{"facts": ["Имя: факт", "Имя: факт"]}} или {{"facts": []}} если фактов нет."""

    try:
        response = await ai_service.mistral_client.chat.complete_async(
            model=MISTRAL_MODEL,
            max_tokens=200,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip() if response.choices else ""

        try:
            data = json.loads(raw)
            raw_facts = data.get("facts", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"Failed to parse JSON facts from conversation: {raw[:80]}")
            return []

        facts = [f for f in raw_facts if isinstance(f, str) and 5 < len(f) < 120]
        return facts[:5]
    except Exception as e:
        logger.error(f"Proactive fact extraction failed: {e}")
        return []


# ── Quiet-mode per chat ──────────────────────────────────────────────

async def set_quiet_mode(chat_id: int, enabled: bool) -> None:
    """Toggle proactive messages for a chat."""
    if not redis_client:
        return
    try:
        if enabled:
            await redis_client.set(f"chat:{chat_id}:quiet", "1")
        else:
            await redis_client.delete(f"chat:{chat_id}:quiet")
    except Exception as e:
        logger.error(f"Failed to set quiet mode: {e}")


async def is_quiet_mode(chat_id: int) -> bool:
    """Check if proactive messages are disabled for a chat."""
    if not redis_client:
        return False
    try:
        return await redis_client.exists(f"chat:{chat_id}:quiet") > 0
    except Exception:
        return False


# ── Debate mode per chat/thread ───────────────────────────────────────

def _debate_key(chat_id: int, thread_id: int | None = None) -> str:
    return f"chat:{chat_id}:{thread_id or 0}:debate"


async def set_debate_mode(chat_id: int, topic: str, thread_id: int | None = None, ttl: int = 1800) -> None:
    """Activate debate mode for a chat/thread for `ttl` seconds."""
    if not redis_client:
        return
    try:
        await redis_client.set(_debate_key(chat_id, thread_id), topic, ex=ttl)
    except Exception as e:
        logger.error(f"Failed to set debate mode: {e}")


async def get_debate_topic(chat_id: int, thread_id: int | None = None) -> str | None:
    """Return the active debate topic for a chat/thread, or None if inactive."""
    if not redis_client:
        return None
    try:
        return await redis_client.get(_debate_key(chat_id, thread_id))
    except Exception:
        return None


async def clear_debate_mode(chat_id: int, thread_id: int | None = None) -> None:
    """Deactivate debate mode for a chat/thread."""
    if not redis_client:
        return
    try:
        await redis_client.delete(_debate_key(chat_id, thread_id))
    except Exception as e:
        logger.error(f"Failed to clear debate mode: {e}")
