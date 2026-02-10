"""Redis-backed persistent memory for user and group facts."""

import time
import re
import logging

import httpx

from config import PERPLEXITY_API_KEY, STYLE_RECENT_MESSAGES_KEPT

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


# ── Recent messages buffer (for proactive memory + style) ────────────

async def store_recent_message(
    chat_id: int, user_id: int, user_name: str, text: str,
) -> None:
    """Push a message into per-chat and per-user recent-message lists in Redis."""
    if not redis_client:
        return
    try:
        entry = f"{user_name}: {text[:300]}"
        pipe = redis_client.pipeline()
        # Per-chat buffer (for proactive memory extraction)
        pipe.lpush(f"chat:{chat_id}:recent_msgs", entry)
        pipe.ltrim(f"chat:{chat_id}:recent_msgs", 0, 29)  # keep 30
        # Per-user buffer (for style analysis)
        pipe.lpush(f"user:{user_id}:recent_msgs", text[:300])
        pipe.ltrim(f"user:{user_id}:recent_msgs", 0, STYLE_RECENT_MESSAGES_KEPT - 1)
        await pipe.execute()
    except Exception as e:
        logger.error(f"Failed to store recent message: {e}")


async def get_recent_chat_messages(chat_id: int, count: int = 20) -> list[str]:
    """Get the last N messages from a chat (newest first)."""
    if not redis_client:
        return []
    try:
        return await redis_client.lrange(f"chat:{chat_id}:recent_msgs", 0, count - 1)
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

    conversation = "\n".join(reversed(messages))  # oldest first
    prompt = f"""Проанализируй эти сообщения из группового чата:

{conversation}

Извлеки важные факты о людях: интересы, предпочтения, планы, работа, настроение,
отношения. Формат: "Имя: факт" — один факт на строку, коротко (3-7 слов).
Если фактов нет — ответь НЕТ. Максимум 5 фактов."""

    try:
        client = http_client or httpx.AsyncClient(timeout=15.0)
        resp = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()

        if "НЕТ" in result.upper() or len(result) < 5:
            return []

        facts = []
        for line in result.split("\n"):
            line = line.strip().lstrip("-•0123456789.").strip()
            if 5 < len(line) < 120:
                facts.append(line)
        return facts[:5]
    except Exception as e:
        logger.error(f"Proactive fact extraction failed: {e}")
        return []


# ── Redis cleanup / maintenance ──────────────────────────────────────

async def cleanup_stale_redis_keys(max_age_days: int = 90) -> dict:
    """Scan Redis for orphaned keys and delete those untouched for > max_age_days.

    Returns a summary dict of what was cleaned up.
    """
    if not redis_client:
        return {"error": "Redis not connected"}

    stats = {"scanned": 0, "deleted": 0, "patterns": {}}
    now = time.time()
    cutoff = now - (max_age_days * 86400)

    try:
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, count=100)
            for key in keys:
                stats["scanned"] += 1
                key_type = await redis_client.type(key)

                should_delete = False

                if key_type == "zset":
                    # Sorted sets: check highest score (most recent timestamp)
                    top = await redis_client.zrange(key, -1, -1, withscores=True)
                    if not top:
                        should_delete = True
                    elif top[0][1] < cutoff:
                        should_delete = True

                elif key_type == "hash":
                    last_seen = await redis_client.hget(key, "last_seen")
                    if last_seen:
                        # Profile hash — check last_seen
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(last_seen)
                            if dt.timestamp() < cutoff:
                                should_delete = True
                        except (ValueError, TypeError):
                            pass
                    else:
                        # Style hash — no TTL check needed, counter-based
                        pass

                elif key_type == "list":
                    # Recent message lists — check TTL or if empty
                    length = await redis_client.llen(key)
                    if length == 0:
                        should_delete = True

                if should_delete:
                    await redis_client.delete(key)
                    stats["deleted"] += 1
                    # Track pattern
                    pattern = ":".join(key.split(":")[:1] + ["*"] + key.split(":")[2:])
                    stats["patterns"][pattern] = stats["patterns"].get(pattern, 0) + 1

            if cursor == 0:
                break
    except Exception as e:
        logger.error(f"Redis cleanup failed: {e}")
        stats["error"] = str(e)

    logger.info(f"Redis cleanup: scanned={stats['scanned']}, deleted={stats['deleted']}")
    return stats


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
