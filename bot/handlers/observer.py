"""Silent observer + spontaneous replies.

Registered in handler group 1 (separate from the main handlers in group 0).
Runs on EVERY group message, even ones the bot doesn't respond to.

Responsibilities:
1. Store every message in the Redis recent-messages buffer
2. Update per-user communication-style counters
3. Occasionally (probability-based) generate a spontaneous reply
"""

import time
import random
import asyncio
import logging
import zoneinfo
from datetime import datetime

import httpx

from telegram import Update
from telegram.ext import ContextTypes

from config import (
    BOT_USERNAME,
    PERPLEXITY_API_KEY,
    SPONTANEOUS_REPLY_PROBABILITY,
    SPONTANEOUS_REPLY_KEYWORD_BOOST,
    SPONTANEOUS_REPLY_COOLDOWN,
    SPONTANEOUS_REPLY_MIN_MESSAGES,
    PROACTIVE_MAX_PER_HOUR,
    INTERESTING_TOPICS,
    QUIET_HOURS_START,
    QUIET_HOURS_END,
)
from bot.utils.helpers import get_message_content, get_display_name
from bot.utils.context import get_context_string
from bot.services.memory import (
    store_recent_message, is_quiet_mode, save_group_fact, redis_client,
)
from bot.services.style import update_style_counters
from bot.services import memory as memory_service

logger = logging.getLogger(__name__)

# In-memory tracking for spontaneous replies
_last_spontaneous: dict[int, float] = {}     # chat_id -> timestamp
_messages_since_reply: dict[int, int] = {}   # chat_id -> count
_hourly_sends: dict[int, list[float]] = {}   # chat_id -> [timestamps]
_OBSERVER_MAX_CHATS = 500   # evict stale entries if tracking more than this many chats
_OBSERVER_STALE_AGE = 7200  # 2 hours — consider a chat stale if no spontaneous activity

TALLINN_TZ = zoneinfo.ZoneInfo("Europe/Tallinn")


def _is_quiet_hours() -> bool:
    """Check if we're in quiet hours (nighttime in Tallinn)."""
    hour = datetime.now(TALLINN_TZ).hour
    if QUIET_HOURS_START > QUIET_HOURS_END:
        # Wraps past midnight (e.g., 23–08)
        return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END
    return QUIET_HOURS_START <= hour < QUIET_HOURS_END


def _check_rate_ok(chat_id: int) -> bool:
    """Check if we can send a spontaneous message to this chat right now."""
    now = time.time()

    # Cooldown since last spontaneous message
    last = _last_spontaneous.get(chat_id, 0)
    if now - last < SPONTANEOUS_REPLY_COOLDOWN:
        return False

    # Minimum messages since last bot reply
    if _messages_since_reply.get(chat_id, 0) < SPONTANEOUS_REPLY_MIN_MESSAGES:
        return False

    # Hourly cap
    if chat_id not in _hourly_sends:
        _hourly_sends[chat_id] = []
    _hourly_sends[chat_id] = [t for t in _hourly_sends[chat_id] if now - t < 3600]
    if len(_hourly_sends[chat_id]) >= PROACTIVE_MAX_PER_HOUR:
        return False

    return True


def evict_stale_observer_data() -> None:
    """Remove stale entries from observer tracking dicts to prevent unbounded growth."""
    now = time.time()
    if len(_last_spontaneous) <= _OBSERVER_MAX_CHATS:
        return
    stale = [
        cid for cid, ts in _last_spontaneous.items()
        if now - ts > _OBSERVER_STALE_AGE
    ]
    for cid in stale:
        _last_spontaneous.pop(cid, None)
        _messages_since_reply.pop(cid, None)
        _hourly_sends.pop(cid, None)
    if stale:
        logger.info(f"Observer evicted {len(stale)} stale chat tracking entries")


def record_bot_replied(chat_id: int) -> None:
    """Call this from the main message handler after the bot sends a normal reply.

    Resets the "messages since last reply" counter so the bot doesn't
    immediately chime in again right after a normal interaction.
    """
    _messages_since_reply[chat_id] = 0


# ── The silent observer handler (handler group 1) ────────────────────

async def observe_and_learn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process every group message: store, profile, maybe spontaneously reply."""
    message = update.message
    if not message:
        return

    # Only groups
    if message.chat.type == "private":
        return

    text = get_message_content(message)
    if not text:
        return

    chat_id = message.chat.id
    user = update.effective_user
    if not user:
        return

    user_id = user.id
    user_name = get_display_name(user) or user.first_name or "user"

    # 1. Store message in Redis buffer (for proactive memory + style)
    asyncio.create_task(_store_and_profile(chat_id, user_id, user_name, text))

    # 2. Increment "messages since last bot reply" counter
    _messages_since_reply[chat_id] = _messages_since_reply.get(chat_id, 0) + 1

    # Periodic cleanup of observer tracking dicts
    evict_stale_observer_data()

    # 3. Maybe send a spontaneous reply
    #    Skip if: message is from the bot itself, quiet hours, quiet mode
    if user.username == BOT_USERNAME:
        return
    if _is_quiet_hours():
        return
    if not _check_rate_ok(chat_id):
        return

    # Probability check
    probability = SPONTANEOUS_REPLY_PROBABILITY
    text_lower = text.lower()
    if any(kw in text_lower for kw in INTERESTING_TOPICS):
        probability += SPONTANEOUS_REPLY_KEYWORD_BOOST

    if random.random() > probability:
        return

    # Check quiet mode (async, so checked last to avoid Redis call on every msg)
    if await is_quiet_mode(chat_id):
        return

    # Generate and send
    conv_context = get_context_string(chat_id)
    comment = await _generate_spontaneous_comment(conv_context, text, user_name)

    if comment:
        now = time.time()
        _last_spontaneous[chat_id] = now
        _messages_since_reply[chat_id] = 0
        if chat_id not in _hourly_sends:
            _hourly_sends[chat_id] = []
        _hourly_sends[chat_id].append(now)

        await message.reply_text(comment, reply_to_message_id=message.message_id)
        logger.info(f"[spontaneous] Replied in chat {chat_id}: {comment[:60]}...")


async def _store_and_profile(
    chat_id: int, user_id: int, user_name: str, text: str,
) -> None:
    """Background task: store recent message + update style counters."""
    try:
        await store_recent_message(chat_id, user_id, user_name, text)
        await update_style_counters(memory_service.redis_client, user_id, text)
    except Exception as e:
        logger.error(f"Observer store/profile failed: {e}")


async def _generate_spontaneous_comment(
    conv_context: str, trigger_msg: str, user_name: str,
) -> str | None:
    """Ask LLM whether to chime in and what to say.

    Returns None if the LLM decides to stay silent.
    """
    prompt = (
        f"Ты участник группового чата о жизни в Таллинне. "
        f"Вот недавний контекст:\n{conv_context}\n\n"
        f"Последнее сообщение от {user_name}: {trigger_msg}\n\n"
        f"Если тебе есть что добавить — напиши ОДНО короткое неформальное сообщение "
        f"(1-2 предложения, без эмодзи). Будь как обычный участник чата, не как бот. "
        f"Если добавить нечего — ответь только НЕТ."
    )

    try:
        client = memory_service.http_client or httpx.AsyncClient(timeout=15.0)
        resp = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 80,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()

        if not result or "НЕТ" in result.upper() or len(result) < 3:
            return None
        # Clean citation markers
        import re
        result = re.sub(r'\[\d+\]', '', result).strip()
        return result
    except Exception as e:
        logger.error(f"Spontaneous comment generation failed: {e}")
        return None
