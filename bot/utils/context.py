"""Chat conversation context and in-memory eviction."""

import time
import logging
from collections import defaultdict

from config import CONTEXT_SIZE, CONTEXT_MAX_AGE, RATE_LIMIT_MAX_AGE, EVICTION_INTERVAL

logger = logging.getLogger(__name__)

# In-memory stores
chat_context: dict[int, list[dict]] = defaultdict(list)
user_last_query: dict[int, float] = defaultdict(float)
_last_eviction: float = 0.0


def add_to_context(chat_id: int, role: str, name: str, content: str) -> None:
    """Add a message to the chat context."""
    chat_context[chat_id].append({
        "role": role,
        "name": name,
        "content": content[:500],
        "time": time.time(),
    })
    if len(chat_context[chat_id]) > CONTEXT_SIZE:
        chat_context[chat_id] = chat_context[chat_id][-CONTEXT_SIZE:]


def get_context_string(chat_id: int) -> str:
    """Get recent conversation context as a string."""
    if not chat_context[chat_id]:
        return ""
    lines = []
    for msg in chat_context[chat_id][-CONTEXT_SIZE:]:
        name = msg.get("name", "user")
        lines.append(f"{name}: {msg['content']}")
    return "\n".join(lines)


def evict_stale_data() -> None:
    """Remove stale entries from in-memory dicts.

    Called periodically from handle_message (every EVICTION_INTERVAL seconds).
    """
    global _last_eviction
    now = time.time()
    if now - _last_eviction < EVICTION_INTERVAL:
        return
    _last_eviction = now

    stale_chats = [
        cid for cid, msgs in chat_context.items()
        if msgs and now - msgs[-1].get("time", 0) > CONTEXT_MAX_AGE
    ]
    for cid in stale_chats:
        del chat_context[cid]

    stale_users = [
        uid for uid, ts in user_last_query.items()
        if now - ts > RATE_LIMIT_MAX_AGE
    ]
    for uid in stale_users:
        del user_last_query[uid]

    if stale_chats or stale_users:
        logger.info(f"Evicted {len(stale_chats)} stale contexts, {len(stale_users)} rate-limit entries")
