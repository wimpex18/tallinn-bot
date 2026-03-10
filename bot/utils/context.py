"""Chat conversation context and in-memory eviction.

Context is keyed by (chat_id, thread_id) so Telegram forum topics each get
their own isolated history.  thread_id=0 is used for non-topic chats.
"""

import time
import logging
from collections import defaultdict

from config import (
    CONTEXT_SIZE, CONTEXT_MAX_AGE, RATE_LIMIT_MAX_AGE, EVICTION_INTERVAL,
    CONTEXT_COMPACT_THRESHOLD, CONTEXT_COMPACT_KEEP,
)

logger = logging.getLogger(__name__)

# Context key: (chat_id, thread_id) — thread_id=0 for non-topic chats
_CtxKey = tuple[int, int]


def _key(chat_id: int, thread_id: int | None = None) -> _CtxKey:
    return (chat_id, thread_id or 0)


# In-memory stores
chat_context: dict[_CtxKey, list[dict]] = defaultdict(list)
user_last_query: dict[int, float] = defaultdict(float)
_last_eviction: float = 0.0


def add_to_context(
    chat_id: int, role: str, name: str, content: str,
    thread_id: int | None = None,
) -> None:
    """Add a message to the chat context."""
    k = _key(chat_id, thread_id)
    chat_context[k].append({
        "role": role,
        "name": name,
        "content": content[:1000],
        "time": time.time(),
    })
    if len(chat_context[k]) > CONTEXT_SIZE:
        chat_context[k] = chat_context[k][-CONTEXT_SIZE:]


def get_context_string(chat_id: int, thread_id: int | None = None) -> str:
    """Get recent conversation context as a string."""
    k = _key(chat_id, thread_id)
    if not chat_context[k]:
        return ""
    lines = []
    for msg in chat_context[k][-CONTEXT_SIZE:]:
        name = msg.get("name", "user")
        lines.append(f"{name}: {msg['content']}")
    return "\n".join(lines)


def get_context_messages(chat_id: int, thread_id: int | None = None) -> list[dict]:
    """Get recent conversation as a list of {role, content} for the API.

    Merges consecutive same-role messages and maps to user/assistant roles.
    Returns at most CONTEXT_SIZE messages.
    """
    k = _key(chat_id, thread_id)
    if not chat_context[k]:
        return []

    api_msgs = []
    for msg in chat_context[k][-CONTEXT_SIZE:]:
        role = "assistant" if msg["role"] == "assistant" else "user"
        name = msg.get("name", "user")
        text = msg["content"]
        # Prefix user messages with the speaker's name (groups have multiple users)
        if role == "user":
            text = f"{name}: {text}"

        # Merge consecutive same-role messages (API requires alternating roles)
        if api_msgs and api_msgs[-1]["role"] == role:
            api_msgs[-1]["content"] += f"\n{text}"
        else:
            api_msgs.append({"role": role, "content": text})

    # API requires first message after system to be "user".
    # Drop leading assistant messages (rare edge case).
    while api_msgs and api_msgs[0]["role"] == "assistant":
        api_msgs.pop(0)

    return api_msgs


def trim_context_for_api(messages: list[dict]) -> list[dict]:
    """Trim a long context list to CONTEXT_COMPACT_KEEP recent turns.

    Inspired by OpenClaw's session compaction lifecycle.  When the context
    list exceeds CONTEXT_COMPACT_THRESHOLD entries, drops the oldest
    (messages[0 .. -CONTEXT_COMPACT_KEEP]) and prepends a single placeholder
    so the model knows history was omitted.  No LLM call — pure windowing.
    """
    if len(messages) <= CONTEXT_COMPACT_THRESHOLD:
        return messages

    omitted = len(messages) - CONTEXT_COMPACT_KEEP
    recent = list(messages[-CONTEXT_COMPACT_KEEP:])
    placeholder = f"[Контекст: пропущено {omitted} ранних сообщений беседы]"

    # Merge placeholder into the first user turn to preserve alternating roles
    if recent and recent[0]["role"] == "user":
        recent[0] = {**recent[0], "content": placeholder + "\n" + recent[0]["content"]}
    else:
        recent.insert(0, {"role": "user", "content": placeholder})

    return recent


def clear_context(chat_id: int, thread_id: int | None = None) -> None:
    """Clear the in-memory conversation history for a chat/thread."""
    k = _key(chat_id, thread_id)
    if k in chat_context:
        del chat_context[k]


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
        k for k, msgs in chat_context.items()
        if msgs and now - msgs[-1].get("time", 0) > CONTEXT_MAX_AGE
    ]
    for k in stale_chats:
        del chat_context[k]

    stale_users = [
        uid for uid, ts in user_last_query.items()
        if now - ts > RATE_LIMIT_MAX_AGE
    ]
    for uid in stale_users:
        del user_last_query[uid]

    if stale_chats or stale_users:
        logger.info(f"Evicted {len(stale_chats)} stale contexts, {len(stale_users)} rate-limit entries")
