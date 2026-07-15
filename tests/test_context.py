from bot.utils.context import (
    add_to_context,
    chat_context,
    clear_context,
    get_context_messages,
    get_context_string,
    get_last_bot_reply_target,
    last_bot_reply_target,
    set_last_bot_reply_target,
    trim_context_for_api,
)


def _reset(chat_id, thread_id=None):
    clear_context(chat_id, thread_id)


def test_add_and_get_context_messages_alternates_roles():
    chat_id = 111
    _reset(chat_id)
    add_to_context(chat_id, "user", "Alice", "hello there", thread_id=None)
    add_to_context(chat_id, "assistant", "bot", "hi Alice", thread_id=None)
    add_to_context(chat_id, "user", "Bob", "hey", thread_id=None)

    msgs = get_context_messages(chat_id)
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user"]
    assert "Alice" in msgs[0]["content"]


def test_context_merges_consecutive_same_role():
    chat_id = 112
    _reset(chat_id)
    add_to_context(chat_id, "user", "Alice", "first message", thread_id=None)
    add_to_context(chat_id, "user", "Bob", "second message", thread_id=None)

    msgs = get_context_messages(chat_id)
    assert len(msgs) == 1
    assert "first message" in msgs[0]["content"]
    assert "second message" in msgs[0]["content"]


def test_context_drops_leading_assistant_message():
    chat_id = 113
    _reset(chat_id)
    add_to_context(chat_id, "assistant", "bot", "orphaned reply", thread_id=None)
    add_to_context(chat_id, "user", "Alice", "a question", thread_id=None)

    msgs = get_context_messages(chat_id)
    assert msgs[0]["role"] == "user"


def test_context_is_isolated_per_thread():
    chat_id = 114
    _reset(chat_id, thread_id=1)
    _reset(chat_id, thread_id=2)
    add_to_context(chat_id, "user", "Alice", "in topic 1", thread_id=1)
    add_to_context(chat_id, "user", "Bob", "in topic 2", thread_id=2)

    topic1_string = get_context_string(chat_id, thread_id=1)
    topic2_string = get_context_string(chat_id, thread_id=2)
    assert "in topic 1" in topic1_string
    assert "in topic 2" not in topic1_string
    assert "in topic 2" in topic2_string


def test_trim_context_for_api_below_threshold_unchanged():
    messages = [{"role": "user", "content": "hi"}]
    assert trim_context_for_api(messages) == messages


def test_trim_context_for_api_compacts_long_history():
    messages = []
    for i in range(20):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"msg {i}"})

    trimmed = trim_context_for_api(messages)
    assert len(trimmed) <= 10
    assert "пропущено" in trimmed[0]["content"]
    # Most recent message should still be present
    assert trimmed[-1]["content"] == messages[-1]["content"]


def test_clear_context_removes_entry():
    chat_id = 115
    add_to_context(chat_id, "user", "Alice", "hello", thread_id=None)
    assert (chat_id, 0) in chat_context
    clear_context(chat_id)
    assert (chat_id, 0) not in chat_context


def test_last_bot_reply_target_roundtrip():
    chat_id = 116
    assert get_last_bot_reply_target(chat_id) is None
    set_last_bot_reply_target(chat_id, 42, "Alice")
    assert get_last_bot_reply_target(chat_id) == (42, "Alice")


def test_last_bot_reply_target_isolated_per_thread():
    chat_id = 117
    set_last_bot_reply_target(chat_id, 1, "Alice", thread_id=1)
    set_last_bot_reply_target(chat_id, 2, "Bob", thread_id=2)
    assert get_last_bot_reply_target(chat_id, thread_id=1) == (1, "Alice")
    assert get_last_bot_reply_target(chat_id, thread_id=2) == (2, "Bob")


def test_clear_context_removes_last_bot_reply_target():
    chat_id = 118
    set_last_bot_reply_target(chat_id, 1, "Alice")
    assert (chat_id, 0) in last_bot_reply_target
    clear_context(chat_id)
    assert (chat_id, 0) not in last_bot_reply_target
