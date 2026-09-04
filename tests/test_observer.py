"""Tests for bot/handlers/observer.py — specifically the emoji-reaction gate.

Regression coverage for a behavior change the user asked for directly:
emoji reactions used to fire on ANY group message at REACTION_PROBABILITY
(the observer runs on every message regardless of whether the bot was
addressed), which read as the bot randomly reacting to unrelated chat.
Reactions are now only even considered for messages that address the bot
(@mention, reply to it, or its plain-text name) — and even then only
sometimes, per REACTION_PROBABILITY.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.handlers import observer


def _make_update_context(*, text, reply_to_bot=False, bot_username="test_bot", bot_id=999):
    reply_to_message = None
    if reply_to_bot:
        reply_to_message = SimpleNamespace(from_user=SimpleNamespace(username=bot_username, id=bot_id))

    message = SimpleNamespace(
        chat=SimpleNamespace(type="group", id=1),
        message_thread_id=None,
        message_id=100,
        text=text,
        caption=None,
        reply_to_message=reply_to_message,
    )
    user = SimpleNamespace(id=1, username="alice", first_name="Alice")
    update = SimpleNamespace(message=message, effective_user=user)
    bot = SimpleNamespace(id=bot_id, set_message_reaction=AsyncMock())
    context = SimpleNamespace(bot=bot)
    return update, context


@pytest.fixture(autouse=True)
def _bypass_unrelated_engagement_gates(monkeypatch):
    """Isolate the should_respond() reaction gate from orthogonal concerns
    (quiet hours depend on real wall-clock time, rate caps on shared
    module-level counters) that would otherwise make these tests flaky."""
    monkeypatch.setattr(observer, "_is_quiet_hours", lambda: False)
    monkeypatch.setattr(observer, "_check_rate_ok", lambda chat_id: True)
    monkeypatch.setattr(observer, "_store_and_profile", AsyncMock())
    monkeypatch.setattr(observer, "is_quiet_mode", AsyncMock(return_value=False))
    monkeypatch.setattr(observer.random, "random", lambda: 0.0)  # always clears probability checks


@pytest.mark.asyncio
async def test_no_reaction_on_unaddressed_message():
    update, context = _make_update_context(text="просто болтаем в чате")

    await observer.observe_and_learn(update, context)

    context.bot.set_message_reaction.assert_not_called()


@pytest.mark.asyncio
async def test_reacts_when_bot_mentioned():
    update, context = _make_update_context(text="@test_bot привет!")

    await observer.observe_and_learn(update, context)

    context.bot.set_message_reaction.assert_called_once()


@pytest.mark.asyncio
async def test_reacts_when_replying_to_bot():
    update, context = _make_update_context(text="ага, согласен", reply_to_bot=True)

    await observer.observe_and_learn(update, context)

    context.bot.set_message_reaction.assert_called_once()


@pytest.mark.asyncio
async def test_reacts_when_bot_named_in_text():
    update, context = _make_update_context(text="Сэм, красава")

    await observer.observe_and_learn(update, context)

    context.bot.set_message_reaction.assert_called_once()


@pytest.mark.asyncio
async def test_no_reaction_on_unaddressed_message_even_with_interesting_keyword(monkeypatch):
    """The INTERESTING_TOPICS keyword boost shouldn't bypass the
    addressed-to-bot gate — only make the reaction more likely once gated."""
    monkeypatch.setattr(observer, "INTERESTING_TOPICS", ["погода"])
    update, context = _make_update_context(text="ужасная сегодня погода")

    await observer.observe_and_learn(update, context)

    context.bot.set_message_reaction.assert_not_called()
