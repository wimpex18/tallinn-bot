from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.handlers import actions
from bot.services import ai as ai_module
from bot.services import memory as memory_module
from bot.services import search as search_module


def _make_update_context(chat_id=1, thread_id=None, message_id=100):
    message = SimpleNamespace(
        message_thread_id=thread_id,
        message_id=message_id,
        reply_to_message=None,
        reply_text=AsyncMock(),
    )
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=chat_id, type="group"),
    )
    bot = SimpleNamespace(
        send_chat_action=AsyncMock(),
        send_poll=AsyncMock(),
    )
    context = SimpleNamespace(bot=bot)
    return update, context


@pytest.mark.asyncio
async def test_do_summary_sends_summary(monkeypatch):
    update, context = _make_update_context()
    monkeypatch.setattr(
        memory_module, "get_recent_chat_messages", AsyncMock(return_value=["Alice: hi", "Bob: hey"]),
    )
    monkeypatch.setattr(ai_module, "summarize_conversation", AsyncMock(return_value="Коротко поболтали"))

    await actions.do_summary(update, context, count=30)

    update.message.reply_text.assert_called_once_with("Коротко поболтали")


@pytest.mark.asyncio
async def test_do_summary_no_messages(monkeypatch):
    update, context = _make_update_context()
    monkeypatch.setattr(memory_module, "get_recent_chat_messages", AsyncMock(return_value=[]))

    await actions.do_summary(update, context)

    update.message.reply_text.assert_called_once()
    assert "нечего" in update.message.reply_text.call_args.args[0].lower()


@pytest.mark.asyncio
async def test_do_debate_sets_mode_and_confirms(monkeypatch):
    update, context = _make_update_context(chat_id=42, thread_id=7)
    set_mode_mock = AsyncMock()
    monkeypatch.setattr(memory_module, "set_debate_mode", set_mode_mock)

    await actions.do_debate(update, context, "удалёнка лучше офиса")

    set_mode_mock.assert_called_once()
    args, kwargs = set_mode_mock.call_args
    assert args[0] == 42
    assert args[1] == "удалёнка лучше офиса"
    assert kwargs["thread_id"] == 7
    update.message.reply_text.assert_called_once()
    assert "удалёнка лучше офиса" in update.message.reply_text.call_args.args[0]


@pytest.mark.asyncio
async def test_do_factcheck_replies_with_verdict(monkeypatch):
    update, context = _make_update_context()
    monkeypatch.setattr(
        search_module, "search_web", AsyncMock(return_value="[WEB SEARCH: ...] found stuff"),
    )
    monkeypatch.setattr(ai_module, "query_ai", AsyncMock(return_value="Похоже на правду"))

    await actions.do_factcheck(update, context, "в Таллинне живёт 500 тысяч человек")

    update.message.reply_text.assert_called_once()
    assert update.message.reply_text.call_args.args[0] == "Похоже на правду"


@pytest.mark.asyncio
async def test_do_factcheck_search_fails(monkeypatch):
    update, context = _make_update_context()
    monkeypatch.setattr(search_module, "search_web", AsyncMock(return_value=None))

    await actions.do_factcheck(update, context, "что-то")

    update.message.reply_text.assert_called_once()
    assert "не получилось" in update.message.reply_text.call_args.args[0].lower()


@pytest.mark.asyncio
async def test_do_poll_suggest_sends_poll(monkeypatch):
    update, context = _make_update_context(chat_id=99, thread_id=None)
    monkeypatch.setattr(actions, "get_context_string", lambda chat_id, thread_id: "обсуждали еду")
    monkeypatch.setattr(
        ai_module, "suggest_poll",
        AsyncMock(return_value={"question": "Пицца или суши?", "options": ["Пицца", "Суши"]}),
    )

    await actions.do_poll_suggest(update, context)

    context.bot.send_poll.assert_called_once()
    kwargs = context.bot.send_poll.call_args.kwargs
    assert kwargs["chat_id"] == 99
    assert kwargs["allows_revoting"] is True
    assert kwargs["question"] == "Пицца или суши?"


@pytest.mark.asyncio
async def test_do_poll_suggest_no_context(monkeypatch):
    update, context = _make_update_context()
    monkeypatch.setattr(actions, "get_context_string", lambda chat_id, thread_id: "")

    await actions.do_poll_suggest(update, context)

    update.message.reply_text.assert_called_once()
    context.bot.send_poll.assert_not_called()


@pytest.mark.asyncio
async def test_do_poll_suggest_model_declines(monkeypatch):
    update, context = _make_update_context()
    monkeypatch.setattr(actions, "get_context_string", lambda chat_id, thread_id: "обсуждали ерунду")
    monkeypatch.setattr(ai_module, "suggest_poll", AsyncMock(return_value=None))

    await actions.do_poll_suggest(update, context)

    update.message.reply_text.assert_called_once()
    context.bot.send_poll.assert_not_called()
