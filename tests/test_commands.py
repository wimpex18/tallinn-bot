from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.handlers.commands import forget_command, poll_command
from bot.services import memory as memory_module


def _make_update_context(chat_type="private", args=None, member_status="member"):
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(
        message=message,
        effective_user=SimpleNamespace(id=7),
        effective_chat=SimpleNamespace(id=1, type=chat_type),
    )
    bot = SimpleNamespace(
        get_chat_member=AsyncMock(return_value=SimpleNamespace(status=member_status)),
    )
    context = SimpleNamespace(bot=bot, args=args or [])
    return update, context


def _fake_redis():
    redis_client = MagicMock()
    redis_client.delete = AsyncMock(return_value=None)
    return redis_client


@pytest.mark.asyncio
async def test_forget_in_private_chat_wipes_own_facts(monkeypatch):
    redis_client = _fake_redis()
    monkeypatch.setattr(memory_module, "redis_client", redis_client)
    update, context = _make_update_context(chat_type="private")

    await forget_command(update, context)

    redis_client.delete.assert_called_once_with("user:7:facts")
    update.message.reply_text.assert_called_once()


@pytest.mark.asyncio
async def test_forget_me_in_group_lets_non_admin_wipe_own_facts(monkeypatch):
    redis_client = _fake_redis()
    monkeypatch.setattr(memory_module, "redis_client", redis_client)
    update, context = _make_update_context(chat_type="group", args=["me"], member_status="member")

    await forget_command(update, context)

    redis_client.delete.assert_called_once_with("user:7:facts")
    context.bot.get_chat_member.assert_not_called()


@pytest.mark.asyncio
async def test_bare_forget_in_group_requires_admin(monkeypatch):
    redis_client = _fake_redis()
    monkeypatch.setattr(memory_module, "redis_client", redis_client)
    update, context = _make_update_context(chat_type="group", args=[], member_status="member")

    await forget_command(update, context)

    redis_client.delete.assert_not_called()
    assert "админ" in update.message.reply_text.call_args.args[0].lower()


@pytest.mark.asyncio
async def test_bare_forget_in_group_admin_wipes_group_facts(monkeypatch):
    redis_client = _fake_redis()
    monkeypatch.setattr(memory_module, "redis_client", redis_client)
    update, context = _make_update_context(chat_type="group", args=[], member_status="administrator")

    await forget_command(update, context)

    redis_client.delete.assert_called_once_with("group:1:facts")


@pytest.mark.asyncio
async def test_forget_no_redis_reports_unavailable(monkeypatch):
    monkeypatch.setattr(memory_module, "redis_client", None)
    update, context = _make_update_context(chat_type="private")

    await forget_command(update, context)

    assert "не подключена" in update.message.reply_text.call_args.args[0].lower()


@pytest.mark.asyncio
async def test_poll_command_manual_allows_revoting():
    message = SimpleNamespace(message_thread_id=None, reply_text=AsyncMock())
    update = SimpleNamespace(
        message=message, effective_chat=SimpleNamespace(id=1, type="group"),
    )
    bot = SimpleNamespace(send_poll=AsyncMock())
    context = SimpleNamespace(bot=bot, args=["Пицца или суши?", "|", "Пицца", "|", "Суши"])

    await poll_command(update, context)

    bot.send_poll.assert_called_once()
    kwargs = bot.send_poll.call_args.kwargs
    assert kwargs["allows_revoting"] is True
    assert kwargs["is_anonymous"] is False
