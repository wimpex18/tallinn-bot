from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.handlers import commands as commands_module
from bot.handlers.commands import forget_command, poll_command, quote_command, quotes_command
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


def _make_quote_reply_context(chat_type="group", reply_content="это было смешно", reply_user_name="Миша"):
    reply_from_user = SimpleNamespace(id=99, username="misha", first_name=reply_user_name, is_bot=False)
    reply_msg = SimpleNamespace(
        text=reply_content, caption=None, from_user=reply_from_user,
    )
    message = SimpleNamespace(
        reply_to_message=reply_msg, reply_text=AsyncMock(),
    )
    update = SimpleNamespace(
        message=message, effective_chat=SimpleNamespace(id=1, type=chat_type),
    )
    context = SimpleNamespace(args=[])
    return update, context


@pytest.mark.asyncio
async def test_quote_command_saves_reply_with_author(monkeypatch):
    save_mock = AsyncMock()
    monkeypatch.setattr(commands_module, "save_quote", save_mock)
    update, context = _make_quote_reply_context()

    await quote_command(update, context)

    save_mock.assert_called_once()
    args = save_mock.call_args.args
    assert args[0] == 1
    assert "Миша" in args[1]
    assert "это было смешно" in args[1]
    update.message.reply_text.assert_called_once()


@pytest.mark.asyncio
async def test_quote_command_requires_reply():
    message = SimpleNamespace(reply_to_message=None, reply_text=AsyncMock())
    update = SimpleNamespace(
        message=message, effective_chat=SimpleNamespace(id=1, type="group"),
    )
    context = SimpleNamespace(args=[])

    await quote_command(update, context)

    assert "ответь" in update.message.reply_text.call_args.args[0].lower()


@pytest.mark.asyncio
async def test_quote_command_private_chat_rejected():
    update, context = _make_quote_reply_context(chat_type="private")

    await quote_command(update, context)

    assert "групп" in update.message.reply_text.call_args.args[0].lower()


@pytest.mark.asyncio
async def test_quotes_command_shows_random_quote(monkeypatch):
    monkeypatch.setattr(commands_module, "get_random_quote", AsyncMock(return_value="Миша: это было смешно"))
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1, type="group"))
    context = SimpleNamespace(args=[])

    await quotes_command(update, context)

    assert "это было смешно" in message.reply_text.call_args.args[0]


@pytest.mark.asyncio
async def test_quotes_command_none_saved_yet(monkeypatch):
    monkeypatch.setattr(commands_module, "get_random_quote", AsyncMock(return_value=None))
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1, type="group"))
    context = SimpleNamespace(args=[])

    await quotes_command(update, context)

    assert "/quote" in message.reply_text.call_args.args[0]


@pytest.mark.asyncio
async def test_quotes_command_list_mode(monkeypatch):
    monkeypatch.setattr(
        commands_module, "get_all_quotes", AsyncMock(return_value=["Миша: раз", "Сергей: два"]),
    )
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1, type="group"))
    context = SimpleNamespace(args=["list"])

    await quotes_command(update, context)

    text = message.reply_text.call_args.args[0]
    assert "Миша: раз" in text
    assert "Сергей: два" in text
