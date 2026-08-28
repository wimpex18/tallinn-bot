"""Tests for the Telegram command menu (BotCommand list) synced via set_my_commands.

Regression coverage for a real bug: /quiz, /quote, /quotes were added as
handlers in main.py but the BotFather-visible command menu was never updated,
because this repo relied on manual BotFather configuration and had no code
path that calls set_my_commands.
"""

from unittest.mock import AsyncMock

import pytest

import main


def test_bot_commands_cover_every_registered_handler():
    registered = {
        "start", "help", "remember", "forget", "memory", "quiet", "clear",
        "summary", "tldr", "debate", "factcheck", "poll", "quiz", "quote", "quotes",
    }
    listed = {cmd.command for cmd in main.BOT_COMMANDS}
    assert listed == registered


def test_bot_commands_have_short_descriptions():
    for cmd in main.BOT_COMMANDS:
        assert 1 <= len(cmd.description) <= 256


@pytest.mark.asyncio
async def test_init_clients_syncs_command_menu(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(main, "REDIS_URL", None)
    set_my_commands_mock = AsyncMock()
    bot = SimpleNamespace(set_my_commands=set_my_commands_mock)
    application = SimpleNamespace(bot=bot, job_queue=None)

    await main.init_clients(application)

    set_my_commands_mock.assert_called_once_with(main.BOT_COMMANDS)
