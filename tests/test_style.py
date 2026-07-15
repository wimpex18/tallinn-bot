from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.services.style import analyze_message_style, get_style_summary


def test_analyze_message_style_detects_emoji():
    signals = analyze_message_style("hello there 😂")
    assert signals["uses_emoji"] is True


def test_analyze_message_style_detects_caps():
    signals = analyze_message_style("WHY IS THIS HAPPENING")
    assert signals["uses_caps"] is True


def test_analyze_message_style_ignores_short_caps():
    signals = analyze_message_style("OK")
    assert signals["uses_caps"] is False


def test_analyze_message_style_detects_profanity():
    signals = analyze_message_style("это просто пиздец какой-то")
    assert signals["uses_profanity"] is True


def test_analyze_message_style_detects_slang():
    signals = analyze_message_style("го тусить, кста я норм")
    assert signals["uses_slang"] is True


def test_analyze_message_style_neutral_message():
    signals = analyze_message_style("Сегодня хорошая погода в городе")
    assert signals["uses_emoji"] is False
    assert signals["uses_caps"] is False
    assert signals["uses_profanity"] is False


def test_analyze_message_style_message_length():
    signals = analyze_message_style("12345")
    assert signals["msg_length"] == 5


def test_analyze_message_style_parenthesis_smileys():
    signals = analyze_message_style("ну норм)))")
    assert signals["uses_parenthesis_smileys"] is True


def _fake_redis_for_summary(data: dict):
    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.hgetall = AsyncMock(return_value=data)
    return redis_client


@pytest.mark.asyncio
async def test_get_style_summary_high_profanity_scopes_to_casual_tone():
    redis_client = _fake_redis_for_summary({
        "msg_count": "10",
        "profanity_count": "5",
        "slang_count": "0",
        "emoji_count": "0",
        "total_msg_length": "300",
    })

    summary = await get_style_summary(redis_client, user_id=1)

    assert summary is not None
    # The instruction must scope profanity to casual tone, never license hostility.
    assert "НЕ грубить" in summary
    assert "враждебность" in summary
    assert "можно отвечать грубовато и с юмором" not in summary


@pytest.mark.asyncio
async def test_get_style_summary_no_redis_returns_none():
    assert await get_style_summary(None, user_id=1) is None
