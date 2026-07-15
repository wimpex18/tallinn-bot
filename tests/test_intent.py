from unittest.mock import AsyncMock

import pytest

from bot.services import ai as ai_module
from bot.services.intent import Intent, detect_action_intent


@pytest.mark.asyncio
async def test_strong_summary_trigger_no_llm_call(monkeypatch):
    classify_mock = AsyncMock()
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    result = await detect_action_intent("о чём тут говорили?", None)

    assert result == Intent("summary")
    classify_mock.assert_not_called()


@pytest.mark.asyncio
async def test_strong_poll_trigger_no_llm_call(monkeypatch):
    classify_mock = AsyncMock()
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    result = await detect_action_intent("сделай опрос", None)

    assert result == Intent("poll")
    classify_mock.assert_not_called()


@pytest.mark.asyncio
async def test_strong_debate_trigger_with_topic_no_llm_call(monkeypatch):
    classify_mock = AsyncMock()
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    result = await detect_action_intent("давай поспорим про удалёнку", None)

    assert result.action == "debate"
    assert result.topic == "удалёнку"
    classify_mock.assert_not_called()


@pytest.mark.asyncio
async def test_strong_factcheck_trigger_with_claim_no_llm_call(monkeypatch):
    classify_mock = AsyncMock()
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    result = await detect_action_intent("проверь факт: в Таллинне живёт 500 тысяч человек", None)

    assert result.action == "factcheck"
    assert "500 тысяч" in result.claim
    classify_mock.assert_not_called()


@pytest.mark.asyncio
async def test_no_signal_returns_none_without_llm_call(monkeypatch):
    classify_mock = AsyncMock()
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    result = await detect_action_intent("какая сегодня погода в Таллинне?", None)

    assert result is None
    classify_mock.assert_not_called()


@pytest.mark.asyncio
async def test_weak_signal_falls_through_to_llm_classification(monkeypatch):
    classify_mock = AsyncMock(return_value={"action": "debate", "topic": "IPA vs lager", "claim": None})
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    result = await detect_action_intent("не хочу дебатировать но всё же", "some context")

    classify_mock.assert_called_once()
    assert result.action == "debate"
    assert result.topic == "IPA vs lager"


@pytest.mark.asyncio
async def test_strong_debate_trigger_without_topic_falls_through_to_llm(monkeypatch):
    classify_mock = AsyncMock(
        return_value={"action": "debate", "topic": "офис или удалёнка", "claim": None},
    )
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    result = await detect_action_intent("давай поспорим", "context about remote work")

    classify_mock.assert_called_once()
    assert result.action == "debate"
    assert result.topic == "офис или удалёнка"


@pytest.mark.asyncio
async def test_llm_classification_returning_none_falls_through(monkeypatch):
    classify_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    result = await detect_action_intent("у меня внутренний дебат какой-то", None)

    classify_mock.assert_called_once()
    assert result is None


@pytest.mark.asyncio
async def test_empty_question_returns_none():
    assert await detect_action_intent("", None) is None
    assert await detect_action_intent(None, None) is None
