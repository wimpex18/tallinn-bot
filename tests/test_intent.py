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


@pytest.mark.asyncio
async def test_quoted_example_phrases_in_announcement_do_not_trigger(monkeypatch):
    classify_mock = AsyncMock()
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    # Real reported case: an announcement message quoting the four trigger
    # phrases as usage examples, ending with an actual question. The quoted
    # phrases must not be treated as commands, and the real question at the
    # end must not be swallowed by a false-positive action match.
    announcement = (
        'Что нового: можно попросить «сделай саммари», «давай поспорим про Х», '
        '«проверь факт» или «сделай опрос». Команды через "/" тоже работают.\n'
        'я ничего не упустил?'
    )

    result = await detect_action_intent(announcement, None)

    assert result is None
    classify_mock.assert_not_called()


@pytest.mark.asyncio
async def test_word_containing_opros_as_substring_does_not_trigger_poll(monkeypatch):
    classify_mock = AsyncMock()
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    # "вопрос" (question) and "попросить" (to ask) both contain "опрос" as a
    # mid-word substring — must not be mistaken for the poll stem "опрос".
    result = await detect_action_intent("у меня есть вопрос, можно попросить помощи?", None)

    assert result is None
    classify_mock.assert_not_called()


@pytest.mark.asyncio
async def test_opros_as_actual_word_still_triggers_weak_poll_signal(monkeypatch):
    classify_mock = AsyncMock(return_value={"action": "poll", "topic": None, "claim": None})
    monkeypatch.setattr(ai_module, "classify_intent", classify_mock)

    result = await detect_action_intent("может замутим опрос про отпуск?", None)

    classify_mock.assert_called_once()
    assert result.action == "poll"
