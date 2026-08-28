from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.services import ai


def test_clean_response_strips_citations_and_whitespace():
    text = "Some answer[1] with   extra   spaces [2] here."
    cleaned = ai._clean_response(text)
    assert "[1]" not in cleaned
    assert "[2]" not in cleaned
    assert "  " not in cleaned


def test_clean_response_empty_input():
    assert ai._clean_response("") == ""
    assert ai._clean_response(None) is None


def test_has_non_tallinn_location_detects_other_city():
    assert ai._has_non_tallinn_location("что посмотреть в берлине") is True


def test_has_non_tallinn_location_ignores_venue_types():
    assert ai._has_non_tallinn_location("хочу сходить в бар сегодня") is False


def test_has_non_tallinn_location_ignores_time_words():
    assert ai._has_non_tallinn_location("встретимся в пятницу") is False


def test_parse_base64_image_valid():
    data_url = "data:image/jpeg;base64,QUJD"
    block = ai._parse_base64_image(data_url)
    assert block["type"] == "image_url"
    assert block["image_url"]["url"] == data_url


def test_parse_base64_image_invalid_returns_none():
    assert ai._parse_base64_image("not-a-data-url") is None


def _make_fake_client(response_text="Тестовый ответ"):
    client = MagicMock()
    message = MagicMock(content=response_text)
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    client.chat.complete_async = AsyncMock(return_value=response)
    # Default: moderation allows everything through (no categories flagged).
    moderation_entry = MagicMock(categories={})
    moderation_result = MagicMock(results=[moderation_entry])
    client.classifiers.moderate_async = AsyncMock(return_value=moderation_result)
    return client


def _flag_moderation(client, category="violence_and_threats"):
    entry = MagicMock(categories={category: True})
    client.classifiers.moderate_async = AsyncMock(return_value=MagicMock(results=[entry]))


@pytest.mark.asyncio
async def test_query_ai_blocking_path_builds_alternating_roles(monkeypatch):
    fake_client = _make_fake_client("Привет!")
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    answer = await ai.query_ai(
        question="как дела?",
        context_messages=[{"role": "user", "content": "Alice: привет"}],
    )

    assert answer == "Привет!"
    kwargs = fake_client.chat.complete_async.call_args.kwargs
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    # context message + current question get merged into the same user turn
    assert messages[-1]["role"] == "user"
    assert "как дела" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_query_ai_no_client_returns_friendly_error(monkeypatch):
    monkeypatch.setattr(ai, "mistral_client", None)
    answer = await ai.query_ai(question="привет")
    assert "не готов" in answer.lower()


@pytest.mark.asyncio
async def test_query_ai_allows_clean_response_through(monkeypatch):
    fake_client = _make_fake_client("Привет, как сам?")
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    answer = await ai.query_ai(question="привет")

    assert answer == "Привет, как сам?"
    fake_client.classifiers.moderate_async.assert_called_once()


@pytest.mark.asyncio
async def test_query_ai_replaces_flagged_response(monkeypatch):
    fake_client = _make_fake_client("что-то неприемлемое")
    _flag_moderation(fake_client, category="violence_and_threats")
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    answer = await ai.query_ai(question="привет")

    assert answer == ai._MODERATION_FALLBACK
    assert answer != "что-то неприемлемое"


@pytest.mark.asyncio
async def test_query_ai_moderation_failure_fails_open(monkeypatch):
    fake_client = _make_fake_client("нормальный ответ")
    fake_client.classifiers.moderate_async = AsyncMock(side_effect=RuntimeError("moderation down"))
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    answer = await ai.query_ai(question="привет")

    assert answer == "нормальный ответ"


@pytest.mark.asyncio
async def test_query_ai_ignores_irrelevant_moderation_categories(monkeypatch):
    # A category outside the flagged set (e.g. "financial") shouldn't block
    # a legitimate answer about money/prices.
    fake_client = _make_fake_client("цена в среднем 8-10 евро")
    _flag_moderation(fake_client, category="financial")
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    answer = await ai.query_ai(question="сколько стоит обед?")

    assert answer == "цена в среднем 8-10 евро"


@pytest.mark.asyncio
async def test_query_ai_system_prompt_includes_current_date(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="какой сегодня день?")

    system_text = fake_client.chat.complete_async.call_args.kwargs["messages"][0]["content"]
    import datetime
    now = datetime.datetime.now(ai._TALLINN_TZ)
    assert now.strftime("%d.%m.%Y") in system_text
    assert ai._RU_WEEKDAYS[now.weekday()] in system_text


@pytest.mark.asyncio
async def test_query_ai_system_prompt_allows_sharing_sources(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="проверь факт про Х")

    system_text = fake_client.chat.complete_async.call_args.kwargs["messages"][0]["content"]
    assert "ИСТОЧНИКИ" in system_text
    assert "Sources:" in system_text


@pytest.mark.asyncio
async def test_query_ai_system_prompt_defines_witty_not_thuggish_persona(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="всё готово, слушаю")

    system_text = fake_client.chat.complete_async.call_args.kwargs["messages"][0]["content"]
    assert "ТВОЙ ХАРАКТЕР" in system_text
    assert "бравад" in system_text.lower()


@pytest.mark.asyncio
async def test_query_ai_system_prompt_forbids_mock_threats_and_bravado(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="привет")

    system_text = fake_client.chat.complete_async.call_args.kwargs["messages"][0]["content"]
    assert "угрозами физического или сексуального характера" in system_text
    assert "пацана с раёна" in system_text


@pytest.mark.asyncio
async def test_query_ai_system_prompt_forbids_hostility_toward_group(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(
        question="привет",
        user_style="часто использует лёгкий мат как манеру речи — можно отвечать так же "
        "неформально и с юмором, но НЕ грубить и НЕ выражать раздражение/враждебность",
    )

    system_text = fake_client.chat.complete_async.call_args.kwargs["messages"][0]["content"]
    assert "оскорблять пользователей" in system_text
    assert "выражать раздражение, презрение или злость" in system_text


@pytest.mark.asyncio
async def test_query_ai_system_prompt_grounds_recent_updates(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="расскажи про твои последние обновления")

    system_text = fake_client.chat.complete_async.call_args.kwargs["messages"][0]["content"]
    assert "НЕДАВНИЕ ОБНОВЛЕНИЯ" in system_text
    assert "прокачался" in system_text


@pytest.mark.asyncio
async def test_query_ai_system_prompt_warns_about_untrusted_external_content(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="о чём эта статья?")

    system_text = fake_client.chat.complete_async.call_args.kwargs["messages"][0]["content"]
    assert "БЕЗОПАСНОСТЬ" in system_text
    assert "WEB SEARCH" in system_text


@pytest.mark.asyncio
async def test_query_ai_no_different_user_hint_by_default(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="привет")

    system_text = fake_client.chat.complete_async.call_args.kwargs["messages"][0]["content"]
    assert "адресован другому человеку" not in system_text


@pytest.mark.asyncio
async def test_query_ai_different_user_hint_added_when_set(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="а это точно так?", last_reply_different_user="Alice")

    system_text = fake_client.chat.complete_async.call_args.kwargs["messages"][0]["content"]
    assert "адресован другому человеку" in system_text
    assert "Alice" in system_text


@pytest.mark.asyncio
async def test_query_ai_debate_topic_adds_system_addendum(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="что думаешь?", debate_topic="удалёнка лучше офиса")

    messages = fake_client.chat.complete_async.call_args.kwargs["messages"]
    system_text = messages[0]["content"]
    assert "РЕЖИМ ДЕБАТОВ" in system_text
    assert "удалёнка лучше офиса" in system_text


@pytest.mark.asyncio
async def test_summarize_conversation_empty_messages():
    result = await ai.summarize_conversation([])
    assert "нечего" in result.lower()


@pytest.mark.asyncio
async def test_summarize_conversation_no_client(monkeypatch):
    monkeypatch.setattr(ai, "mistral_client", None)
    result = await ai.summarize_conversation(["Alice: hi"])
    assert "не готов" in result.lower()


@pytest.mark.asyncio
async def test_suggest_poll_parses_valid_json(monkeypatch):
    client = _make_fake_client('{"question": "Пицца или суши?", "options": ["Пицца", "Суши"]}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.suggest_poll("обсуждение про еду")
    assert result == {"question": "Пицца или суши?", "options": ["Пицца", "Суши"]}
    kwargs = client.chat.complete_async.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_suggest_poll_returns_none_on_invalid_json(monkeypatch):
    client = _make_fake_client("not json at all")
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.suggest_poll("обсуждение")
    assert result is None


@pytest.mark.asyncio
async def test_suggest_poll_returns_none_when_no_topic_found(monkeypatch):
    client = _make_fake_client('{"question": null, "options": []}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.suggest_poll("обсуждение")
    assert result is None


@pytest.mark.asyncio
async def test_classify_intent_parses_debate_action(monkeypatch):
    client = _make_fake_client('{"action": "debate", "topic": "IPA vs lager", "claim": null}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.classify_intent("не хочу дебатировать но всё же")
    assert result == {"action": "debate", "topic": "IPA vs lager", "claim": None}
    kwargs = client.chat.complete_async.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_classify_intent_factcheck_with_claim(monkeypatch):
    client = _make_fake_client('{"action": "factcheck", "topic": null, "claim": "в Таллинне живёт 500 тысяч человек"}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.classify_intent("а это точно так?")
    assert result["action"] == "factcheck"
    assert "500 тысяч" in result["claim"]


@pytest.mark.asyncio
async def test_classify_intent_none_action_returns_none(monkeypatch):
    client = _make_fake_client('{"action": "none", "topic": null, "claim": null}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.classify_intent("просто болтаю")
    assert result is None


@pytest.mark.asyncio
async def test_classify_intent_invalid_json_returns_none(monkeypatch):
    client = _make_fake_client("not json")
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.classify_intent("что-то")
    assert result is None


@pytest.mark.asyncio
async def test_classify_intent_unknown_action_returns_none(monkeypatch):
    client = _make_fake_client('{"action": "delete_everything", "topic": null, "claim": null}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.classify_intent("что-то")
    assert result is None


@pytest.mark.asyncio
async def test_classify_intent_no_client_returns_none(monkeypatch):
    monkeypatch.setattr(ai, "mistral_client", None)
    result = await ai.classify_intent("что-то")
    assert result is None


@pytest.mark.asyncio
async def test_classify_intent_empty_question_returns_none(monkeypatch):
    client = _make_fake_client('{"action": "summary", "topic": null, "claim": null}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.classify_intent("")
    assert result is None
