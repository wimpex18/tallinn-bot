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


def test_is_error_response_recognizes_all_query_ai_fallbacks():
    """Regression coverage: a failed query_ai() reply used to get saved into
    conversation context as if it were a real answer, then resent as fake
    assistant history on every following request (confirmed live — a 429's
    error text showed up twice in the very next request's message log)."""
    assert ai.is_error_response(ai._ERR_NOT_READY)
    assert ai.is_error_response(ai._ERR_AUTH)
    assert ai.is_error_response(ai._ERR_RATE_LIMIT)
    assert ai.is_error_response(ai._ERR_BAD_REQUEST)
    assert ai.is_error_response(ai._ERR_UNEXPECTED)
    assert ai.is_error_response(f"{ai._ERR_SERVER_PREFIX}503)")


def test_is_error_response_false_for_real_answers():
    assert not ai.is_error_response("Таллинн — столица Эстонии.")
    assert not ai.is_error_response("")
    assert not ai.is_error_response(ai._MODERATION_FALLBACK)


def test_log_rate_limit_headers_logs_the_header_dict(caplog):
    """Mistral's actual rate-limit dimension (RPS/TPM/monthly) and reset time
    live in response headers (x-ratelimit-*, retry-after), not the generic
    {"message":"Rate limit exceeded"} body we were already logging — this is
    what lets us see the real reason on the next 429 instead of guessing."""
    exc = Exception("rate limited")
    exc.headers = {"retry-after": "30", "x-ratelimit-remaining-requests": "0"}
    with caplog.at_level("WARNING"):
        ai._log_rate_limit_headers(exc)
    assert "retry-after" in caplog.text
    assert "30" in caplog.text


def test_log_rate_limit_headers_handles_missing_headers_attr(caplog):
    exc = Exception("rate limited")
    with caplog.at_level("WARNING"):
        ai._log_rate_limit_headers(exc)
    assert "no .headers attribute" in caplog.text


def test_extract_text_passes_through_plain_string():
    assert ai._extract_text("Привет!") == "Привет!"


def test_extract_text_handles_none_and_empty():
    assert ai._extract_text(None) == ""
    assert ai._extract_text([]) == ""


def test_extract_text_joins_text_chunks_and_skips_thinking():
    # reasoning_effort="high" (debate mode, /factcheck) returns content as a
    # list: a ThinkChunk (internal reasoning, not for the user) plus one or
    # more TextChunks (the actual answer) — confirmed live in production.
    content = [
        MagicMock(type="thinking", text="скрытые рассуждения модели"),
        MagicMock(type="text", text="Похоже на правду."),
    ]
    assert ai._extract_text(content) == "Похоже на правду."


def test_extract_text_joins_multiple_text_chunks():
    content = [MagicMock(type="text", text="Часть 1. "), MagicMock(type="text", text="Часть 2.")]
    assert ai._extract_text(content) == "Часть 1. Часть 2."


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
async def test_query_ai_defaults_to_none_reasoning_effort(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="как дела?")

    kwargs = fake_client.chat.complete_async.call_args.kwargs
    assert kwargs["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_query_ai_debate_mode_bumps_reasoning_effort(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="что думаешь?", debate_topic="удалёнка лучше офиса")

    kwargs = fake_client.chat.complete_async.call_args.kwargs
    assert kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_query_ai_respects_explicit_reasoning_effort(monkeypatch):
    fake_client = _make_fake_client()
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    await ai.query_ai(question="проверь факт", reasoning_effort="high")

    kwargs = fake_client.chat.complete_async.call_args.kwargs
    assert kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_query_ai_blocking_handles_chunked_content(monkeypatch):
    """Regression test for a real production crash (confirmed via Render logs):

    with reasoning_effort="high" (/factcheck, debate mode), Mistral Small 4
    returns message.content as a list of chunks (ThinkChunk + TextChunk)
    instead of a plain string. _blocking_response used to hand that straight
    to _clean_response's re.sub(), which requires a str, and crashed with
    TypeError("expected string or bytes-like object, got 'list'") — every
    /factcheck reply hit this and fell back to "Что-то пошло не так(".
    """
    fake_client = MagicMock()
    content = [
        MagicMock(type="thinking", text="internal reasoning"),
        MagicMock(type="text", text="Похоже на правду."),
    ]
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    fake_client.chat.complete_async = AsyncMock(return_value=response)
    moderation_entry = MagicMock(categories={})
    fake_client.classifiers.moderate_async = AsyncMock(
        return_value=MagicMock(results=[moderation_entry])
    )
    monkeypatch.setattr(ai, "mistral_client", fake_client)

    answer = await ai.query_ai(question="проверь факт", reasoning_effort="high")

    assert answer == "Похоже на правду."


class _FakeStream:
    """Minimal async-context-manager + async-iterator fake for client.chat.stream_async."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_stream_response_handles_chunked_delta_content():
    """Same production bug as blocking (list content with a ThinkChunk before
    the TextChunk), but on the streaming delta path used by debate mode."""
    events = [
        MagicMock(data=MagicMock(choices=[MagicMock(
            delta=MagicMock(content=[MagicMock(type="thinking", text="reasoning...")])
        )])),
        MagicMock(data=MagicMock(choices=[MagicMock(
            delta=MagicMock(content=[MagicMock(type="text", text="Финальный ответ.")])
        )])),
    ]
    fake_client = MagicMock()
    fake_client.chat.stream_async = AsyncMock(return_value=_FakeStream(events))
    telegram_bot = MagicMock(edit_message_text=AsyncMock())

    result = await ai._stream_response(
        fake_client, [{"role": "user", "content": "test"}],
        telegram_bot, chat_id=1, message_id=2, reasoning_effort="high",
    )

    assert result == "Финальный ответ."


@pytest.mark.asyncio
async def test_stream_response_throttles_before_calling(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.stream_async = AsyncMock(return_value=_FakeStream([]))
    throttle_mock = AsyncMock()
    monkeypatch.setattr(ai, "throttle_call", throttle_mock)

    await ai._stream_response(
        fake_client, [{"role": "user", "content": "test"}],
        MagicMock(edit_message_text=AsyncMock()), chat_id=1, message_id=2,
    )

    throttle_mock.assert_awaited_once()


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
async def test_suggest_quiz_parses_valid_json(monkeypatch):
    client = _make_fake_client(
        '{"question": "Столица Эстонии?", "options": ["Таллинн", "Тарту", "Нарва", "Пярну"], '
        '"correct_option_id": 0}',
    )
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.suggest_quiz("Эстония")

    assert result == {
        "question": "Столица Эстонии?",
        "options": ["Таллинн", "Тарту", "Нарва", "Пярну"],
        "correct_option_id": 0,
    }
    kwargs = client.chat.complete_async.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "Эстония" in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_suggest_quiz_no_topic_uses_general_prompt(monkeypatch):
    client = _make_fake_client('{"question": "Q?", "options": ["A", "B"], "correct_option_id": 1}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.suggest_quiz()

    assert result["correct_option_id"] == 1
    kwargs = client.chat.complete_async.call_args.kwargs
    assert "Таллинн" in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_suggest_quiz_rejects_out_of_range_correct_id(monkeypatch):
    client = _make_fake_client('{"question": "Q?", "options": ["A", "B"], "correct_option_id": 5}')
    monkeypatch.setattr(ai, "mistral_client", client)

    assert await ai.suggest_quiz() is None


@pytest.mark.asyncio
async def test_suggest_quiz_returns_none_on_invalid_json(monkeypatch):
    client = _make_fake_client("not json")
    monkeypatch.setattr(ai, "mistral_client", client)

    assert await ai.suggest_quiz() is None


@pytest.mark.asyncio
async def test_suggest_quiz_no_client_returns_none(monkeypatch):
    monkeypatch.setattr(ai, "mistral_client", None)
    assert await ai.suggest_quiz() is None


@pytest.mark.asyncio
async def test_classify_intent_parses_debate_action(monkeypatch):
    client = _make_fake_client('{"action": "debate", "topic": "IPA vs lager", "claim": null}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.classify_intent("не хочу дебатировать но всё же")
    assert result == {"action": "debate", "topic": "IPA vs lager", "claim": None}
    kwargs = client.chat.complete_async.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_classify_intent_parses_quiz_action(monkeypatch):
    client = _make_fake_client('{"action": "quiz", "topic": "Эстония", "claim": null}')
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await ai.classify_intent("устроим что-то типа викторинки?")
    assert result == {"action": "quiz", "topic": "Эстония", "claim": None}


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


# ── Rate limiting (throttle_call) ─────────────────────────────────────
# Regression coverage for a real production incident: every reply started
# failing with "Слишком много запросов, подожди минутку (429)" because the
# free tier caps at ~1 request/s and this process makes 2+ Mistral calls per
# reply (completion + moderation) under concurrent_updates=True.

@pytest.mark.asyncio
async def test_throttle_call_does_not_wait_when_interval_elapsed(monkeypatch):
    monkeypatch.setattr(ai, "_last_call_at", 0.0)
    sleep_mock = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep_mock)

    await ai.throttle_call()

    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_throttle_call_waits_when_called_too_soon(monkeypatch):
    # Pin _last_call_at to "now" (real monotonic time — patching the global
    # time.monotonic breaks asyncio's own internal scheduling) so the very
    # next call falls well inside the minimum interval and must wait.
    import time as real_time
    monkeypatch.setattr(ai, "_last_call_at", real_time.monotonic())
    sleep_mock = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep_mock)

    await ai.throttle_call()

    sleep_mock.assert_awaited_once()
    waited = sleep_mock.call_args.args[0]
    assert 0 < waited <= ai._MISTRAL_MIN_CALL_INTERVAL


@pytest.mark.asyncio
async def test_blocking_response_throttles_before_calling(monkeypatch):
    fake_client = _make_fake_client("ответ")
    throttle_mock = AsyncMock()
    monkeypatch.setattr(ai, "throttle_call", throttle_mock)

    await ai._blocking_response(fake_client, [{"role": "user", "content": "привет"}])

    throttle_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_moderate_own_response_throttles_before_calling(monkeypatch):
    fake_client = _make_fake_client()
    throttle_mock = AsyncMock()
    monkeypatch.setattr(ai, "throttle_call", throttle_mock)

    await ai._moderate_own_response(fake_client, "какой-то ответ")

    throttle_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_ai_retries_once_on_429_then_succeeds(monkeypatch):
    fake_client = _make_fake_client("успешный ответ после ретрая")
    monkeypatch.setattr(ai, "mistral_client", fake_client)
    monkeypatch.setattr(ai, "_429_RETRY_DELAY", 0.0)

    rate_limit_error = Exception("rate limited")
    rate_limit_error.status_code = 429
    call_count = 0
    real_complete = fake_client.chat.complete_async

    async def flaky(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise rate_limit_error
        return await real_complete(*args, **kwargs)

    fake_client.chat.complete_async = AsyncMock(side_effect=flaky)

    answer = await ai.query_ai(question="привет")

    assert answer == "успешный ответ после ретрая"
    assert call_count == 2


@pytest.mark.asyncio
async def test_query_ai_gives_up_after_max_429_retries(monkeypatch):
    fake_client = MagicMock()
    rate_limit_error = Exception("rate limited")
    rate_limit_error.status_code = 429
    fake_client.chat.complete_async = AsyncMock(side_effect=rate_limit_error)
    monkeypatch.setattr(ai, "mistral_client", fake_client)
    monkeypatch.setattr(ai, "_429_RETRY_DELAY", 0.0)

    answer = await ai.query_ai(question="привет")

    assert "429" in answer
    assert fake_client.chat.complete_async.call_count == ai._MAX_429_RETRIES + 1


@pytest.mark.asyncio
async def test_query_ai_surfaces_rate_limit_headers_on_final_failure(monkeypatch, caplog):
    fake_client = MagicMock()
    rate_limit_error = Exception("rate limited")
    rate_limit_error.status_code = 429
    rate_limit_error.headers = {"x-ratelimit-remaining-tokens-minute": "0", "retry-after": "60"}
    fake_client.chat.complete_async = AsyncMock(side_effect=rate_limit_error)
    monkeypatch.setattr(ai, "mistral_client", fake_client)
    monkeypatch.setattr(ai, "_429_RETRY_DELAY", 0.0)

    with caplog.at_level("WARNING"):
        await ai.query_ai(question="привет")

    assert "x-ratelimit-remaining-tokens-minute" in caplog.text
    assert "retry-after" in caplog.text
