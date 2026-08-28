from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.services import memory


def _fake_redis_with_pipeline():
    redis_client = MagicMock()
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=None)
    redis_client.pipeline = MagicMock(return_value=pipe)
    redis_client.zcard = AsyncMock(return_value=1)
    redis_client.zremrangebyrank = AsyncMock(return_value=None)
    return redis_client, pipe


@pytest.mark.asyncio
async def test_save_user_fact_refreshes_ttl(monkeypatch):
    redis_client, pipe = _fake_redis_with_pipeline()
    monkeypatch.setattr(memory, "redis_client", redis_client)

    await memory.save_user_fact(42, "likes coffee")

    pipe.zadd.assert_called_once()
    pipe.expire.assert_called_once()
    key, ttl = pipe.expire.call_args.args
    assert key == "user:42:facts"
    assert ttl == memory._KEY_TTL_SECONDS
    pipe.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_user_fact_noop_without_redis(monkeypatch):
    monkeypatch.setattr(memory, "redis_client", None)
    # Should not raise even though there's nothing to write to.
    await memory.save_user_fact(42, "some fact")


@pytest.mark.asyncio
async def test_get_user_facts_empty_without_redis(monkeypatch):
    monkeypatch.setattr(memory, "redis_client", None)
    assert await memory.get_user_facts(42) == []


@pytest.mark.asyncio
async def test_debate_mode_round_trip(monkeypatch):
    redis_client = MagicMock()
    redis_client.set = AsyncMock(return_value=None)
    redis_client.get = AsyncMock(return_value="удалёнка лучше офиса")
    redis_client.delete = AsyncMock(return_value=None)
    monkeypatch.setattr(memory, "redis_client", redis_client)

    await memory.set_debate_mode(1, "удалёнка лучше офиса", thread_id=None, ttl=1800)
    redis_client.set.assert_called_once_with(
        "chat:1:0:debate", "удалёнка лучше офиса", ex=1800,
    )

    topic = await memory.get_debate_topic(1)
    assert topic == "удалёнка лучше офиса"

    await memory.clear_debate_mode(1)
    redis_client.delete.assert_called_once_with("chat:1:0:debate")


@pytest.mark.asyncio
async def test_get_debate_topic_none_without_redis(monkeypatch):
    monkeypatch.setattr(memory, "redis_client", None)
    assert await memory.get_debate_topic(1) is None


def test_extract_facts_from_response_regex_patterns():
    facts = memory.extract_facts_from_response("я живу в Таллинне и люблю кофе", "", "Иван")
    joined = " ".join(facts)
    assert "живёт" in joined or "любит" in joined


def _make_fake_client(response_text: str):
    client = MagicMock()
    message = MagicMock(content=response_text)
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    client.chat.complete_async = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_smart_extract_facts_uses_structured_json_output(monkeypatch):
    from bot.services import ai as ai_module

    client = _make_fake_client('{"facts": ["любит IPA"]}')
    monkeypatch.setattr(ai_module, "mistral_client", client)

    facts = await memory.smart_extract_facts(
        question="какое пиво лучше?", answer="IPA конечно", user_name="Иван",
    )

    assert facts == ["Иван: любит IPA"]
    kwargs = client.chat.complete_async.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_smart_extract_facts_handles_markdown_fenced_json(monkeypatch):
    # The exact real-world failure this fix targets: the model wraps JSON in
    # a markdown code fence, which response_format=json_object should prevent
    # — but the fallback parsing must still degrade gracefully if it happens.
    from bot.services import ai as ai_module

    client = _make_fake_client('```json\n{"facts": ["любит IPA"]}\n```')
    monkeypatch.setattr(ai_module, "mistral_client", client)

    facts = await memory.smart_extract_facts(
        question="какое пиво лучше?", answer="IPA конечно", user_name="Иван",
    )

    assert facts == []


@pytest.mark.asyncio
async def test_extract_facts_from_conversation_uses_structured_json_output(monkeypatch):
    from bot.services import ai as ai_module

    client = _make_fake_client('{"facts": ["Иван: любит IPA", "Мария: едет в Ригу"]}')
    monkeypatch.setattr(ai_module, "mistral_client", client)

    facts = await memory.extract_facts_from_conversation(
        chat_id=1, messages=["Иван: люблю IPA", "Мария: еду в Ригу", "Иван: круто"],
    )

    assert facts == ["Иван: любит IPA", "Мария: едет в Ригу"]
    kwargs = client.chat.complete_async.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
