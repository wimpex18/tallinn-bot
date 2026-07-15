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
