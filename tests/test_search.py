from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.services import ai, search


def test_is_search_trigger_positive():
    assert search.is_search_trigger("найди бар с живой музыкой") is True
    assert search.is_search_trigger("search for jazz clubs") is True


def test_is_search_trigger_negative():
    assert search.is_search_trigger("как дела?") is False


def test_is_search_trigger_empty():
    assert search.is_search_trigger("") is False
    assert search.is_search_trigger(None) is False


def test_is_search_trigger_ignores_quoted_example():
    # Same false-positive class as the intent.py bug: quoting a trigger word
    # as a usage example shouldn't fire a real (costly) web search.
    text = 'в боте теперь можно написать "найди бар" и он поищет'
    assert search.is_search_trigger(text) is False


def test_is_search_trigger_ignores_substring_inside_unrelated_word():
    # "search for" is a real substring of "research for" — must not
    # false-trigger on the unrelated word "research".
    assert search.is_search_trigger("we're doing research for a paper") is False


def test_is_search_trigger_still_matches_real_word():
    assert search.is_search_trigger("найди что-нибудь интересное") is True


def _make_response(text_chunks, reference_urls=None):
    content = [SimpleNamespace(type="text", text=t) for t in text_chunks]
    for url in reference_urls or []:
        content.append(SimpleNamespace(type="tool_reference", url=url))
    entry = SimpleNamespace(content=content)
    return SimpleNamespace(outputs=[entry])


@pytest.mark.asyncio
async def test_search_web_formats_answer_with_sources(monkeypatch):
    client = MagicMock()
    response = _make_response(
        ["В Таллинне сегодня концерт джаза."],
        reference_urls=["https://example.com/event"],
    )
    client.beta.conversations.start_async = AsyncMock(return_value=response)
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await search.search_web("джаз концерты сегодня")

    assert result is not None
    assert "[WEB SEARCH: джаз концерты сегодня]" in result
    assert "концерт джаза" in result
    assert "https://example.com/event" in result


@pytest.mark.asyncio
async def test_search_web_no_client_returns_none(monkeypatch):
    monkeypatch.setattr(ai, "mistral_client", None)
    result = await search.search_web("что-то")
    assert result is None


@pytest.mark.asyncio
async def test_search_web_empty_query_returns_none(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(ai, "mistral_client", client)
    result = await search.search_web("")
    assert result is None


@pytest.mark.asyncio
async def test_search_web_handles_api_error(monkeypatch):
    client = MagicMock()
    client.beta.conversations.start_async = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await search.search_web("что-то")
    assert result is None


@pytest.mark.asyncio
async def test_search_web_no_text_content_returns_none(monkeypatch):
    client = MagicMock()
    response = SimpleNamespace(outputs=[SimpleNamespace(content=[])])
    client.beta.conversations.start_async = AsyncMock(return_value=response)
    monkeypatch.setattr(ai, "mistral_client", client)

    result = await search.search_web("что-то")
    assert result is None
