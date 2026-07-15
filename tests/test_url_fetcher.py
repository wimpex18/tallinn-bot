from unittest.mock import AsyncMock

import pytest

from bot.services import search as search_module
from bot.services import url_fetcher


@pytest.fixture(autouse=True)
def _clear_url_cache():
    url_fetcher._url_cache.clear()
    yield
    url_fetcher._url_cache.clear()


async def _fail_all(url, impersonate):
    return None, "connection error"


@pytest.mark.asyncio
async def test_fetch_url_content_falls_back_to_mistral_search_when_direct_fetch_fails(monkeypatch):
    monkeypatch.setattr(url_fetcher, "_curl_fetch", _fail_all)
    monkeypatch.setattr(
        search_module, "search_web", AsyncMock(return_value="[WEB SEARCH: url] mistral-fetched content"),
    )

    result = await url_fetcher.fetch_url_content("https://example.com/blocked-page")

    assert result == "[WEB SEARCH: url] mistral-fetched content"


@pytest.mark.asyncio
async def test_fetch_url_content_falls_back_to_url_heuristic_when_both_tiers_fail(monkeypatch):
    monkeypatch.setattr(url_fetcher, "_curl_fetch", _fail_all)
    monkeypatch.setattr(search_module, "search_web", AsyncMock(return_value=None))

    result = await url_fetcher.fetch_url_content("https://example.com/still-blocked")

    assert "PAGE NOT ACCESSIBLE" in result


@pytest.mark.asyncio
async def test_fetch_url_content_skips_mistral_fallback_when_direct_fetch_succeeds(monkeypatch):
    async def succeed(url, impersonate):
        html = (
            "<html><head><title>Real Page</title></head><body><p>"
            + ("Genuine article content. " * 10)
            + "</p></body></html>"
        )
        return html, None

    search_mock = AsyncMock()
    monkeypatch.setattr(url_fetcher, "_curl_fetch", succeed)
    monkeypatch.setattr(search_module, "search_web", search_mock)

    result = await url_fetcher.fetch_url_content("https://example.com/works-fine")

    assert "Real Page" in result
    assert "Genuine article content" in result
    search_mock.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_url_content_falls_back_when_content_extraction_is_empty(monkeypatch):
    async def succeed_but_empty(url, impersonate):
        # Fetched fine, but nothing extractable (e.g. a JS-only shell page).
        return "<html><body></body></html>", None

    monkeypatch.setattr(url_fetcher, "_curl_fetch", succeed_but_empty)
    monkeypatch.setattr(
        search_module, "search_web", AsyncMock(return_value="[WEB SEARCH: url] rescued content"),
    )

    result = await url_fetcher.fetch_url_content("https://example.com/empty-shell")

    assert result == "[WEB SEARCH: url] rescued content"


def test_extract_content_keeps_short_title_alongside_long_page_text():
    # Regression test: a title alone ("Title: Real Page", ~17 chars) sits below
    # the 50-char "is metadata substantial enough on its own" bar, but it must
    # not be silently dropped just because of that when there's also real page
    # text to go with it.
    html = (
        "<html><head><title>Real Page</title></head><body><p>"
        + ("Genuine article content. " * 10)
        + "</p></body></html>"
    )
    result = url_fetcher._extract_content_from_html(html, "https://example.com/works-fine")
    assert "Real Page" in result
    assert "Genuine article content" in result


def test_extract_content_short_metadata_alone_without_page_text_is_dropped():
    # No real page text, and metadata alone is too thin (~17 chars) to be
    # worth returning by itself — should return "".
    html = "<html><head><title>Real Page</title></head><body></body></html>"
    result = url_fetcher._extract_content_from_html(html, "https://example.com/thin-page")
    assert result == ""
