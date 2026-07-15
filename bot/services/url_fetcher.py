"""URL fetching with curl_cffi (browser TLS impersonation) and content extraction."""

import asyncio
import logging
import time

from curl_cffi.requests import AsyncSession as CurlAsyncSession

from bot.utils.helpers import clean_url, extract_url_info
from bot.utils.html_parser import (
    extract_metadata,
    extract_page_text,
    format_metadata_text,
    is_cloudflare_block,
)
from config import IMPERSONATE_PROFILES, URL_CACHE_TTL, URL_HEAD_CHARS, URL_MAX_CHARS, URL_TAIL_CHARS

logger = logging.getLogger(__name__)

# Set by main.py post_init
curl_session: CurlAsyncSession = None

# {cleaned_url: (content_str, timestamp)}
_url_cache: dict[str, tuple[str, float]] = {}


def _truncate_content(content: str) -> str:
    """Apply head+tail truncation to long content.

    Keeps the opening (title, date, lead paragraph) and closing (prices,
    contacts, conclusions) sections which carry the most useful information,
    dropping the bulk of the middle.
    """
    if len(content) <= URL_MAX_CHARS:
        return content
    head = content[:URL_HEAD_CHARS]
    tail = content[-URL_TAIL_CHARS:]
    omitted = len(content) - URL_HEAD_CHARS - URL_TAIL_CHARS
    return f"{head}\n\n[...{omitted} символов пропущено...]\n\n{tail}"


def _extract_content_from_html(html: str, url: str) -> str:
    """Combine metadata + article text from raw HTML, then truncate if needed."""
    metadata = extract_metadata(html)
    metadata_text = format_metadata_text(metadata)
    page_text = extract_page_text(html)

    has_page_text = bool(page_text and len(page_text) > 50)

    if metadata_text and has_page_text:
        # Always prepend metadata (even a short one, like just a title) when we
        # also have real page text — a short title shouldn't get dropped just
        # because it alone wouldn't clear the "is this substantial" bar below.
        combined = f"{metadata_text}\n\n[Page content]:\n{page_text}"
    elif has_page_text:
        combined = page_text
    elif metadata_text and len(metadata_text) > 50:
        # No usable page text — only fall back to metadata alone if there's
        # enough of it (multiple fields) to be worth returning by itself.
        combined = metadata_text
    else:
        return ""

    return _truncate_content(combined)


async def _curl_fetch(url: str, impersonate: str) -> tuple[str | None, str | None]:
    """Single fetch attempt.  Returns (html, None) or (None, error)."""
    try:
        session = curl_session
        if not session:
            session = CurlAsyncSession()

        response = await session.get(
            url, impersonate=impersonate, timeout=20, allow_redirects=True,
        )

        if response.status_code in (403, 429, 503):
            html = response.text
            if is_cloudflare_block(html):
                return None, "cloudflare"
            return None, f"HTTP {response.status_code}"

        if response.status_code >= 400:
            return None, f"HTTP {response.status_code}"

        html = response.text
        if is_cloudflare_block(html):
            return None, "cloudflare"

        return html, None
    except Exception as e:
        return None, str(e)


async def _fetch_via_mistral_search(url: str) -> str | None:
    """Best-effort fallback: ask Mistral's web-search-enabled model to open and
    summarize the URL directly, for when our own TLS-impersonating fetch got
    blocked or came back empty.

    This runs from a different network (Mistral's infra, not this process), so
    it can succeed against a different class of bot-protection than curl_cffi
    impersonation can — not a guaranteed bypass (paywalls and heavily
    JS-gated sites can still defeat both), but worth trying before giving up
    on getting real content entirely.
    """
    try:
        from bot.services.search import search_web
        query = (
            f"Открой страницу {url} и подробно перескажи её содержание: "
            f"заголовок, дату, основные факты, цены и контакты если есть."
        )
        return await search_web(query)
    except Exception as exc:
        logger.warning(f"Mistral-mediated fetch failed for {url}: {exc}")
        return None


async def fetch_url_content(url: str) -> str:
    """Fetch webpage content using curl_cffi with browser TLS impersonation.

    Tries multiple impersonation profiles in parallel, returns first success.
    Falls back to a Mistral-mediated open-and-summarize attempt (different
    infra/IP, so it clears a different class of bot-protection) before
    finally falling back to a non-fetching URL heuristic.
    """
    clean_url_str = clean_url(url)

    # Cache check
    now = time.time()
    cached = _url_cache.get(clean_url_str)
    if cached:
        content, cached_at = cached
        if now - cached_at < URL_CACHE_TTL:
            logger.info(f"URL cache hit: {clean_url_str}")
            return content
        else:
            del _url_cache[clean_url_str]

    t0 = time.monotonic()
    logger.info(f"Fetching URL: {clean_url_str}")

    tasks = {
        asyncio.create_task(_curl_fetch(clean_url_str, profile)): profile
        for profile in IMPERSONATE_PROFILES
    }

    result = None
    pending = set(tasks.keys())

    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                html, error = task.result()

                if error == "cloudflare":
                    logger.warning(f"Cloudflare block on {clean_url_str}")
                    for t in pending:
                        t.cancel()
                    pending = set()
                    break

                if html is not None:
                    content = _extract_content_from_html(html, clean_url_str)
                    if content:
                        result = content
                    else:
                        result = ""
                        logger.warning(f"No content extracted from {clean_url_str}")
                    for t in pending:
                        t.cancel()
                    pending = set()
                    break

                if error:
                    logger.warning(f"Fetch error ({tasks[task]}): {error}")

            if result is not None:
                break
    except Exception as e:
        logger.error(f"Error in parallel fetch: {e}")

    elapsed_ms = (time.monotonic() - t0) * 1000

    if not result:
        logger.warning(
            f"Direct fetch failed for {clean_url_str} ({elapsed_ms:.0f}ms), "
            f"trying Mistral-mediated fetch"
        )
        mistral_result = await _fetch_via_mistral_search(clean_url_str)
        if mistral_result:
            result = mistral_result
            logger.info(f"Mistral-mediated fetch succeeded for {clean_url_str} ({len(result)} chars)")
        else:
            logger.error(f"Mistral-mediated fetch also failed for {clean_url_str}, falling back to URL heuristic")
            url_info = extract_url_info(clean_url_str)
            result = url_info if url_info else ""
    else:
        logger.info(f"Fetched {len(result)} chars from {clean_url_str} in {elapsed_ms:.0f}ms")

    # Cache (including failures)
    _url_cache[clean_url_str] = (result, time.time())
    if len(_url_cache) > 50:
        now_ts = time.time()
        expired = [k for k, (_, ts) in _url_cache.items() if now_ts - ts > URL_CACHE_TTL]
        for k in expired:
            del _url_cache[k]
        # If still over limit (all entries fresh), evict the oldest
        if len(_url_cache) > 50:
            oldest = sorted(_url_cache, key=lambda k: _url_cache[k][1])
            for k in oldest[:len(_url_cache) - 50]:
                del _url_cache[k]

    return result
