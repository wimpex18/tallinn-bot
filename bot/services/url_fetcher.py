"""URL fetching with curl_cffi (browser TLS impersonation) and content extraction."""

import time
import asyncio
import logging

from curl_cffi.requests import AsyncSession as CurlAsyncSession

from config import URL_CACHE_TTL, IMPERSONATE_PROFILES
from bot.utils.helpers import clean_url, extract_url_info
from bot.utils.html_parser import (
    extract_metadata,
    format_metadata_text,
    extract_page_text,
    is_cloudflare_block,
)

logger = logging.getLogger(__name__)

# Set by main.py post_init
curl_session: CurlAsyncSession = None

# {cleaned_url: (content_str, timestamp)}
_url_cache: dict[str, tuple[str, float]] = {}


def _extract_content_from_html(html: str, url: str) -> str:
    """Combine metadata + article text from raw HTML."""
    metadata = extract_metadata(html)
    metadata_text = format_metadata_text(metadata)
    page_text = extract_page_text(html)

    if metadata_text and len(metadata_text) > 50:
        if page_text and len(page_text) > 50:
            return f"{metadata_text}\n\n[Page content]:\n{page_text}"
        return metadata_text

    if page_text and len(page_text) > 50:
        return page_text

    return ""


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


async def fetch_url_content(url: str) -> str:
    """Fetch webpage content using curl_cffi with browser TLS impersonation.

    Tries multiple impersonation profiles in parallel, returns first success.
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

    if result is None:
        logger.error(f"Failed to fetch {clean_url_str} ({elapsed_ms:.0f}ms)")
        url_info = extract_url_info(clean_url_str)
        result = url_info if url_info else ""
    else:
        logger.info(f"Fetched {len(result)} chars from {clean_url_str} in {elapsed_ms:.0f}ms")

    # Cache (including failures)
    _url_cache[clean_url_str] = (result, time.time())
    if len(_url_cache) > 50:
        expired = [k for k, (_, ts) in _url_cache.items() if time.time() - ts > URL_CACHE_TTL]
        for k in expired:
            del _url_cache[k]

    return result
