import os
import re
import time
import json
import asyncio
import logging
import base64
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import httpx
import trafilatura
import redis.asyncio as aioredis
from curl_cffi.requests import AsyncSession as CurlAsyncSession
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
BOT_USERNAME = os.getenv("BOT_USERNAME", "")
REDIS_URL = os.getenv("REDIS_URL")

# Rate limiting
user_last_query: dict[int, float] = defaultdict(float)
RATE_LIMIT_SECONDS = 5

# Conversation context
CONTEXT_SIZE = 10
chat_context: dict[int, list[dict]] = defaultdict(list)

# Username to name mapping
USERNAME_TO_NAME = {
    "Vitalina_Bohaichuk": "Виталина",
    "hramus": "Миша",
    "I_lovet": "Полина",
    "Psychonauter": "Миша",
    "wimpex18": "Сергей",
}

# Redis connection for persistent memory (initialized async in post_init)
redis_client = None

# Global HTTP clients
http_client: httpx.AsyncClient = None  # For Perplexity API calls (no impersonation needed)
curl_session: CurlAsyncSession = None  # For URL fetching (browser TLS impersonation)

# URL fetch cache: {cleaned_url: (content, timestamp)}
URL_CACHE_TTL = 300  # 5 minutes
_url_cache: dict[str, tuple[str, float]] = {}

# In-memory eviction settings
CONTEXT_MAX_AGE = 3600  # 1 hour — evict chat contexts older than this
RATE_LIMIT_MAX_AGE = 300  # 5 min — evict stale rate limit entries
_last_eviction: float = 0.0
EVICTION_INTERVAL = 300  # run eviction check every 5 minutes


# ============ MEMORY FUNCTIONS ============

async def save_user_fact(user_id: int, fact: str) -> None:
    """Save a fact about a user to persistent memory (sorted set, newest kept)."""
    if not redis_client:
        return
    try:
        key = f"user:{user_id}:facts"
        await redis_client.zadd(key, {fact: time.time()})
        count = await redis_client.zcard(key)
        if count > 20:
            # Remove oldest entries, keep newest 20
            await redis_client.zremrangebyrank(key, 0, -(20 + 1))
    except Exception as e:
        logger.error(f"Failed to save user fact: {e}")


async def get_user_facts(user_id: int) -> list[str]:
    """Get all facts about a user from memory (ordered oldest to newest)."""
    if not redis_client:
        return []
    try:
        return await redis_client.zrange(f"user:{user_id}:facts", 0, -1)
    except Exception as e:
        logger.error(f"Failed to get user facts: {e}")
        return []


async def save_group_fact(chat_id: int, fact: str) -> None:
    """Save a fact about the group to persistent memory (sorted set, newest kept)."""
    if not redis_client:
        return
    try:
        key = f"group:{chat_id}:facts"
        await redis_client.zadd(key, {fact: time.time()})
        count = await redis_client.zcard(key)
        if count > 30:
            await redis_client.zremrangebyrank(key, 0, -(30 + 1))
    except Exception as e:
        logger.error(f"Failed to save group fact: {e}")


async def get_group_facts(chat_id: int) -> list[str]:
    """Get all facts about the group from memory (ordered oldest to newest)."""
    if not redis_client:
        return []
    try:
        return await redis_client.zrange(f"group:{chat_id}:facts", 0, -1)
    except Exception as e:
        logger.error(f"Failed to get group facts: {e}")
        return []


# ============ CONTEXT FUNCTIONS ============

def add_to_context(chat_id: int, role: str, name: str, content: str) -> None:
    """Add a message to the chat context."""
    chat_context[chat_id].append({
        "role": role,
        "name": name,
        "content": content[:500],
        "time": time.time()
    })
    if len(chat_context[chat_id]) > CONTEXT_SIZE:
        chat_context[chat_id] = chat_context[chat_id][-CONTEXT_SIZE:]


def get_context_string(chat_id: int) -> str:
    """Get recent conversation context as a string."""
    if not chat_context[chat_id]:
        return ""
    context_lines = []
    for msg in chat_context[chat_id][-CONTEXT_SIZE:]:
        name = msg.get("name", "user")
        context_lines.append(f"{name}: {msg['content']}")
    return "\n".join(context_lines)


def evict_stale_data() -> None:
    """Remove stale entries from in-memory dicts to prevent unbounded growth.

    Called periodically from handle_message (every EVICTION_INTERVAL seconds).
    """
    global _last_eviction
    now = time.time()
    if now - _last_eviction < EVICTION_INTERVAL:
        return
    _last_eviction = now

    # Evict chat contexts with no recent messages
    stale_chats = [
        chat_id for chat_id, msgs in chat_context.items()
        if msgs and now - msgs[-1].get("time", 0) > CONTEXT_MAX_AGE
    ]
    for chat_id in stale_chats:
        del chat_context[chat_id]

    # Evict expired rate limit entries
    stale_users = [
        uid for uid, ts in user_last_query.items()
        if now - ts > RATE_LIMIT_MAX_AGE
    ]
    for uid in stale_users:
        del user_last_query[uid]

    if stale_chats or stale_users:
        logger.info(f"Evicted {len(stale_chats)} stale contexts, {len(stale_users)} rate limit entries")


# ============ UTILITY FUNCTIONS ============

def check_rate_limit(user_id: int) -> tuple[bool, int]:
    """Check if user is rate limited. Returns (is_limited, seconds_remaining)."""
    now = time.time()
    last_query = user_last_query[user_id]
    if last_query and now - last_query < RATE_LIMIT_SECONDS:
        remaining = int(RATE_LIMIT_SECONDS - (now - last_query))
        return True, remaining
    return False, 0


def set_rate_limit(user_id: int) -> None:
    """Set rate limit timestamp after successful query."""
    user_last_query[user_id] = time.time()


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def extract_urls_from_entities(message) -> list[str]:
    """Extract URLs from message entities (hyperlinks)."""
    urls = []
    if not message:
        return urls

    # Get entities from text or caption
    entities = message.entities or message.caption_entities or []
    text = message.text or message.caption or ""

    for entity in entities:
        # TEXT_LINK: text that links to a URL (e.g., "click here" -> url)
        if entity.type == "text_link" and entity.url:
            urls.append(entity.url)
        # URL: plain URL visible in text
        elif entity.type == "url":
            url = text[entity.offset:entity.offset + entity.length]
            if url:
                urls.append(url)

    return urls


def get_all_urls(message) -> list[str]:
    """Get all URLs from message: plain text + hyperlink entities."""
    urls = []

    # URLs from entities (hyperlinks)
    urls.extend(extract_urls_from_entities(message))

    # URLs from plain text (fallback)
    text = get_message_content(message)
    if text:
        text_urls = extract_urls(text)
        for url in text_urls:
            if url not in urls:
                urls.append(url)

    return urls


def get_message_content(message) -> str:
    """Extract text content from a message."""
    if message.text:
        return message.text
    if message.caption:
        return message.caption
    return ""


def clean_url(url: str) -> str:
    """Remove tracking parameters from URL (fbclid, utm_*, etc.)."""
    try:
        parsed = urlparse(url)
        # Parse query parameters
        params = parse_qs(parsed.query, keep_blank_values=False)

        # List of tracking parameters to remove
        tracking_params = {
            'fbclid', 'gclid', 'utm_source', 'utm_medium', 'utm_campaign',
            'utm_term', 'utm_content', 'ref', 'source', 'mc_cid', 'mc_eid',
            'aem_', '_ga', 'yclid', 'wickedid', 'twclid', 'ttclid'
        }

        # Filter out tracking parameters
        cleaned_params = {
            k: v for k, v in params.items()
            if k.lower() not in tracking_params and not k.lower().startswith(('utm_', 'aem_'))
        }

        # Rebuild query string
        new_query = urlencode(cleaned_params, doseq=True)

        # Rebuild URL
        cleaned = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            ''  # Remove fragment
        ))

        return cleaned
    except Exception:
        return url


def extract_metadata(html: str) -> dict:
    """Extract metadata from HTML (og:*, meta description, title, JSON-LD)."""
    metadata = {}

    # Extract <title>
    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    if title_match:
        metadata['title'] = title_match.group(1).strip()

    # Extract Open Graph tags (og:title, og:description, og:type, etc.)
    og_patterns = [
        ('og_title', r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']'),
        ('og_title', r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:title["\']'),
        ('og_description', r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']'),
        ('og_description', r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:description["\']'),
        ('og_site_name', r'<meta[^>]*property=["\']og:site_name["\'][^>]*content=["\']([^"\']+)["\']'),
        ('og_site_name', r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:site_name["\']'),
    ]

    for key, pattern in og_patterns:
        if key not in metadata:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()

    # Extract meta description
    desc_patterns = [
        r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
        r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']description["\']',
    ]
    for pattern in desc_patterns:
        if 'description' not in metadata:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                metadata['description'] = match.group(1).strip()

    # Extract ALL JSON-LD blocks (some pages have multiple)
    jsonld_matches = re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE
    )
    for jsonld_text in jsonld_matches:
        try:
            jsonld_data = json.loads(jsonld_text.strip())
            items = jsonld_data if isinstance(jsonld_data, list) else [jsonld_data]
            for item in items:
                if isinstance(item, dict):
                    _extract_jsonld_item(item, metadata)
        except json.JSONDecodeError:
            pass

    # Detect paywall from HTML patterns (fallback if JSON-LD didn't catch it)
    if 'is_paywalled' not in metadata:
        paywall_indicators = [
            'paywall', 'piano-paywall', 'reg-wall', 'subscribe-wall',
            'premium-content', 'locked-content', 'article__pw',
        ]
        html_lower = html.lower()
        if any(ind in html_lower for ind in paywall_indicators):
            metadata['is_paywalled'] = True

    return metadata


def _extract_jsonld_item(data: dict, metadata: dict) -> None:
    """Extract structured data from JSON-LD (events, articles, etc.)."""
    schema_type = str(data.get('@type', ''))

    # Detect paywall from JSON-LD (most reliable signal)
    if data.get('isAccessibleForFree') is False:
        metadata['is_paywalled'] = True

    # Handle Article/NewsArticle types
    if any(t in schema_type for t in ('Article', 'NewsArticle', 'BlogPosting', 'WebPage')):
        if 'headline' in data and 'article_headline' not in metadata:
            metadata['article_headline'] = data['headline']
        if 'description' in data and 'article_description' not in metadata:
            metadata['article_description'] = data['description'][:500]
        if 'author' in data:
            author = data['author']
            if isinstance(author, dict):
                metadata['author'] = author.get('name', '')
            elif isinstance(author, list):
                names = [a.get('name', '') for a in author if isinstance(a, dict)]
                if names:
                    metadata['author'] = ', '.join(names)
            elif isinstance(author, str):
                metadata['author'] = author
        if 'datePublished' in data:
            metadata['date_published'] = data['datePublished']

    # Handle Event type
    if 'Event' in schema_type:
        if 'name' in data:
            metadata['event_name'] = data['name']
        if 'startDate' in data:
            metadata['event_date'] = data['startDate']
        if 'endDate' in data:
            metadata['event_end_date'] = data['endDate']
        if 'description' in data:
            metadata['event_description'] = data['description']

        # Location info
        location = data.get('location', {})
        if isinstance(location, dict):
            if 'name' in location:
                metadata['venue'] = location['name']
            address = location.get('address', {})
            if isinstance(address, dict):
                metadata['address'] = address.get('streetAddress', '')
            elif isinstance(address, str):
                metadata['address'] = address

        # Performer info
        performer = data.get('performer', {})
        if isinstance(performer, dict):
            if 'name' in performer:
                metadata['performer'] = performer['name']
        elif isinstance(performer, list) and performer:
            names = [p.get('name', '') for p in performer if isinstance(p, dict) and 'name' in p]
            if names:
                metadata['performer'] = ', '.join(names)

        # Offers/tickets
        offers = data.get('offers', {})
        if isinstance(offers, dict):
            if 'price' in offers:
                metadata['price'] = f"{offers.get('price', '')} {offers.get('priceCurrency', '')}"
            if 'url' in offers:
                metadata['ticket_url'] = offers['url']


def format_metadata_text(metadata: dict) -> str:
    """Format extracted metadata into readable text."""
    parts = []

    # Paywall notice
    if metadata.get('is_paywalled'):
        parts.append("[PAYWALL: article is behind a paywall, only preview available]")

    # Event-specific formatting
    if 'event_name' in metadata:
        parts.append(f"Event: {metadata['event_name']}")
    elif 'article_headline' in metadata:
        parts.append(f"Article: {metadata['article_headline']}")
    elif 'og_title' in metadata:
        parts.append(f"Title: {metadata['og_title']}")
    elif 'title' in metadata:
        parts.append(f"Title: {metadata['title']}")

    if 'author' in metadata:
        parts.append(f"Author: {metadata['author']}")

    if 'date_published' in metadata:
        parts.append(f"Published: {metadata['date_published']}")

    if 'performer' in metadata:
        parts.append(f"Artist/Performer: {metadata['performer']}")

    if 'event_date' in metadata:
        parts.append(f"Date: {metadata['event_date']}")

    if 'venue' in metadata:
        venue_str = metadata['venue']
        if 'address' in metadata and metadata['address']:
            venue_str += f", {metadata['address']}"
        parts.append(f"Venue: {venue_str}")

    if 'price' in metadata:
        parts.append(f"Price: {metadata['price']}")

    if 'event_description' in metadata:
        desc = metadata['event_description'][:500]
        parts.append(f"Description: {desc}")
    elif 'article_description' in metadata:
        desc = metadata['article_description'][:500]
        parts.append(f"Description: {desc}")
    elif 'og_description' in metadata:
        desc = metadata['og_description'][:500]
        parts.append(f"Description: {desc}")
    elif 'description' in metadata:
        desc = metadata['description'][:500]
        parts.append(f"Description: {desc}")

    return '\n'.join(parts)


# Browser impersonation targets for curl_cffi (TLS fingerprint level)
IMPERSONATE_PROFILES = ["chrome", "safari"]


def _extract_content_from_html(html: str, url: str) -> str:
    """Extract content from successfully fetched HTML.

    Structured fallback chain:
    1. Metadata (OG tags, JSON-LD) — always extracted
    2. Article text via trafilatura — best for news/blog content
    3. Regex fallback — for non-article pages
    Combines metadata + body text for maximum context.
    """
    # Layer 1: Structured metadata (always try)
    metadata = extract_metadata(html)
    metadata_text = format_metadata_text(metadata)

    # Layer 2: Article body text (trafilatura > regex fallback)
    page_text = extract_page_text(html)

    # Combine: metadata header + body content
    if metadata_text and len(metadata_text) > 50:
        if page_text and len(page_text) > 50:
            return f"{metadata_text}\n\n[Page content]:\n{page_text}"
        return metadata_text

    if page_text and len(page_text) > 50:
        return page_text

    return ""


async def _curl_fetch(url: str, impersonate: str) -> tuple[str | None, str | None]:
    """Single curl_cffi fetch attempt with browser TLS impersonation.

    Returns (html, None) on success, (None, error_string) on failure.
    """
    try:
        session = curl_session
        if not session:
            # Fallback if global session not yet initialized
            session = CurlAsyncSession()

        response = await session.get(
            url,
            impersonate=impersonate,
            timeout=20,
            allow_redirects=True,
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

    Uses curl_cffi which produces real Chrome/Safari TLS fingerprints,
    bypassing WAF checks that block Python HTTP libraries.
    Tries multiple impersonation profiles in parallel.
    """
    clean_url_str = clean_url(url)

    # Check cache first
    now = time.time()
    cached = _url_cache.get(clean_url_str)
    if cached:
        content, cached_at = cached
        if now - cached_at < URL_CACHE_TTL:
            logger.info(f"URL cache hit: {clean_url_str}")
            return content
        else:
            del _url_cache[clean_url_str]

    logger.info(f"Fetching URL: {clean_url_str}")

    # Launch parallel fetches with different browser impersonation profiles
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
                        logger.info(f"Fetched {len(content)} chars from {clean_url_str}")
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

    # All attempts failed — extract minimal info from URL
    if result is None:
        logger.error(f"Failed to fetch {clean_url_str}")
        url_info = extract_url_info(clean_url_str)
        result = url_info if url_info else ""

    # Cache the result
    _url_cache[clean_url_str] = (result, time.time())
    if len(_url_cache) > 50:
        expired = [k for k, (_, ts) in _url_cache.items() if time.time() - ts > URL_CACHE_TTL]
        for k in expired:
            del _url_cache[k]

    return result


def is_cloudflare_block(html: str) -> bool:
    """Detect Cloudflare bot protection page."""
    indicators = [
        'just a moment',
        'checking your browser',
        'cloudflare',
        'ray id',
        'please wait',
        'ddos protection',
        'enable javascript',
    ]
    html_lower = html.lower()
    return any(indicator in html_lower for indicator in indicators)


def extract_url_info(url: str) -> str:
    """Extract minimal factual info from URL when page cannot be fetched.

    IMPORTANT: Only returns domain and platform type. Never interprets URL path
    segments as event names or content — this caused AI hallucinations.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')

        # Known platform descriptions (factual, no content interpretation)
        platform_names = {
            'tickettailor.com': 'TicketTailor (ticket sales platform)',
            'eventbrite.com': 'Eventbrite (event platform)',
            'facebook.com': 'Facebook',
            'piletilevi.ee': 'Piletilevi (Estonian ticket platform)',
            'fienta.com': 'Fienta (Baltic ticket platform)',
            'piletimaailm.com': 'Piletimaailm (Estonian ticket platform)',
        }

        # Determine platform name
        platform = None
        for site, name in platform_names.items():
            if site in domain:
                platform = name
                break

        info = ["[PAGE NOT ACCESSIBLE - content could not be loaded]"]
        info.append(f"Site: {platform or domain}")
        info.append(f"URL: {url}")

        # Only extract event name from Eventbrite (reliable slug format: event-name-tickets-ID)
        if 'eventbrite.com' in domain:
            path_parts = [p for p in parsed.path.split('/') if p]
            for part in path_parts:
                if '-tickets-' in part:
                    event_name = part.split('-tickets-')[0].replace('-', ' ').title()
                    info.append(f"Possible event name from URL: {event_name}")
                    break

        info.append("")
        info.append("CRITICAL: The page content is NOT available. URL path segments are NOT reliable.")
        info.append("You MUST search the web for this URL or event to find actual details.")
        info.append("DO NOT interpret or guess based on URL slugs, organizer IDs, or path fragments.")

        return '\n'.join(info)
    except Exception as e:
        logger.error(f"Error extracting URL info: {e}")
        return ""


def extract_page_text(html: str) -> str:
    """Extract readable article text from HTML using trafilatura.

    trafilatura (F1=0.958) accurately separates article content from
    navigation, ads, and boilerplate. Falls back to basic regex stripping
    if trafilatura returns nothing (e.g., non-article pages).
    """
    # Primary: trafilatura (handles articles, blog posts, news)
    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,  # prefer getting more text over precision
        )
        if text and len(text.strip()) > 50:
            if len(text) > 3000:
                text = text[:3000] + "..."
            return text
    except Exception as e:
        logger.warning(f"trafilatura extraction failed: {e}")

    # Fallback: basic regex stripping (for non-article pages)
    cleaned = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<nav[^>]*>.*?</nav>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<footer[^>]*>.*?</footer>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', cleaned)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')

    if len(text) > 3000:
        text = text[:3000] + "..."
    return text


def is_forwarded_message(message) -> bool:
    """Check if message is forwarded (PTB v21+ uses forward_origin)."""
    if not message:
        return False
    return message.forward_origin is not None


async def download_photo_as_base64(photo, bot) -> str:
    """Download a photo from Telegram and convert to base64."""
    try:
        # Get the file
        file = await bot.get_file(photo.file_id)

        # Download file bytes
        photo_bytes = await file.download_as_bytearray()

        # Convert to base64
        base64_string = base64.b64encode(photo_bytes).decode('utf-8')

        # Telegram photos are always JPEG, but check file path for PNG
        mime_type = "image/jpeg"
        if hasattr(file, 'file_path') and file.file_path:
            if file.file_path.endswith('.png'):
                mime_type = "image/png"
            logger.info(f"Photo MIME type: {mime_type}")

        return f"data:{mime_type};base64,{base64_string}"
    except Exception as e:
        logger.error(f"Failed to download photo: {e}")
        return None


def has_photo(message) -> bool:
    """Check if message has photo attachments."""
    if not message:
        return False
    return message.photo is not None and len(message.photo) > 0


async def send_typing(bot, chat_id: int) -> None:
    """Send typing action once."""
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass


# ============ PERPLEXITY API ============

async def query_perplexity(
    question: str,
    referenced_content: str = None,
    user_name: str = None,
    context: str = None,
    user_facts: list[str] = None,
    group_facts: list[str] = None,
    photo_urls: list[str] = None,
) -> str:
    """Query Perplexity API with context, memory, and photos."""
    # System prompt - respond in Russian but search in English/Estonian for Tallinn
    system_prompt = (
        'Отвечай на русском. Используй "ты". Кратко, 2-4 предложения. Без эмодзи. '
        'ВАЖНО: При поиске информации о местах, событиях и мероприятиях в Таллинне - '
        'ищи на АНГЛИЙСКОМ и ЭСТОНСКОМ языках (не на русском), так как большинство '
        'актуальной информации о Таллинне на этих языках. '
        'Хорошие источники: Facebook Events, visitestonia.com, tallinn.ee. '
        'Если не находишь на английском/эстонском - попробуй gloss.ee (русскоязычный сайт о Таллинне). '
        'Если видишь "[PAGE NOT ACCESSIBLE]" - страница не загрузилась. '
        'СТРОГО ЗАПРЕЩЕНО угадывать содержание по URL-адресу или частям ссылки. '
        'Вместо этого ПОИЩИ информацию по этой ссылке или событию через веб-поиск. '
        'Если не нашёл - честно скажи что страница недоступна и ты не смог найти информацию. '
        'Если видишь "[PAYWALL]" - статья за пейволлом, доступен только превью. '
        'Расскажи что есть из превью и упомяни что полная статья доступна по подписке.'
    )

    # Add memory context
    if user_facts:
        system_prompt += f"\n\nТы помнишь про этого человека: {', '.join(user_facts[:5])}"
    if group_facts:
        system_prompt += f"\n\nТы помнишь про эту группу: {', '.join(group_facts[:5])}"

    # Auto-add location for place/event-related queries
    question_lower = question.lower()
    place_keywords = [
        # Places
        "бар", "ресторан", "кафе", "клуб", "кино", "магазин", "музей", "театр", "галерея",
        # Events
        "концерт", "мероприятие", "событие", "фестиваль", "выставка", "вечеринка", "шоу",
        "ивент", "event", "афиша", "тусовка", "движ",
        # Time-related (implies looking for events)
        "сегодня", "завтра", "выходные", "вечером", "weekend",
        # Actions
        "куда", "где", "посоветуй", "порекомендуй", "подскажи", "сходить", "пойти"
    ]
    location_keywords = ["таллин", "tallinn", "эстони", "estonia"]

    # Append location in English for better search results
    if any(kw in question_lower for kw in place_keywords) and not any(loc in question_lower for loc in location_keywords):
        question = f"{question} (Tallinn, Estonia)"

    # Build user message
    if referenced_content:
        user_message = f"{referenced_content}\n\nВопрос: {question}"
    else:
        user_message = question

    # Build message content (with photos if provided)
    if photo_urls:
        user_content = [{"type": "text", "text": user_message}]
        for photo_url in photo_urls[:3]:
            user_content.append({"type": "image_url", "image_url": {"url": photo_url}})
        user_message_content = user_content
    else:
        user_message_content = user_message

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_content}
    ]

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "sonar",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.3,
    }

    try:
        global http_client
        client = http_client or httpx.AsyncClient(timeout=30.0)
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.error(f"Unexpected API response format: {data}")
            return "Не получил ответ от API("
        return clean_response(answer)
    except httpx.TimeoutException:
        return "Слишком долго думаю, попробуй ещё раз)"
    except httpx.HTTPStatusError as e:
        logger.error(f"Perplexity API error: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 429:
            return "Много запросов, подожди минутку)"
        return "Проблема с API, попробуй позже)"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "Что-то пошло не так("


def clean_response(text: str) -> str:
    """Clean up response: remove citations and fix emoticon spacing."""
    if not text:
        return text
    text = re.sub(r'\[\d+\]', '', text)  # Remove [1], [2] citations
    text = re.sub(r'\s+(\)+|\(+)', r'\1', text)  # Fix emoticon spacing
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_facts_from_response(question: str, answer: str, user_name: str) -> list[str]:
    """Extract memorable facts from a conversation using regex patterns."""
    facts = []
    patterns = [
        (r"люблю\s+(\w+)", "любит {}"),
        (r"нравится\s+(\w+)", "нравится {}"),
        (r"не люблю\s+(\w+)", "не любит {}"),
        (r"не ем\s+(\w+)", "не ест {}"),
        (r"работаю\s+(.+?)(?:\.|$)", "работает {}"),
        (r"живу\s+(.+?)(?:\.|$)", "живёт {}"),
    ]
    for pattern, fact_template in patterns:
        match = re.search(pattern, question.lower())
        if match:
            fact = fact_template.format(match.group(1))
            if user_name:
                fact = f"{user_name} {fact}"
            facts.append(fact)
    return facts


async def smart_extract_facts(question: str, answer: str, user_name: str, chat_context: str = None) -> list[str]:
    """Use LLM to extract important facts from conversation."""
    if not question or len(question) < 10:
        return []

    context_part = f"Контекст чата: {chat_context}" if chat_context else ""
    prompt = f"""Извлеки важные факты о пользователе из этого диалога.
Пользователь: {user_name or 'unknown'}

Вопрос: {question}
Ответ: {answer}

{context_part}

Выдай ТОЛЬКО факты о человеке (интересы, предпочтения, планы, работа, и т.д.)
Формат: один факт на строку, коротко (3-7 слов)
Если фактов нет - напиши "НЕТ"
Максимум 3 факта."""

    try:
        client = http_client or httpx.AsyncClient(timeout=10.0)
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.1,
            },
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()

        if "НЕТ" in result.upper() or len(result) < 5:
            return []

        facts = []
        for line in result.split("\n"):
            line = line.strip().lstrip("-•").strip()
            if 3 < len(line) < 100:
                if user_name and not line.startswith(user_name):
                    line = f"{user_name}: {line}"
                facts.append(line)
        return facts[:3]
    except Exception as e:
        logger.error(f"Failed to extract facts: {e}")
        return []


async def save_user_interaction(user_id: int, user_name: str, username: str) -> None:
    """Save info about user who interacted with the bot."""
    if not redis_client or not user_name:
        return
    try:
        key = f"user:{user_id}:profile"
        await redis_client.hset(key, mapping={
            "name": user_name,
            "username": username or "",
            "last_seen": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Failed to save user interaction: {e}")


# ============ MESSAGE HANDLERS ============

def should_respond(update: Update, bot_username: str) -> bool:
    """Check if bot should respond to this message."""
    message = update.message
    if not message:
        return False

    # Get text content (text or caption for photos)
    content = get_message_content(message)

    # Must have some content
    if not content and not is_forwarded_message(message) and not has_photo(message):
        return False

    # In private chats, always respond to messages with text/caption or photos
    if message.chat.type == "private" and (content or has_photo(message)):
        return True

    # Respond if replying to bot's message
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.username == bot_username:
            return True

    # Respond if @mentioned (check both text and caption)
    if content and f"@{bot_username}" in content:
        return True

    return False


def extract_question(text: str, bot_username: str) -> str:
    """Remove bot mention from the question."""
    if not text:
        return ""
    return text.replace(f"@{bot_username}", "").strip()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "Привет! Спрашивай про ивенты, бары, кино, погоду - что угодно по Таллинну.\n\n"
        "Можешь пересылать посты, ссылки или фото:\n"
        "- 'о чём это?'\n"
        "- 'какой фильм лучше?'\n"
        "- 'это правда?'\n"
        "- 'что на фото?'\n\n"
        "В группе тэгай меня или отвечай на мои сообщения."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "Спрашивай что угодно про Таллинн!\n\n"
        "Анализ постов/ссылок:\n"
        "1. Перешли пост или скинь ссылку\n"
        "2. Ответь на него и спроси что хочешь\n\n"
        "Анализ фото:\n"
        "1. Скинь фото (меню, афиша, что угодно)\n"
        "2. Спроси что хочешь или просто жди ответ\n\n"
        "Анализ сообщений из чата:\n"
        "1. Сделай reply на любое сообщение\n"
        "2. Тэгни меня и спроси\n"
        "3. Я прочитаю сообщение + контекст разговора\n\n"
        "Примеры:\n"
        "- 'это правда?'\n"
        "- 'подробнее про это'\n"
        "- 'какой вариант лучше?'\n"
        "- 'что посоветуешь из меню?'\n\n"
        "Память:\n"
        "/memory - посмотреть что помню\n"
        "/remember <факт> - запомнить\n"
        "/forget - забыть всё"
    )


async def remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /remember command to save facts."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    username = update.effective_user.username
    user_name = USERNAME_TO_NAME.get(username, username)

    if not context.args:
        await update.message.reply_text(
            "Использование: /remember <факт>\n"
            "Например: /remember люблю IPA"
        )
        return

    fact = " ".join(context.args)
    if user_name:
        fact = f"{user_name}: {fact}"

    if update.effective_chat.type == "private":
        await save_user_fact(user_id, fact)
    else:
        await save_group_fact(chat_id, fact)

    await update.message.reply_text("Запомнил)")


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /forget command to clear memory."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # In group chats, check if user is admin
    if update.effective_chat.type != "private":
        member = await context.bot.get_chat_member(chat_id, user_id)
        if member.status not in ["creator", "administrator"]:
            await update.message.reply_text("Только админ может это делать)")
            return

    if redis_client:
        try:
            if update.effective_chat.type == "private":
                await redis_client.delete(f"user:{user_id}:facts")
            else:
                await redis_client.delete(f"group:{chat_id}:facts")
            await update.message.reply_text("Забыл всё)")
        except Exception as e:
            logger.error(f"Failed to forget: {e}")
            await update.message.reply_text("Не получилось забыть(")
    else:
        await update.message.reply_text("Память не подключена(")


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /memory command to view stored facts."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    username = update.effective_user.username
    user_name = USERNAME_TO_NAME.get(username, username) if username else "Ты"

    if not redis_client:
        await update.message.reply_text("Память не подключена(")
        return

    if update.effective_chat.type == "private":
        # Show user facts in private chat
        facts = await get_user_facts(user_id)
        if facts:
            facts_text = "\n".join([f"- {fact}" for fact in facts])
            await update.message.reply_text(f"Что я помню про тебя:\n\n{facts_text}")
        else:
            await update.message.reply_text("Пока ничего не помню про тебя")
    else:
        # Show both user and group facts in group chat
        user_facts = await get_user_facts(user_id)
        group_facts = await get_group_facts(chat_id)

        response = ""
        if user_facts:
            facts_text = "\n".join([f"- {fact}" for fact in user_facts])
            response += f"Про {user_name}:\n{facts_text}\n\n"

        if group_facts:
            facts_text = "\n".join([f"- {fact}" for fact in group_facts])
            response += f"Про группу:\n{facts_text}"

        if not user_facts and not group_facts:
            response = "Пока ничего не помню"

        await update.message.reply_text(response.strip())


async def _extract_and_save_facts(
    question: str, answer: str, user_name: str,
    conv_context: str, chat_id: int, user_id: int,
) -> None:
    """Background task: extract facts from conversation and save to Redis.

    Runs as asyncio.create_task() so it doesn't block the message handler.
    """
    try:
        facts = await smart_extract_facts(
            question=question,
            answer=answer,
            user_name=user_name,
            chat_context=conv_context,
        )
        if not facts:
            facts = extract_facts_from_response(question, answer, user_name)

        for fact in facts:
            if chat_id == user_id:
                await save_user_fact(user_id, fact)
            else:
                await save_group_fact(chat_id, fact)

        if facts:
            logger.info(f"Learned facts: {facts}")
    except Exception as e:
        logger.error(f"Background fact extraction failed: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    message = update.message
    if not message:
        return

    # Periodically clean up stale in-memory data
    evict_stale_data()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user = update.effective_user

    # Get display name with fallback: mapping -> first_name -> None
    if user.username and user.username in USERNAME_TO_NAME:
        user_name = USERNAME_TO_NAME[user.username]
    elif user.first_name:
        user_name = user.first_name
    else:
        user_name = None

    # Track context for all messages in groups (even if not responding)
    msg_content = get_message_content(message)
    if msg_content and update.effective_chat.type != "private":
        context_name = user_name or "user"
        add_to_context(chat_id, "user", context_name, msg_content)

    # Check if we should respond
    if not should_respond(update, BOT_USERNAME):
        return

    # Get the user's question
    # Get question from text OR caption (for photos with text)
    question = extract_question(get_message_content(message), BOT_USERNAME)

    # Check for referenced content (reply to forwarded message, message with links, etc.)
    referenced_content = None
    reply_msg = message.reply_to_message

    # Case 1: User replies to another message
    # When bot is tagged in a reply, ALWAYS analyze the replied message
    msg_text = get_message_content(message)
    if reply_msg and msg_text and f"@{BOT_USERNAME}" in msg_text:
        reply_content = get_message_content(reply_msg)
        if reply_content:
            # Get author info if available
            reply_author = "unknown"
            if reply_msg.from_user:
                reply_user = reply_msg.from_user
                if reply_user.username and reply_user.username in USERNAME_TO_NAME:
                    reply_author = USERNAME_TO_NAME[reply_user.username]
                elif reply_user.first_name:
                    reply_author = reply_user.first_name
                elif reply_user.username:
                    reply_author = reply_user.username

            # Extract URLs from entities (hyperlinks) + plain text
            reply_urls = get_all_urls(reply_msg)

            # Check if replied message is forwarded
            if is_forwarded_message(reply_msg):
                referenced_content = f"[Forwarded post]: {reply_content}"
                if reply_urls:
                    referenced_content += f"\n[URLs in post]: {', '.join(reply_urls[:5])}"
            # Check if replied message has URLs
            elif reply_urls:
                referenced_content = f"[Message with links]: {reply_content}"
                referenced_content += f"\n[URLs]: {', '.join(reply_urls[:5])}"
            # ANY other message - include it with author
            else:
                referenced_content = f"[Message from {reply_author}]: {reply_content}"

    # Case 2: Current message is forwarded (user forwarded + asked in same message or separately)
    if is_forwarded_message(message) and not referenced_content:
        content = get_message_content(message)
        if content:
            msg_urls = get_all_urls(message)
            referenced_content = f"[Forwarded post]: {content}"
            if msg_urls:
                referenced_content += f"\n[URLs in post]: {', '.join(msg_urls[:5])}"
            # If no explicit question, default to analysis
            if not question:
                question = "расскажи об этом"

    # Case 3: Current message has URLs (no reply) - check entities too
    if not referenced_content and question:
        urls = get_all_urls(message) or extract_urls(question)
        if urls:
            referenced_content = f"[Shared link]: {urls[0]}"

    # Fetch URL content if we have URLs (helps when Perplexity can't access the site)
    urls_to_fetch = []
    if reply_msg:
        urls_to_fetch = get_all_urls(reply_msg)
    if not urls_to_fetch:
        urls_to_fetch = get_all_urls(message) or extract_urls(question or "")

    if urls_to_fetch and referenced_content:
        # Try to fetch the first URL's content
        first_url = urls_to_fetch[0]
        logger.info(f"Fetching URL content: {first_url}")
        url_content = await fetch_url_content(first_url)
        if url_content and len(url_content) > 100:
            referenced_content += f"\n\n[Article content]:\n{url_content}"

    # Check if photo without text
    has_current_photo = has_photo(message)
    has_reply_photo = reply_msg and has_photo(reply_msg)

    # If still no question, prompt user (unless there's a photo)
    if not question and not referenced_content and not has_current_photo and not has_reply_photo:
        await message.reply_text(
            "Чё спросить хотел?",
            reply_to_message_id=message.message_id,
        )
        return

    # Default question if only content provided
    if not question and referenced_content:
        question = "о чём это?"

    # Default question if only photo provided
    if not question and (has_current_photo or has_reply_photo):
        question = "что на фото?"

    # NOW check rate limit (after we know we will process)
    is_limited, remaining = check_rate_limit(user_id)
    if is_limited:
        await message.reply_text(
            f"Подожди {remaining} сек, не спеши)",
            reply_to_message_id=message.message_id,
        )
        return

    # Send typing indicator
    await send_typing(context.bot, chat_id)

    # Get context and memory
    conv_context = get_context_string(chat_id)
    user_facts = await get_user_facts(user_id)
    group_facts = await get_group_facts(chat_id) if chat_id != user_id else []

    # Check for photos to analyze
    photo_urls = []

    # Check current message for photos
    if has_photo(message):
        # Get the highest quality photo
        photo = message.photo[-1]
        photo_url = await download_photo_as_base64(photo, context.bot)
        if photo_url:
            photo_urls.append(photo_url)
            logger.info(f"Added photo from current message")

    # Check replied message for photos
    if reply_msg and has_photo(reply_msg):
        photo = reply_msg.photo[-1]
        photo_url = await download_photo_as_base64(photo, context.bot)
        if photo_url:
            photo_urls.append(photo_url)
            logger.info(f"Added photo from replied message")

    # Simplified: Use Perplexity Sonar for everything
    # Perplexity Sonar has built-in web search that works better than routing through Brave
    # This matches the behavior of the Perplexity mobile app

    logger.info(f"Query from {user_id} ({user_name}): {question[:50]}... [ref={referenced_content is not None}, photos={len(photo_urls)}]")

    answer = await query_perplexity(
        question=question,
        referenced_content=referenced_content,
        user_name=user_name,
        context=conv_context,
        user_facts=user_facts,
        group_facts=group_facts,
        photo_urls=photo_urls if photo_urls else None,
    )

    # Set rate limit AFTER successful query
    set_rate_limit(user_id)

    # Add to context
    add_to_context(chat_id, "user", user_name or "user", question)
    add_to_context(chat_id, "assistant", "bot", answer)

    # Save user interaction (learn who talks to us)
    await save_user_interaction(user_id, user_name, user.username)

    # Send response immediately, then learn in background
    await message.reply_text(answer, reply_to_message_id=message.message_id)

    # Fire-and-forget: extract and save facts without blocking the handler
    asyncio.create_task(_extract_and_save_facts(
        question=question,
        answer=answer,
        user_name=user_name,
        conv_context=conv_context,
        chat_id=chat_id,
        user_id=user_id,
    ))


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors."""
    logger.error(f"Exception while handling an update: {context.error}")


async def init_clients(application) -> None:
    """Initialize global HTTP clients and async Redis on startup."""
    global http_client, curl_session, redis_client

    # httpx for Perplexity API calls (no impersonation needed)
    http_client = httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    )

    # curl_cffi for URL fetching (browser TLS impersonation)
    curl_session = CurlAsyncSession(
        timeout=20,
        allow_redirects=True,
        max_clients=10,
    )

    logger.info("HTTP clients initialized (httpx for API, curl_cffi for URL fetching)")

    # Initialize async Redis
    if REDIS_URL:
        try:
            redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await redis_client.ping()
            logger.info("Connected to Redis (async) for memory storage")
        except Exception as e:
            logger.warning(f"Redis connection failed, memory disabled: {e}")
            redis_client = None


async def cleanup_clients(application) -> None:
    """Cleanup global HTTP clients and Redis on shutdown."""
    global http_client, curl_session, redis_client
    if http_client:
        await http_client.aclose()
    if curl_session:
        await curl_session.close()
    if redis_client:
        await redis_client.aclose()
    logger.info("All clients closed")


def main() -> None:
    """Start the bot."""
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required")
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY environment variable is required")
    if not BOT_USERNAME:
        raise ValueError("BOT_USERNAME environment variable is required")

    logger.info(f"Starting bot @{BOT_USERNAME}")
    logger.info(f"Redis URL configured: {REDIS_URL is not None}")
    logger.info("Using Perplexity Sonar with built-in web search")

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Initialize HTTP clients and Redis
    application.post_init = init_clients
    application.post_shutdown = cleanup_clients

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("remember", remember_command))
    application.add_handler(CommandHandler("forget", forget_command))
    application.add_handler(CommandHandler("memory", memory_command))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.FORWARDED | filters.PHOTO) & ~filters.COMMAND,
        handle_message
    ))

    application.add_error_handler(error_handler)

    if os.getenv("RENDER"):
        port = int(os.getenv("PORT", 10000))
        webhook_url = os.getenv("WEBHOOK_URL")
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=TELEGRAM_TOKEN,
            webhook_url=f"{webhook_url}/{TELEGRAM_TOKEN}",
        )
    else:
        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
