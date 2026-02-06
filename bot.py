import os
import re
import time
import json
import logging
import base64
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import httpx
import redis
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

# Redis connection for persistent memory
redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis for memory storage")
    except Exception as e:
        logger.warning(f"Redis connection failed, memory disabled: {e}")
        redis_client = None

# Global httpx clients for connection pooling
http_client: httpx.AsyncClient = None
http2_client: httpx.AsyncClient = None

# URL fetch cache: {cleaned_url: (content, timestamp)}
URL_CACHE_TTL = 300  # 5 minutes
_url_cache: dict[str, tuple[str, float]] = {}


# ============ MEMORY FUNCTIONS ============

def save_user_fact(user_id: int, fact: str) -> None:
    """Save a fact about a user to persistent memory."""
    if not redis_client:
        return
    try:
        key = f"user:{user_id}:facts"
        redis_client.sadd(key, fact)
        if redis_client.scard(key) > 20:
            facts = list(redis_client.smembers(key))
            redis_client.delete(key)
            for f in facts[-20:]:
                redis_client.sadd(key, f)
    except Exception as e:
        logger.error(f"Failed to save user fact: {e}")


def get_user_facts(user_id: int) -> list[str]:
    """Get all facts about a user from memory."""
    if not redis_client:
        return []
    try:
        return list(redis_client.smembers(f"user:{user_id}:facts"))
    except Exception as e:
        logger.error(f"Failed to get user facts: {e}")
        return []


def save_group_fact(chat_id: int, fact: str) -> None:
    """Save a fact about the group to persistent memory."""
    if not redis_client:
        return
    try:
        key = f"group:{chat_id}:facts"
        redis_client.sadd(key, fact)
        if redis_client.scard(key) > 30:
            facts = list(redis_client.smembers(key))
            redis_client.delete(key)
            for f in facts[-30:]:
                redis_client.sadd(key, f)
    except Exception as e:
        logger.error(f"Failed to save group fact: {e}")


def get_group_facts(chat_id: int) -> list[str]:
    """Get all facts about the group from memory."""
    if not redis_client:
        return []
    try:
        return list(redis_client.smembers(f"group:{chat_id}:facts"))
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

    # Extract JSON-LD structured data (events, products, etc.)
    jsonld_match = re.search(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE
    )
    if jsonld_match:
        try:
            jsonld_text = jsonld_match.group(1).strip()
            jsonld_data = json.loads(jsonld_text)

            # Handle array of objects
            if isinstance(jsonld_data, list):
                for item in jsonld_data:
                    if isinstance(item, dict):
                        extract_jsonld_event(item, metadata)
            elif isinstance(jsonld_data, dict):
                extract_jsonld_event(jsonld_data, metadata)
        except json.JSONDecodeError:
            pass

    return metadata


def extract_jsonld_event(data: dict, metadata: dict) -> None:
    """Extract event data from JSON-LD structured data."""
    schema_type = data.get('@type', '')

    # Handle Event type
    if schema_type == 'Event' or 'Event' in str(schema_type):
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

    # Event-specific formatting
    if 'event_name' in metadata:
        parts.append(f"Event: {metadata['event_name']}")
    elif 'og_title' in metadata:
        parts.append(f"Title: {metadata['og_title']}")
    elif 'title' in metadata:
        parts.append(f"Title: {metadata['title']}")

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
    elif 'og_description' in metadata:
        desc = metadata['og_description'][:500]
        parts.append(f"Description: {desc}")
    elif 'description' in metadata:
        desc = metadata['description'][:500]
        parts.append(f"Description: {desc}")

    return '\n'.join(parts)


# Browser-like headers for different scenarios
BROWSER_HEADERS = {
    'chrome': {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    },
    'firefox': {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    },
    'safari': {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
}


def _build_fetch_attempts() -> list[tuple[str, httpx.AsyncClient]]:
    """Build ordered list of (profile_name, client) attempts for URL fetching.

    Uses global pooled clients instead of creating new ones per request.
    Reduced cascade: chrome (HTTP/1.1), chrome (HTTP/2), firefox (HTTP/1.1).
    """
    attempts = []
    if http_client:
        attempts.append(('chrome', http_client))
    if http2_client:
        attempts.append(('chrome', http2_client))
    if http_client:
        attempts.append(('firefox', http_client))
    # Fallback if global clients not yet initialized
    if not attempts:
        fallback = httpx.AsyncClient(timeout=20.0, follow_redirects=True)
        attempts.append(('chrome', fallback))
    return attempts


async def fetch_url_content(url: str) -> str:
    """Fetch webpage content and extract text with anti-blocking.

    Uses global connection-pooled HTTP clients. Tries up to 3 attempts
    (chrome HTTP/1.1, chrome HTTP/2, firefox HTTP/1.1) before falling back
    to URL-based info extraction.
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

    logger.info(f"Fetching URL (cleaned): {clean_url_str}")

    parsed = urlparse(clean_url_str)
    last_error = None

    result = None

    for profile, client in _build_fetch_attempts():
        try:
            headers = BROWSER_HEADERS[profile].copy()
            headers['Referer'] = f"{parsed.scheme}://{parsed.netloc}/"

            response = await client.get(clean_url_str, headers=headers)

            # Check for soft blocks (403, 429, etc.)
            if response.status_code in (403, 429, 503):
                html = response.text
                if is_cloudflare_block(html):
                    logger.warning(f"Cloudflare/bot protection detected with {profile}")
                    last_error = "Bot protection (Cloudflare)"
                    break  # Cloudflare blocks all profiles
                logger.warning(f"Got {response.status_code} with {profile}, trying next")
                last_error = f"HTTP {response.status_code}"
                continue

            response.raise_for_status()
            html = response.text

            if is_cloudflare_block(html):
                logger.warning(f"Got challenge page with {profile}")
                last_error = "Bot protection (challenge page)"
                break

            # Try structured metadata first (more reliable)
            metadata = extract_metadata(html)
            metadata_text = format_metadata_text(metadata)

            if metadata_text and len(metadata_text) > 50:
                logger.info(f"Extracted metadata from {clean_url_str}: {len(metadata_text)} chars")
                page_text = extract_page_text(html)
                if page_text:
                    result = f"{metadata_text}\n\n[Page content]:\n{page_text}"
                else:
                    result = metadata_text
                break

            # Fallback to page text extraction
            text = extract_page_text(html)
            if text:
                logger.info(f"Fetched {len(text)} chars from {clean_url_str}")
                result = text
                break

            logger.warning(f"No content extracted from {clean_url_str}")
            result = ""
            break

        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}"
            logger.warning(f"HTTP error with {profile}: {e.response.status_code}")
            continue
        except httpx.TimeoutException:
            last_error = "Timeout"
            logger.warning(f"Timeout with {profile}")
            continue
        except Exception as e:
            if 'h2' in str(e).lower() or 'http2' in str(e).lower():
                continue
            last_error = str(e)
            logger.warning(f"Error with {profile}: {e}")
            continue

    # All attempts failed — extract minimal info from URL
    if result is None:
        logger.error(f"Failed to fetch URL {clean_url_str} after all attempts: {last_error}")
        url_info = extract_url_info(clean_url_str)
        if url_info:
            logger.info(f"Extracted URL info as fallback: {url_info}")
            result = url_info
        else:
            result = ""

    # Cache the result (including failures, to avoid re-fetching blocked pages)
    _url_cache[clean_url_str] = (result, time.time())

    # Prune expired cache entries periodically (keep cache bounded)
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
    """Extract readable text from HTML page."""
    # Remove script and style elements
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Decode HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)

    # Limit to first 3000 chars
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
        'Если не нашёл - честно скажи что страница недоступна и ты не смог найти информацию.'
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
        async with httpx.AsyncClient(timeout=10.0) as client:
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


def save_user_interaction(user_id: int, user_name: str, username: str) -> None:
    """Save info about user who interacted with the bot."""
    if not redis_client or not user_name:
        return
    try:
        key = f"user:{user_id}:profile"
        redis_client.hset(key, mapping={
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
        save_user_fact(user_id, fact)
    else:
        save_group_fact(chat_id, fact)

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
                redis_client.delete(f"user:{user_id}:facts")
            else:
                redis_client.delete(f"group:{chat_id}:facts")
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
        facts = get_user_facts(user_id)
        if facts:
            facts_text = "\n".join([f"- {fact}" for fact in facts])
            await update.message.reply_text(f"Что я помню про тебя:\n\n{facts_text}")
        else:
            await update.message.reply_text("Пока ничего не помню про тебя")
    else:
        # Show both user and group facts in group chat
        user_facts = get_user_facts(user_id)
        group_facts = get_group_facts(chat_id)

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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    message = update.message
    if not message:
        return

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
    user_facts = get_user_facts(user_id)
    group_facts = get_group_facts(chat_id) if chat_id != user_id else []

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
    save_user_interaction(user_id, user_name, user.username)

    # Send response immediately, then learn in background
    await message.reply_text(answer, reply_to_message_id=message.message_id)

    # Smart fact extraction (runs after response sent)
    try:
        # Use LLM to extract facts from conversation
        facts = await smart_extract_facts(
            question=question,
            answer=answer,
            user_name=user_name,
            chat_context=conv_context
        )

        # Fallback to regex if LLM fails
        if not facts:
            facts = extract_facts_from_response(question, answer, user_name)

        for fact in facts:
            if chat_id == user_id:
                save_user_fact(user_id, fact)
            else:
                save_group_fact(chat_id, fact)

        if facts:
            logger.info(f"Learned facts: {facts}")
    except Exception as e:
        logger.error(f"Learning failed: {e}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors."""
    logger.error(f"Exception while handling an update: {context.error}")


async def init_http_client(application) -> None:
    """Initialize global HTTP clients on startup."""
    global http_client, http2_client
    client_limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    http_client = httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        limits=client_limits,
    )
    http2_client = httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        http2=True,
        limits=client_limits,
    )
    logger.info("HTTP clients initialized with connection pooling (HTTP/1.1 + HTTP/2)")


async def cleanup_http_client(application) -> None:
    """Cleanup global HTTP clients on shutdown."""
    global http_client, http2_client
    if http_client:
        await http_client.aclose()
    if http2_client:
        await http2_client.aclose()
    logger.info("HTTP clients closed")


def main() -> None:
    """Start the bot."""
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required")
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY environment variable is required")
    if not BOT_USERNAME:
        raise ValueError("BOT_USERNAME environment variable is required")

    logger.info(f"Starting bot @{BOT_USERNAME}")
    logger.info(f"Redis connected: {redis_client is not None}")
    logger.info("Using Perplexity Sonar with built-in web search")

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Initialize HTTP client with connection pooling
    application.post_init = init_http_client
    application.post_shutdown = cleanup_http_client

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
