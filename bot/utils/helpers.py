"""URL extraction, rate limiting, and misc helpers."""

import re
import time
import base64
import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from config import RATE_LIMIT_SECONDS, USERNAME_TO_NAME
from bot.utils.context import user_last_query

logger = logging.getLogger(__name__)


# ── Rate limiting ────────────────────────────────────────────────────

def check_rate_limit(user_id: int) -> tuple[bool, int]:
    """Returns (is_limited, seconds_remaining)."""
    now = time.time()
    last = user_last_query[user_id]
    if last and now - last < RATE_LIMIT_SECONDS:
        return True, int(RATE_LIMIT_SECONDS - (now - last))
    return False, 0


def set_rate_limit(user_id: int) -> None:
    user_last_query[user_id] = time.time()


# ── URL helpers ──────────────────────────────────────────────────────

def extract_urls(text: str) -> list[str]:
    return re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)


def extract_urls_from_entities(message) -> list[str]:
    urls = []
    if not message:
        return urls
    entities = message.entities or message.caption_entities or []
    text = message.text or message.caption or ""
    for entity in entities:
        if entity.type == "text_link" and entity.url:
            urls.append(entity.url)
        elif entity.type == "url":
            url = text[entity.offset:entity.offset + entity.length]
            if url:
                urls.append(url)
    return urls


def get_all_urls(message) -> list[str]:
    """Get all URLs from message: plain text + hyperlink entities."""
    urls = list(extract_urls_from_entities(message))
    text = get_message_content(message)
    if text:
        for url in extract_urls(text):
            if url not in urls:
                urls.append(url)
    return urls


def clean_url(url: str) -> str:
    """Remove tracking parameters (fbclid, utm_*, etc.)."""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=False)
        tracking = {
            'fbclid', 'gclid', 'utm_source', 'utm_medium', 'utm_campaign',
            'utm_term', 'utm_content', 'ref', 'source', 'mc_cid', 'mc_eid',
            '_ga', 'yclid', 'wickedid', 'twclid', 'ttclid',
        }
        cleaned_params = {
            k: v for k, v in params.items()
            if k.lower() not in tracking and not k.lower().startswith(('utm_', 'aem_'))
        }
        return urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, urlencode(cleaned_params, doseq=True), '',
        ))
    except Exception:
        return url


def extract_url_info(url: str) -> str:
    """Minimal factual info from URL when page cannot be fetched.

    Never interprets URL path segments — that caused AI hallucinations.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')

        platform_names = {
            'tickettailor.com': 'TicketTailor (ticket sales platform)',
            'eventbrite.com': 'Eventbrite (event platform)',
            'facebook.com': 'Facebook',
            'piletilevi.ee': 'Piletilevi (Estonian ticket platform)',
            'fienta.com': 'Fienta (Baltic ticket platform)',
            'piletimaailm.com': 'Piletimaailm (Estonian ticket platform)',
        }

        platform = None
        for site, name in platform_names.items():
            if site in domain:
                platform = name
                break

        info = ["[PAGE NOT ACCESSIBLE - content could not be loaded]"]
        info.append(f"Site: {platform or domain}")
        info.append(f"URL: {url}")

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


# ── Message helpers ──────────────────────────────────────────────────

def extract_question(text: str, bot_username: str) -> str:
    """Remove bot mention from the question."""
    if not text:
        return ""
    return text.replace(f"@{bot_username}", "").strip()


def get_message_content(message) -> str:
    if message.text:
        return message.text
    if message.caption:
        return message.caption
    return ""


def is_forwarded_message(message) -> bool:
    if not message:
        return False
    return message.forward_origin is not None


def has_photo(message) -> bool:
    if not message:
        return False
    return message.photo is not None and len(message.photo) > 0


async def download_photo_as_base64(photo, bot) -> str | None:
    try:
        file = await bot.get_file(photo.file_id)
        photo_bytes = await file.download_as_bytearray()
        base64_string = base64.b64encode(photo_bytes).decode('utf-8')
        mime_type = "image/jpeg"
        if hasattr(file, 'file_path') and file.file_path and file.file_path.endswith('.png'):
            mime_type = "image/png"
        return f"data:{mime_type};base64,{base64_string}"
    except Exception as e:
        logger.error(f"Failed to download photo: {e}")
        return None


async def send_typing(bot, chat_id: int) -> None:
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass


def get_display_name(user) -> str | None:
    """Resolve Telegram user → display name."""
    if user.username and user.username in USERNAME_TO_NAME:
        return USERNAME_TO_NAME[user.username]
    if user.first_name:
        return user.first_name
    return None
