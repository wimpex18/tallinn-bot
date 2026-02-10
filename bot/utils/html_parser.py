"""HTML metadata extraction and content parsing."""

import re
import json
import logging

import trafilatura

logger = logging.getLogger(__name__)


# ── Metadata extraction ──────────────────────────────────────────────

def extract_metadata(html: str) -> dict:
    """Extract metadata from HTML (og:*, meta description, title, JSON-LD)."""
    metadata = {}

    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    if title_match:
        metadata['title'] = title_match.group(1).strip()

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

    desc_patterns = [
        r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
        r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']description["\']',
    ]
    for pattern in desc_patterns:
        if 'description' not in metadata:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                metadata['description'] = match.group(1).strip()

    # Parse ALL JSON-LD blocks
    jsonld_matches = re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE,
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

    # Paywall detection from HTML patterns (fallback)
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

    if data.get('isAccessibleForFree') is False:
        metadata['is_paywalled'] = True

    # Article / NewsArticle
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

    # Event
    if 'Event' in schema_type:
        if 'name' in data:
            metadata['event_name'] = data['name']
        if 'startDate' in data:
            metadata['event_date'] = data['startDate']
        if 'endDate' in data:
            metadata['event_end_date'] = data['endDate']
        if 'description' in data:
            metadata['event_description'] = data['description']

        location = data.get('location', {})
        if isinstance(location, dict):
            if 'name' in location:
                metadata['venue'] = location['name']
            address = location.get('address', {})
            if isinstance(address, dict):
                metadata['address'] = address.get('streetAddress', '')
            elif isinstance(address, str):
                metadata['address'] = address

        performer = data.get('performer', {})
        if isinstance(performer, dict):
            if 'name' in performer:
                metadata['performer'] = performer['name']
        elif isinstance(performer, list) and performer:
            names = [p.get('name', '') for p in performer if isinstance(p, dict) and 'name' in p]
            if names:
                metadata['performer'] = ', '.join(names)

        offers = data.get('offers', {})
        if isinstance(offers, dict):
            if 'price' in offers:
                metadata['price'] = f"{offers.get('price', '')} {offers.get('priceCurrency', '')}"
            if 'url' in offers:
                metadata['ticket_url'] = offers['url']


def format_metadata_text(metadata: dict) -> str:
    """Format extracted metadata into readable text."""
    parts = []

    if metadata.get('is_paywalled'):
        parts.append("[PAYWALL: article is behind a paywall, only preview available]")

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
        parts.append(f"Description: {metadata['event_description'][:500]}")
    elif 'article_description' in metadata:
        parts.append(f"Description: {metadata['article_description'][:500]}")
    elif 'og_description' in metadata:
        parts.append(f"Description: {metadata['og_description'][:500]}")
    elif 'description' in metadata:
        parts.append(f"Description: {metadata['description'][:500]}")

    return '\n'.join(parts)


# ── Page text extraction ─────────────────────────────────────────────

def extract_page_text(html: str) -> str:
    """Extract article text using trafilatura with regex fallback."""
    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        if text and len(text.strip()) > 50:
            if len(text) > 3000:
                text = text[:3000] + "..."
            return text
    except Exception as e:
        logger.warning(f"trafilatura extraction failed: {e}")

    # Fallback: basic regex stripping
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


def is_cloudflare_block(html: str) -> bool:
    """Detect Cloudflare bot protection page."""
    indicators = [
        'just a moment', 'checking your browser', 'cloudflare',
        'ray id', 'please wait', 'ddos protection', 'enable javascript',
    ]
    html_lower = html.lower()
    return any(ind in html_lower for ind in indicators)
