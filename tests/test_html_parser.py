from bot.utils.html_parser import (
    extract_metadata,
    extract_page_text,
    format_metadata_text,
    is_cloudflare_block,
)


def test_extract_metadata_title_and_og_tags():
    html = """
    <html><head>
    <title>Fallback Title</title>
    <meta property="og:title" content="Real Title">
    <meta property="og:description" content="A description">
    <meta name="description" content="Meta description">
    </head><body>text</body></html>
    """
    meta = extract_metadata(html)
    assert meta["title"] == "Fallback Title"
    assert meta["og_title"] == "Real Title"
    assert meta["og_description"] == "A description"


def test_extract_metadata_jsonld_event():
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@type": "Event", "name": "Concert Night", "startDate": "2026-08-01",
     "location": {"name": "Club X", "address": "Main St 1"},
     "offers": {"price": "20", "priceCurrency": "EUR"}}
    </script>
    </head><body></body></html>
    """
    meta = extract_metadata(html)
    assert meta["event_name"] == "Concert Night"
    assert meta["event_date"] == "2026-08-01"
    assert meta["venue"] == "Club X"
    assert meta["price"] == "20 EUR"


def test_extract_metadata_paywall_detection():
    html = '<html><body><div class="paywall">Subscribe to read more</div></body></html>'
    meta = extract_metadata(html)
    assert meta.get("is_paywalled") is True


def test_format_metadata_text_prefers_event_over_title():
    meta = {"event_name": "Concert Night", "og_title": "Some Title", "venue": "Club X"}
    text = format_metadata_text(meta)
    assert "Concert Night" in text
    assert "Club X" in text


def test_is_cloudflare_block_detects_challenge_page():
    html = "<html><body>Just a moment... Checking your browser before accessing.</body></html>"
    assert is_cloudflare_block(html) is True


def test_is_cloudflare_block_false_for_normal_page():
    html = "<html><body><h1>Welcome to my blog</h1><p>Some article text.</p></body></html>"
    assert is_cloudflare_block(html) is False


def test_is_cloudflare_block_strong_indicator_alone_is_sufficient():
    html = "<html><body>Ray ID: 8f3a9c2b1234</body></html>"
    assert is_cloudflare_block(html) is True


def test_is_cloudflare_block_single_generic_phrase_is_not_a_false_positive():
    # A legitimate page that happens to say "please wait" (e.g. a queue notice)
    # shouldn't be misclassified as a Cloudflare challenge on its own.
    html = "<html><body><p>Your order is processing, please wait a moment.</p></body></html>"
    assert is_cloudflare_block(html) is False


def test_is_cloudflare_block_two_generic_phrases_together_still_detected():
    html = "<html><body>Just a moment, please wait while we verify your request.</body></html>"
    assert is_cloudflare_block(html) is True


def test_extract_page_text_strips_script_and_style_in_fallback():
    html = (
        "<html><body>"
        "<script>var x = 1;</script>"
        "<style>.a { color: red; }</style>"
        "<p>Real article content that is reasonably long so it passes the length check "
        "for the trafilatura fallback path in this test case.</p>"
        "</body></html>"
    )
    text = extract_page_text(html)
    assert "var x" not in text
    assert "color: red" not in text
    assert "Real article content" in text
