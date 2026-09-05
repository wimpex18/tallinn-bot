import datetime
import time
from types import SimpleNamespace

from bot.utils.context import user_last_query
from bot.utils.helpers import (
    RU_WEEKDAYS,
    TALLINN_TZ,
    check_rate_limit,
    clean_url,
    current_tallinn_date_context,
    extract_question,
    extract_url_info,
    extract_urls,
    get_display_name,
    get_forward_origin_info,
    get_message_content,
    is_forwarded_message,
    mask_quoted_spans,
    mentions_bot_by_name,
    set_rate_limit,
)


def test_current_tallinn_date_context_includes_date_weekday_and_time():
    now = datetime.datetime.now(TALLINN_TZ)
    context = current_tallinn_date_context()
    assert now.strftime("%d.%m.%Y") in context
    assert RU_WEEKDAYS[now.weekday()] in context
    assert now.strftime("%H:%M") in context
    assert "Таллинн" in context


def test_extract_urls_finds_plain_links():
    text = "check this out https://example.com/foo and also http://bar.ee"
    urls = extract_urls(text)
    assert urls == ["https://example.com/foo", "http://bar.ee"]


def test_extract_urls_no_links():
    assert extract_urls("just some text") == []


def test_clean_url_strips_tracking_params():
    url = "https://example.com/page?utm_source=fb&fbclid=abc123&id=42"
    cleaned = clean_url(url)
    assert "utm_source" not in cleaned
    assert "fbclid" not in cleaned
    assert "id=42" in cleaned


def test_clean_url_leaves_normal_params_alone():
    url = "https://example.com/page?id=42&sort=asc"
    cleaned = clean_url(url)
    assert "id=42" in cleaned
    assert "sort=asc" in cleaned


def test_extract_url_info_known_platform():
    info = extract_url_info("https://www.piletilevi.ee/some-event")
    assert "Piletilevi" in info
    assert "PAGE NOT ACCESSIBLE" in info


def test_extract_url_info_unknown_domain_uses_domain_name():
    info = extract_url_info("https://randomsite.example/foo")
    assert "randomsite.example" in info


def test_rate_limit_blocks_then_clears(monkeypatch):
    user_id = 999999
    user_last_query.pop(user_id, None)
    is_limited, remaining = check_rate_limit(user_id)
    assert is_limited is False

    set_rate_limit(user_id)
    is_limited, remaining = check_rate_limit(user_id)
    assert is_limited is True
    assert remaining > 0

    # Simulate enough time passing
    user_last_query[user_id] = time.time() - 999
    is_limited, _ = check_rate_limit(user_id)
    assert is_limited is False


def test_extract_question_strips_bot_mention():
    assert extract_question("@my_bot what's up", "my_bot") == "what's up"


def test_extract_question_empty_text():
    assert extract_question("", "my_bot") == ""
    assert extract_question(None, "my_bot") == ""


def test_get_message_content_prefers_text_then_caption():
    msg = SimpleNamespace(text="hello", caption=None)
    assert get_message_content(msg) == "hello"

    msg2 = SimpleNamespace(text=None, caption="a caption")
    assert get_message_content(msg2) == "a caption"

    msg3 = SimpleNamespace(text=None, caption=None)
    assert get_message_content(msg3) == ""


def test_is_forwarded_message():
    assert is_forwarded_message(SimpleNamespace(forward_origin=object())) is True
    assert is_forwarded_message(SimpleNamespace(forward_origin=None)) is False
    assert is_forwarded_message(None) is False


def test_forward_origin_info_channel():
    chat = SimpleNamespace(title="Meduza", username="meduzalive")
    origin = SimpleNamespace(type="channel", chat=chat)
    msg = SimpleNamespace(forward_origin=origin)
    info = get_forward_origin_info(msg)
    assert "Meduza" in info
    assert "@meduzalive" in info


def test_forward_origin_info_user():
    user = SimpleNamespace(username=None, first_name="Иван")
    origin = SimpleNamespace(type="user", sender_user=user)
    msg = SimpleNamespace(forward_origin=origin)
    info = get_forward_origin_info(msg)
    assert "Иван" in info


def test_forward_origin_info_hidden_user():
    origin = SimpleNamespace(type="hidden_user", sender_user_name="Anonymous")
    msg = SimpleNamespace(forward_origin=origin)
    info = get_forward_origin_info(msg)
    assert "Anonymous" in info


def test_forward_origin_info_none_when_not_forwarded():
    msg = SimpleNamespace(forward_origin=None)
    assert get_forward_origin_info(msg) is None


def test_get_display_name_uses_username_mapping():
    user = SimpleNamespace(username="wimpex18", first_name="Someone")
    assert get_display_name(user) == "Сергей"


def test_get_display_name_falls_back_to_first_name():
    user = SimpleNamespace(username="totally_unknown_user", first_name="Alex")
    assert get_display_name(user) == "Alex"


def test_mentions_bot_by_name_matches_latin():
    assert mentions_bot_by_name("hey Sam, what's up") is True
    assert mentions_bot_by_name("SAM!") is True


def test_mentions_bot_by_name_matches_cyrillic():
    assert mentions_bot_by_name("Сэм, погода на завтра?") is True
    assert mentions_bot_by_name("сэм привет") is True


def test_mentions_bot_by_name_no_match_on_substring():
    assert mentions_bot_by_name("Сэмплы отличные") is False
    assert mentions_bot_by_name("I bought a Samsung phone") is False


def test_mentions_bot_by_name_no_match_unrelated_text():
    assert mentions_bot_by_name("какая погода сегодня?") is False


def test_mentions_bot_by_name_empty_text():
    assert mentions_bot_by_name("") is False
    assert mentions_bot_by_name(None) is False


def test_mask_quoted_spans_blanks_guillemets():
    masked = mask_quoted_spans("можно попросить «сделай саммари» или «сделай опрос»")
    assert "саммари" not in masked
    assert "сделай опрос" not in masked
    assert "можно попросить" in masked


def test_mask_quoted_spans_blanks_straight_double_quotes():
    masked = mask_quoted_spans('команда "найди бар" сработает')
    assert "найди" not in masked


def test_mask_quoted_spans_preserves_length_for_index_math():
    text = 'что-то «сделай саммари» ещё текст'
    masked = mask_quoted_spans(text)
    assert len(masked) == len(text)


def test_mask_quoted_spans_leaves_unquoted_text_untouched():
    text = "давай поспорим про удалёнку"
    assert mask_quoted_spans(text) == text
