from bot.services.style import analyze_message_style


def test_analyze_message_style_detects_emoji():
    signals = analyze_message_style("hello there 😂")
    assert signals["uses_emoji"] is True


def test_analyze_message_style_detects_caps():
    signals = analyze_message_style("WHY IS THIS HAPPENING")
    assert signals["uses_caps"] is True


def test_analyze_message_style_ignores_short_caps():
    signals = analyze_message_style("OK")
    assert signals["uses_caps"] is False


def test_analyze_message_style_detects_profanity():
    signals = analyze_message_style("это просто пиздец какой-то")
    assert signals["uses_profanity"] is True


def test_analyze_message_style_detects_slang():
    signals = analyze_message_style("го тусить, кста я норм")
    assert signals["uses_slang"] is True


def test_analyze_message_style_neutral_message():
    signals = analyze_message_style("Сегодня хорошая погода в городе")
    assert signals["uses_emoji"] is False
    assert signals["uses_caps"] is False
    assert signals["uses_profanity"] is False


def test_analyze_message_style_message_length():
    signals = analyze_message_style("12345")
    assert signals["msg_length"] == 5


def test_analyze_message_style_parenthesis_smileys():
    signals = analyze_message_style("ну норм)))")
    assert signals["uses_parenthesis_smileys"] is True
