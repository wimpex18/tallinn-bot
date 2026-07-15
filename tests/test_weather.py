from bot.services.weather import extract_weather_city, is_weather_query


def test_is_weather_query_positive():
    assert is_weather_query("какая погода сегодня?") is True
    assert is_weather_query("what's the weather like") is True


def test_is_weather_query_negative():
    assert is_weather_query("какой фильм посмотреть?") is False


def test_extract_weather_city_finds_explicit_city():
    city = extract_weather_city("погода в Берлине")
    assert city == "Берлине"


def test_extract_weather_city_returns_none_without_city():
    assert extract_weather_city("какая погода") is None


def test_extract_weather_city_english_phrasing():
    city = extract_weather_city("weather in London")
    assert city == "London"
