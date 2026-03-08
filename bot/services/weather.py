"""Real-time weather via wttr.in (no API key required)."""

import re
import json
import logging

import httpx

from config import FETCH_TIMEOUT

logger = logging.getLogger(__name__)

_WTTR_URL = "https://wttr.in/{city}?format=j1"

# Map English wttr.in condition strings to Russian
_CONDITIONS: dict[str, str] = {
    "Sunny": "солнечно",
    "Clear": "ясно",
    "Partly cloudy": "переменная облачность",
    "Cloudy": "облачно",
    "Overcast": "пасмурно",
    "Mist": "туман",
    "Fog": "туман",
    "Haze": "дымка",
    "Light rain": "лёгкий дождь",
    "Moderate rain": "умеренный дождь",
    "Heavy rain": "сильный дождь",
    "Patchy rain possible": "местами дождь",
    "Light drizzle": "морось",
    "Drizzle": "морось",
    "Freezing drizzle": "ледяная морось",
    "Light rain shower": "ливень",
    "Moderate or heavy rain shower": "сильный ливень",
    "Light snow": "лёгкий снег",
    "Moderate snow": "снег",
    "Heavy snow": "сильный снег",
    "Patchy snow possible": "местами снег",
    "Blowing snow": "поземок",
    "Blizzard": "метель",
    "Sleet": "мокрый снег",
    "Light sleet": "мокрый снег",
    "Thunderstorm": "гроза",
    "Thundery outbreaks possible": "возможна гроза",
    "Ice pellets": "ледяная крупа",
    "Freezing fog": "морозный туман",
}

# Regex to extract a city from "погода в Таллинне" / "weather in Berlin" etc.
_CITY_RE = re.compile(
    r'(?:погода?|temperature|weather|прогноз|forecast|дождь|снег|мороз|тепло|жарко|холодно)'
    r'(?:\s+(?:в|во|in|для|for))?\s+'
    r'([А-ЯA-Z][а-яёa-z]{2,})',
    re.IGNORECASE,
)

# Keywords that trigger a weather fetch
WEATHER_KEYWORDS = {
    "погода", "погоду", "погоде", "погодой", "погодку",
    "температура", "температуру", "температуре",
    "weather", "forecast", "прогноз погоды",
    "дождь", "дождя", "дождём", "дождем",
    "снег", "снега", "снегом",
    "гроза", "гозы",
    "мороз", "мороза",
    "холодно", "тепло", "жарко",
    "ветер", "ветра",
}


def is_weather_query(text: str) -> bool:
    """Return True if the text looks like a weather question."""
    words = set(re.findall(r'\w+', text.lower()))
    return bool(words & WEATHER_KEYWORDS)


def extract_weather_city(text: str) -> str | None:
    """Try to extract an explicit city name from a weather query.

    Returns None if no city is found (caller should default to Tallinn).
    """
    m = _CITY_RE.search(text)
    return m.group(1) if m else None


async def fetch_weather(city: str = "Tallinn") -> str | None:
    """Fetch current weather + 2-day forecast from wttr.in.

    Returns a compact formatted string ready to inject as referenced_content,
    or None if the fetch fails (network error, unknown city, etc.).
    """
    url = _WTTR_URL.format(city=city.replace(" ", "+"))
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()

        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        feels = current["FeelsLikeC"]
        humidity = current["humidity"]
        wind_kmph = current["windspeedKmph"]
        wind_dir = current.get("winddir16Point", "")
        desc_en = current["weatherDesc"][0]["value"]
        desc_ru = _CONDITIONS.get(desc_en, desc_en.lower())

        lines = [
            f"[Погода в {city} — актуальные данные от wttr.in]",
            f"Сейчас: {temp_c}°C, {desc_ru}",
            f"Ощущается: {feels}°C | влажность {humidity}% | ветер {wind_kmph} км/ч {wind_dir}",
        ]

        today = data["weather"][0]
        lines.append(
            f"Сегодня: {today['mintempC']}…{today['maxtempC']}°C"
        )

        if len(data["weather"]) > 1:
            tmr = data["weather"][1]
            tmr_desc_en = tmr["hourly"][4]["weatherDesc"][0]["value"]
            tmr_desc_ru = _CONDITIONS.get(tmr_desc_en, tmr_desc_en.lower())
            lines.append(
                f"Завтра: {tmr['mintempC']}…{tmr['maxtempC']}°C, {tmr_desc_ru}"
            )

        if len(data["weather"]) > 2:
            d2 = data["weather"][2]
            d2_desc_en = d2["hourly"][4]["weatherDesc"][0]["value"]
            d2_desc_ru = _CONDITIONS.get(d2_desc_en, d2_desc_en.lower())
            lines.append(
                f"Послезавтра: {d2['mintempC']}…{d2['maxtempC']}°C, {d2_desc_ru}"
            )

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Weather fetch failed for '{city}': {e}")
        return None
