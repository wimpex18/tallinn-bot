"""Real-time weather via wttr.in (no API key required)."""

import re
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

# Normalize common Russian inflected city names to the form wttr.in prefers
_CITY_NORMALIZE: dict[str, str] = {
    "таллинне": "Tallinn",
    "таллинна": "Tallinn",
    "таллинну": "Tallinn",
    "таллинном": "Tallinn",
    "таллинн": "Tallinn",
    "таллин": "Tallinn",
    "москве": "Moscow",
    "москвы": "Moscow",
    "москву": "Moscow",
    "москва": "Moscow",
    "питере": "Saint Petersburg",
    "петербурге": "Saint Petersburg",
    "петербурга": "Saint Petersburg",
    "санкт-петербурге": "Saint Petersburg",
    "риге": "Riga",
    "риги": "Riga",
    "хельсинки": "Helsinki",
    "стокгольме": "Stockholm",
    "берлине": "Berlin",
    "берлина": "Berlin",
    "лондоне": "London",
    "лондона": "London",
}


def _normalize_city(city: str) -> str:
    return _CITY_NORMALIZE.get(city.lower(), city)


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


# Conditions that are always worth mentioning regardless of brevity
_NOTABLE_CONDITIONS = {
    "Blizzard", "Heavy snow", "Blowing snow",
    "Heavy rain", "Moderate or heavy rain shower",
    "Thunderstorm", "Thundery outbreaks possible",
    "Freezing drizzle", "Freezing fog", "Ice pellets",
}

_STRONG_WIND_KMH = 30  # above this, always mention wind


async def fetch_weather(city: str = "Tallinn") -> str | None:
    """Fetch current weather from wttr.in.

    Returns a compact single-line fact string for Claude to summarise,
    e.g.: "[WEATHER: Tallinn] +3°C, солнечно; сегодня +1..+4°C"
    Wind is included only when > 30 km/h. Tomorrow only when meaningfully
    different from today. Claude is instructed to turn this into ≤1 sentence.
    """
    city = _normalize_city(city)
    url = _WTTR_URL.format(city=city.replace(" ", "+"))
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()

        current = data["current_condition"][0]
        temp_c = int(current["temp_C"])
        wind_kmph = int(current["windspeedKmph"])
        desc_en = current["weatherDesc"][0]["value"]
        desc_ru = _CONDITIONS.get(desc_en, desc_en.lower())

        temp_str = f"+{temp_c}°C" if temp_c >= 0 else f"{temp_c}°C"
        parts = [f"{temp_str}, {desc_ru}"]

        # Always mention strong wind or notable conditions
        notable = desc_en in _NOTABLE_CONDITIONS
        if wind_kmph > _STRONG_WIND_KMH:
            parts.append(f"ветер {wind_kmph} км/ч")

        # Today's range
        today = data["weather"][0]
        lo = int(today["mintempC"])
        hi = int(today["maxtempC"])
        lo_s = f"+{lo}" if lo >= 0 else str(lo)
        hi_s = f"+{hi}" if hi >= 0 else str(hi)
        parts.append(f"сегодня {lo_s}..{hi_s}°C")

        # Tomorrow — only if it differs noticeably from today
        if len(data["weather"]) > 1:
            tmr = data["weather"][1]
            tmr_desc_en = tmr["hourly"][4]["weatherDesc"][0]["value"]
            tmr_desc_ru = _CONDITIONS.get(tmr_desc_en, tmr_desc_en.lower())
            tmr_lo = int(tmr["mintempC"])
            tmr_hi = int(tmr["maxtempC"])
            tmr_lo_s = f"+{tmr_lo}" if tmr_lo >= 0 else str(tmr_lo)
            tmr_hi_s = f"+{tmr_hi}" if tmr_hi >= 0 else str(tmr_hi)
            parts.append(f"завтра {tmr_lo_s}..{tmr_hi_s}°C {tmr_desc_ru}")

        return f"[WEATHER: {city}] " + "; ".join(parts)

    except Exception as e:
        logger.warning(f"Weather fetch failed for '{city}': {e}")
        return None
