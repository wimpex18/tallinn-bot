"""Configuration: environment variables and constants."""

import logging
import os

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("tallinn_bot")

# ── Environment variables ────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
BOT_USERNAME = os.getenv("BOT_USERNAME", "")
# Plain-text names that also trigger a response in group chats (in addition to
# @mentions and replies) — e.g. "Сэм, погода на завтра?" with no @-mention.
BOT_DISPLAY_NAMES = ["Sam", "Сэм"]
REDIS_URL = os.getenv("REDIS_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
# Random path segment for the webhook URL — intentionally independent of
# TELEGRAM_TOKEN so the bot token never appears in Render's request logs.
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "")

# ── Rate limiting ────────────────────────────────────────────────────
RATE_LIMIT_SECONDS = 5

# ── Conversation context ─────────────────────────────────────────────
CONTEXT_SIZE = 50            # increased from 20 — Claude handles long contexts well
CONTEXT_MAX_AGE = 3600       # 1 hour — evict stale contexts
RATE_LIMIT_MAX_AGE = 300     # 5 min — evict stale rate-limit entries
EVICTION_INTERVAL = 300      # run eviction every 5 min
CONTEXT_COMPACT_THRESHOLD = 15  # trim API context when it exceeds this many turns
CONTEXT_COMPACT_KEEP = 10       # keep this many recent turns after trimming

# ── URL fetching ─────────────────────────────────────────────────────
URL_CACHE_TTL = 300          # 5 min cache per URL
FETCH_TIMEOUT = 20           # seconds per fetch attempt
IMPERSONATE_PROFILES = ["chrome", "safari"]
URL_MAX_CHARS = 8000         # total character limit for fetched content
URL_HEAD_CHARS = 3000        # characters kept from the start (title, lead, date)
URL_TAIL_CHARS = 2000        # characters kept from the end (conclusions, contacts)

# ── Mistral API ───────────────────────────────────────────────────────
MISTRAL_MODEL = "mistral-small-latest"     # currently resolves to Mistral Small 4 (vision-capable)
MISTRAL_TIMEOUT = 60.0
MISTRAL_MAX_TOKENS = 1024
MISTRAL_TEMPERATURE = 0.3
VOXTRAL_MODEL = "voxtral-mini-latest"      # speech-to-text for voice messages

# Text-to-speech voice replies — experimental, opt-in, priced per minute
# (unlike everything else in this file, not verified against a live API key;
# see bot/services/speech.py). Off unless VOXTRAL_TTS_VOICE_ID is set to a
# voice id from your Mistral account (console.mistral.ai → Voices, preset or
# custom) — there's no default, since the API requires a real voice id.
VOXTRAL_TTS_MODEL = "voxtral-mini-tts-latest"
VOXTRAL_TTS_VOICE_ID = os.getenv("VOXTRAL_TTS_VOICE_ID", "")
VOICE_REPLY_TRIGGER_KEYWORDS = ["ответь голосом", "скажи голосом", "запиши голосовое", "voice reply"]

# Usage observability: log a one-time warning once today's call count hits this
# (no hard cutoff — just visibility so free-tier usage doesn't silently degrade)
QUOTA_WARN_THRESHOLD = 300

# ── Web search (Mistral Conversations API + web_search tool) ─────────
# Explicit natural-language triggers that route a question through a live
# web search before answering (in addition to /factcheck and /debate).
SEARCH_TRIGGER_KEYWORDS = [
    "найди", "найти", "поищи", "поискать", "загугли", "нагугли",
    "search for", "search the web", "погугли",
]

# ── Natural-language action triggers (bot/services/intent.py) ────────
# Tier 1 ("strong"): specific phrases that execute the matching action
# immediately, no extra Mistral call. Tier 2 ("weak"): looser stems that,
# without a tier-1 match, spend one extra Mistral call to disambiguate
# ("is this actually a request, and for what?"). See bot/services/intent.py.
INTENT_STRONG_SUMMARY = [
    "о чём говорили", "о чём тут говорили", "что обсуждали", "что тут обсуждали",
    "что я пропустил", "что произошло пока меня не было",
    "скинь саммари", "сделай саммари", "нужно саммари",
    "перескажи разговор", "перескажи что было", "перескажи о чём",
]
INTENT_WEAK_SUMMARY = ["саммари", "тлдр", "tldr", "резюме"]

# Trigger phrases for debate — everything after the phrase (minus a leading
# "про"/"о"/"на тему") is captured as the topic.
INTENT_STRONG_DEBATE = [
    "давай подебатируем", "подебатируем", "давай поспорим", "поспорим", "устроим дебаты",
]
INTENT_WEAK_DEBATE = ["дебат", "оппонент", "поспор"]

INTENT_STRONG_FACTCHECK = ["проверь факт", "фактчек", "факт-чек", "это точно правда что"]
INTENT_WEAK_FACTCHECK = ["фактчек", "факт-чек", "правда ли"]

INTENT_STRONG_POLL = [
    "сделай опрос", "создай опрос", "запусти голосование", "нужен опрос",
    "давайте проголосуем", "сделай голосование", "заведи опрос",
]
INTENT_WEAK_POLL = ["опрос", "голосование", "проголосу"]

INTENT_STRONG_QUIZ = ["устрой викторину", "сделай викторину", "давай викторину", "хочу викторину"]
INTENT_WEAK_QUIZ = ["викторин", "quiz"]

# ── Debate mode ───────────────────────────────────────────────────────
DEBATE_MODE_TTL = 1800   # 30 min — how long /debate keeps the adversarial persona active

# ── Lighter engagement: emoji reactions instead of spontaneous text ──
# Restricted to Telegram's standard (non-Premium) reaction emoji set.
REACTION_EMOJI = ["👍", "😁", "🔥", "🤔", "😂", "👏"]
REACTION_PROBABILITY = 0.05      # small chance per message, default ON (unlike text replies)
REACTION_KEYWORD_BOOST = 0.10    # extra chance when INTERESTING_TOPICS keywords are present

# Daily fun/icebreaker prompt (JobQueue), skipped for quiet-mode chats.
# Off by default — the bot should only speak when spoken to (or reacted to
# with an emoji, see REACTION_PROBABILITY above); this was too intrusive.
DAILY_PROMPT_ENABLED = False
DAILY_PROMPT_HOUR = 18  # 18:00 Tallinn time — evening, likely active chat
DAILY_PROMPTS = [
    "Какое место в Таллинне вы бы показали другу, который приехал на один день?",
    "Лучший бар/кафе, который вы открыли для себя за последний месяц?",
    "Если бы пришлось переехать из Таллинна — куда и почему?",
    "Какая эстонская привычка/традиция вас удивила больше всего?",
    "Недооценённое место в городе, куда никто не ходит зря?",
    "Какое блюдо эстонской кухни вы бы порекомендовали, а какое — нет?",
    "Что бы вы изменили в Таллинне, будь у вас такая власть?",
    "Лучшее событие/концерт, на котором вы были в этом году?",
]

# ── Telegram connection pool (critical for performance) ──────────────
# PTB v21.9 defaults to pool_size=1 which causes severe bottlenecks.
TELEGRAM_POOL_SIZE = 128
TELEGRAM_POOL_TIMEOUT = 5.0
TELEGRAM_READ_TIMEOUT = 30
TELEGRAM_WRITE_TIMEOUT = 30
TELEGRAM_CONNECT_TIMEOUT = 15

# ── Proactive behaviour ──────────────────────────────────────────────
# Spontaneous replies (bot randomly replies to interesting messages)
SPONTANEOUS_REPLY_PROBABILITY = 0.0       # disabled — bot only replies when asked
SPONTANEOUS_REPLY_KEYWORD_BOOST = 0.0     # disabled
SPONTANEOUS_REPLY_COOLDOWN = 600          # min 10 min between spontaneous replies per chat
SPONTANEOUS_REPLY_MIN_MESSAGES = 5        # need N messages since last bot reply
PROACTIVE_MAX_PER_HOUR = 3               # max spontaneous msgs per group per hour

# Proactive memory: the bot reviews recent conversation 3× per day
# and extracts facts it missed (scheduled via JobQueue)
PROACTIVE_MEMORY_INTERVAL = 8 * 3600     # every ~8 h ≈ 3× per day
RECENT_MESSAGES_BUFFER = 20              # how many recent msgs to keep per chat

# Style profiling
STYLE_MIN_MESSAGES = 5                   # require N msgs before generating a style summary
STYLE_SUMMARY_TTL = 86400                # cache style summary for 24 h
STYLE_RECENT_MESSAGES_KEPT = 20          # number of recent messages stored per user for style

# Night-time guard (Tallinn timezone): no proactive messages between these hours
QUIET_HOURS_START = 23    # 23:00
QUIET_HOURS_END = 8       # 08:00

# Interesting topics that boost spontaneous reply probability
INTERESTING_TOPICS = [
    "таллинн", "tallinn", "эстони", "estonia", "бар", "ресторан",
    "кафе", "клуб", "кино", "концерт", "мероприят", "фестивал",
    "погод", "рекоменд", "посоветуй", "сходить", "пойти",
    "event", "weekend", "выходн",
]

# Redis key TTLs (prevent orphaned data)
REDIS_KEY_TTL_DAYS = 90   # expire user/group keys untouched for 90 days

# ── Username → display name mapping ─────────────────────────────────
USERNAME_TO_NAME = {
    "Vitalina_Bohaichuk": "Виталина",
    "hramus": "Миша",
    "I_lovet": "Полина",
    "Psychonauter": "Миша",
    "wimpex18": "Сергей",
}
