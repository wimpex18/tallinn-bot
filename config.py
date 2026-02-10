"""Configuration: environment variables and constants."""

import os
import logging

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("tallinn_bot")

# ── Environment variables ────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
BOT_USERNAME = os.getenv("BOT_USERNAME", "")
REDIS_URL = os.getenv("REDIS_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

# ── Rate limiting ────────────────────────────────────────────────────
RATE_LIMIT_SECONDS = 5

# ── Conversation context ─────────────────────────────────────────────
CONTEXT_SIZE = 10
CONTEXT_MAX_AGE = 3600       # 1 hour — evict stale contexts
RATE_LIMIT_MAX_AGE = 300     # 5 min — evict stale rate-limit entries
EVICTION_INTERVAL = 300      # run eviction every 5 min

# ── URL fetching ─────────────────────────────────────────────────────
URL_CACHE_TTL = 300          # 5 min cache per URL
FETCH_TIMEOUT = 20           # seconds per fetch attempt
IMPERSONATE_PROFILES = ["chrome", "safari"]

# ── Perplexity API ───────────────────────────────────────────────────
PERPLEXITY_TIMEOUT = 30.0
PERPLEXITY_MAX_TOKENS = 300
PERPLEXITY_TEMPERATURE = 0.3

# ── Telegram connection pool (critical for performance) ──────────────
# PTB v21.9 defaults to pool_size=1 which causes severe bottlenecks.
TELEGRAM_POOL_SIZE = 128
TELEGRAM_POOL_TIMEOUT = 5.0
TELEGRAM_READ_TIMEOUT = 30
TELEGRAM_WRITE_TIMEOUT = 30
TELEGRAM_CONNECT_TIMEOUT = 15

# ── Proactive behaviour ──────────────────────────────────────────────
# Spontaneous replies (bot randomly replies to interesting messages)
SPONTANEOUS_REPLY_PROBABILITY = 0.03      # 3 % base chance per group message
SPONTANEOUS_REPLY_KEYWORD_BOOST = 0.12    # +12 % if message touches Tallinn topics
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
