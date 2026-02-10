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

# ── Username → display name mapping ─────────────────────────────────
USERNAME_TO_NAME = {
    "Vitalina_Bohaichuk": "Виталина",
    "hramus": "Миша",
    "I_lovet": "Полина",
    "Psychonauter": "Миша",
    "wimpex18": "Сергей",
}
