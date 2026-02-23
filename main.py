"""Tallinn Helper Bot — entry point.

Performance-critical settings applied here:
- connection_pool_size=128 (PTB v21.9 defaults to 1, causing severe bottlenecks)
- concurrent_updates=True  (process updates from different chats in parallel)
- drop_pending_updates=True on webhook (clear stale backlog on restart)
- Increased read/write/connect timeouts
- webhook secret_token for security
- JobQueue for proactive memory + scheduled tasks
- Silent observer handler (group 1) for style profiling + spontaneous replies
"""

import os
import datetime
import zoneinfo
import logging

import anthropic
import redis.asyncio as aioredis
from curl_cffi.requests import AsyncSession as CurlAsyncSession
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import (
    TELEGRAM_TOKEN, ANTHROPIC_API_KEY, BOT_USERNAME, REDIS_URL, WEBHOOK_SECRET,
    TELEGRAM_POOL_SIZE, TELEGRAM_POOL_TIMEOUT,
    TELEGRAM_READ_TIMEOUT, TELEGRAM_WRITE_TIMEOUT, TELEGRAM_CONNECT_TIMEOUT,
    PROACTIVE_MEMORY_INTERVAL,
    QUIET_HOURS_START, QUIET_HOURS_END,
    logger,
)
from bot.handlers.commands import (
    start_command, help_command, remember_command, forget_command, memory_command,
    cleanup_command, quiet_command,
)
from bot.handlers.messages import handle_message
from bot.handlers.observer import observe_and_learn
from bot.handlers.errors import error_handler
from bot.services import memory as memory_service
from bot.services import claude as claude_service
from bot.services import url_fetcher as url_fetcher_service
from bot.services.style import generate_style_summary_llm

TALLINN_TZ = zoneinfo.ZoneInfo("Europe/Tallinn")


# ── Scheduled jobs ───────────────────────────────────────────────────

async def proactive_memory_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodically review recent group messages and extract facts.

    Runs ~3× per day, skips quiet hours.
    """
    hour = datetime.datetime.now(TALLINN_TZ).hour
    if QUIET_HOURS_START > QUIET_HOURS_END:
        if hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END:
            return
    elif QUIET_HOURS_START <= hour < QUIET_HOURS_END:
        return

    if not memory_service.redis_client:
        return

    logger.info("[job] Proactive memory extraction starting")

    try:
        # Find all chats that have recent messages
        cursor = 0
        while True:
            cursor, keys = await memory_service.redis_client.scan(
                cursor, match="chat:*:recent_msgs", count=50,
            )
            for key in keys:
                chat_id_str = key.split(":")[1]
                try:
                    chat_id = int(chat_id_str)
                except ValueError:
                    continue

                # Skip quiet-mode chats
                if await memory_service.is_quiet_mode(chat_id):
                    continue

                messages = await memory_service.get_recent_chat_messages(chat_id, 20)
                if len(messages) < 3:
                    continue

                facts = await memory_service.extract_facts_from_conversation(
                    chat_id, messages,
                )
                for fact in facts:
                    await memory_service.save_group_fact(chat_id, fact)

                if facts:
                    logger.info(f"[job] Learned {len(facts)} facts from chat {chat_id}")

            if cursor == 0:
                break
    except Exception as e:
        logger.error(f"[job] Proactive memory extraction failed: {e}")


async def refresh_style_profiles_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodically regenerate LLM-based style summaries for active users."""
    if not memory_service.redis_client:
        return

    logger.info("[job] Refreshing user style profiles")
    try:
        cursor = 0
        refreshed = 0
        while True:
            cursor, keys = await memory_service.redis_client.scan(
                cursor, match="user:*:style", count=50,
            )
            for key in keys:
                user_id_str = key.split(":")[1]
                try:
                    user_id = int(user_id_str)
                except ValueError:
                    continue

                msg_count = int(
                    await memory_service.redis_client.hget(key, "msg_count") or 0
                )
                if msg_count < 10:
                    continue

                profile = await memory_service.redis_client.hgetall(
                    f"user:{user_id}:profile"
                )
                user_name = profile.get("name", "user")
                result = await generate_style_summary_llm(
                    memory_service.redis_client,
                    user_id,
                    user_name,
                )
                if result:
                    refreshed += 1

            if cursor == 0:
                break
        if refreshed:
            logger.info(f"[job] Refreshed {refreshed} style profiles")
    except Exception as e:
        logger.error(f"[job] Style profile refresh failed: {e}")


# ── Client lifecycle ─────────────────────────────────────────────────

async def init_clients(application) -> None:
    """Initialize global HTTP clients, async Redis, and schedule jobs."""
    # Anthropic async client (SDK manages its own HTTP connection pool)
    claude_service.anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    logger.info("Anthropic client initialized")

    # curl_cffi for URL fetching (browser TLS impersonation)
    url_fetcher_service.curl_session = CurlAsyncSession(
        timeout=20, allow_redirects=True, max_clients=10,
    )

    logger.info("HTTP clients initialized (anthropic SDK + curl_cffi for URL fetching)")

    # Async Redis
    if REDIS_URL:
        try:
            memory_service.redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await memory_service.redis_client.ping()
            logger.info("Connected to Redis (async) for memory storage")
        except Exception as e:
            logger.warning(f"Redis connection failed, memory disabled: {e}")
            # Close the connection if it was created but ping failed
            if memory_service.redis_client:
                try:
                    await memory_service.redis_client.aclose()
                except Exception:
                    pass
            memory_service.redis_client = None

    # Schedule periodic jobs
    jq = application.job_queue
    if jq:
        # Proactive memory extraction (~3× per day)
        jq.run_repeating(
            proactive_memory_job,
            interval=PROACTIVE_MEMORY_INTERVAL,
            first=600,   # first run 10 min after startup
            name="proactive_memory",
        )
        # Style profile refresh (once a day at 14:00 Tallinn time)
        jq.run_daily(
            refresh_style_profiles_job,
            time=datetime.time(14, 0, tzinfo=TALLINN_TZ),
            name="refresh_styles",
        )
        logger.info("JobQueue: proactive_memory + refresh_styles scheduled")
    else:
        logger.warning("JobQueue not available — install python-telegram-bot[job-queue]")


async def cleanup_clients(application) -> None:
    """Cleanup global HTTP clients and Redis on shutdown."""
    if claude_service.anthropic_client:
        await claude_service.anthropic_client.close()
    if url_fetcher_service.curl_session:
        await url_fetcher_service.curl_session.close()
    if memory_service.redis_client:
        await memory_service.redis_client.aclose()
    logger.info("All clients closed")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required")
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    if not BOT_USERNAME:
        raise ValueError("BOT_USERNAME environment variable is required")

    logger.info(f"Starting bot @{BOT_USERNAME}")
    logger.info(f"Redis URL configured: {REDIS_URL is not None}")
    logger.info(
        f"Telegram pool: size={TELEGRAM_POOL_SIZE}, "
        f"pool_timeout={TELEGRAM_POOL_TIMEOUT}s, "
        f"read_timeout={TELEGRAM_READ_TIMEOUT}s"
    )

    # ── Build application with performance-tuned settings ────────
    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .connection_pool_size(TELEGRAM_POOL_SIZE)       # default was 1!
        .pool_timeout(TELEGRAM_POOL_TIMEOUT)
        .read_timeout(TELEGRAM_READ_TIMEOUT)
        .write_timeout(TELEGRAM_WRITE_TIMEOUT)
        .connect_timeout(TELEGRAM_CONNECT_TIMEOUT)
        .get_updates_read_timeout(TELEGRAM_READ_TIMEOUT)
        .concurrent_updates(True)                       # parallel update processing
        .build()
    )

    application.post_init = init_clients
    application.post_shutdown = cleanup_clients

    # ── Handler group 0 (default): commands + main message handler ───
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("remember", remember_command))
    application.add_handler(CommandHandler("forget", forget_command))
    application.add_handler(CommandHandler("memory", memory_command))
    application.add_handler(CommandHandler("cleanup", cleanup_command))
    application.add_handler(CommandHandler("quiet", quiet_command))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.FORWARDED | filters.PHOTO) & ~filters.COMMAND,
        handle_message,
    ))

    # ── Handler group 1: silent observer (runs on EVERY group message) ──
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
            observe_and_learn,
        ),
        group=1,
    )

    application.add_error_handler(error_handler)

    # ── Run ──────────────────────────────────────────────────────
    if os.getenv("RENDER"):
        port = int(os.getenv("PORT", 10000))
        webhook_url = os.getenv("WEBHOOK_URL")

        webhook_kwargs = {
            "listen": "0.0.0.0",
            "port": port,
            "url_path": TELEGRAM_TOKEN,
            "webhook_url": f"{webhook_url}/{TELEGRAM_TOKEN}",
            "drop_pending_updates": True,
        }
        if WEBHOOK_SECRET:
            webhook_kwargs["secret_token"] = WEBHOOK_SECRET

        application.run_webhook(**webhook_kwargs)
    else:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )


if __name__ == "__main__":
    main()
