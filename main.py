"""Tallinn Helper Bot — entry point.

Performance-critical settings applied here:
- connection_pool_size=128 (PTB v21.9 defaults to 1, causing severe bottlenecks)
- concurrent_updates=True  (process updates from different chats in parallel)
- drop_pending_updates=True on webhook (clear stale backlog on restart)
- Increased read/write/connect timeouts
- webhook secret_token for security
"""

import os
import logging

import httpx
import redis.asyncio as aioredis
from curl_cffi.requests import AsyncSession as CurlAsyncSession
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from config import (
    TELEGRAM_TOKEN, PERPLEXITY_API_KEY, BOT_USERNAME, REDIS_URL, WEBHOOK_SECRET,
    PERPLEXITY_TIMEOUT,
    TELEGRAM_POOL_SIZE, TELEGRAM_POOL_TIMEOUT,
    TELEGRAM_READ_TIMEOUT, TELEGRAM_WRITE_TIMEOUT, TELEGRAM_CONNECT_TIMEOUT,
    logger,
)
from bot.handlers.commands import (
    start_command, help_command, remember_command, forget_command, memory_command,
)
from bot.handlers.messages import handle_message
from bot.handlers.errors import error_handler
from bot.services import memory as memory_service
from bot.services import perplexity as perplexity_service
from bot.services import url_fetcher as url_fetcher_service


async def init_clients(application) -> None:
    """Initialize global HTTP clients and async Redis on startup."""
    # httpx for Perplexity API calls (no TLS impersonation needed)
    client = httpx.AsyncClient(
        timeout=PERPLEXITY_TIMEOUT,
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    )
    memory_service.http_client = client
    perplexity_service.http_client = client

    # curl_cffi for URL fetching (browser TLS impersonation)
    url_fetcher_service.curl_session = CurlAsyncSession(
        timeout=20, allow_redirects=True, max_clients=10,
    )

    logger.info("HTTP clients initialized (httpx for API, curl_cffi for URL fetching)")

    # Async Redis
    if REDIS_URL:
        try:
            memory_service.redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await memory_service.redis_client.ping()
            logger.info("Connected to Redis (async) for memory storage")
        except Exception as e:
            logger.warning(f"Redis connection failed, memory disabled: {e}")
            memory_service.redis_client = None


async def cleanup_clients(application) -> None:
    """Cleanup global HTTP clients and Redis on shutdown."""
    if perplexity_service.http_client:
        await perplexity_service.http_client.aclose()
    if url_fetcher_service.curl_session:
        await url_fetcher_service.curl_session.close()
    if memory_service.redis_client:
        await memory_service.redis_client.aclose()
    logger.info("All clients closed")


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required")
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY environment variable is required")
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

    # ── Register handlers ────────────────────────────────────────
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("remember", remember_command))
    application.add_handler(CommandHandler("forget", forget_command))
    application.add_handler(CommandHandler("memory", memory_command))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.FORWARDED | filters.PHOTO) & ~filters.COMMAND,
        handle_message,
    ))
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
            "drop_pending_updates": True,  # clear stale backlog on restart
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
