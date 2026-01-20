import os
import time
import logging
from collections import defaultdict
from datetime import datetime

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
BOT_USERNAME = os.getenv("BOT_USERNAME", "")  # Without @, e.g., "tallinn_helper_bot"

# Rate limiting: track last query time per user
user_last_query: dict[int, float] = defaultdict(float)
RATE_LIMIT_SECONDS = 60

# Username to name mapping for friends
USERNAME_TO_NAME = {
    "Vitalina_Bohaichuk": "Виталина",
    "hramus": "Миша",
    "I_lovet": "Полина",
    "Psychonauter": "Миша",
    "wimpex18": "Сергей",
}


def is_rate_limited(user_id: int) -> bool:
    """Check if user is rate limited (1 query per minute)."""
    now = time.time()
    last_query = user_last_query[user_id]
    if now - last_query < RATE_LIMIT_SECONDS:
        return True
    user_last_query[user_id] = now
    return False


def get_remaining_cooldown(user_id: int) -> int:
    """Get remaining seconds until user can query again."""
    now = time.time()
    last_query = user_last_query[user_id]
    return max(0, int(RATE_LIMIT_SECONDS - (now - last_query)))


async def query_perplexity(question: str, user_name: str = None) -> str:
    """Query Perplexity API with Tallinn context."""

    system_prompt = """You are a casual friend helping out your buddies in Tallinn, Estonia.
When they ask about events, bars, restaurants, cinema, weather, or activities without specifying a location, assume they mean Tallinn.

Music/event preferences: your friends are into DIY, punk, rock, metal, hip-hop, trip-hop, underground stuff, and arthouse cinema - NOT mainstream pop, disco, or commercial events.

Keep responses VERY SHORT and casual - 1-2 sentences max. Write like texting a friend: "там прикольный крафт", "сегодня прям прохладно", "есть классный концерт".
Use informal "ты" in Russian (never "вы"). Respond in the same language the user writes in (English or Russian).

IMPORTANT: NEVER use emojis (🎉😅👍 etc). Instead use text emoticons: ) or )) for happy/funny things, ( or (( for sad things. Place emoticons directly after words WITHOUT space. Examples: "классно))", "погода так себе(", "в итоге так день закончился)", "очень жаль(".
Be direct and helpful, no fluff."""

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    # Optionally include user name in the question context
    user_context = f" (from {user_name})" if user_name else ""

    payload = {
        "model": "sonar",  # Most cost-effective model with web search
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{question}{user_context}"},
        ],
        "max_tokens": 200,  # Keep responses even shorter
        "temperature": 0.3,  # More factual responses
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "Sorry, the request timed out. Please try again."
    except httpx.HTTPStatusError as e:
        logger.error(f"Perplexity API error: {e.response.status_code} - {e.response.text}")
        return "Sorry, there was an error processing your request."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "Sorry, something went wrong. Please try again later."


def should_respond(update: Update, bot_username: str) -> bool:
    """Check if bot should respond to this message."""
    message = update.message
    if not message or not message.text:
        return False

    # Respond if replying to bot's message
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.username == bot_username:
            return True

    # Respond if @mentioned
    if f"@{bot_username}" in message.text:
        return True

    # In private chats, always respond
    if message.chat.type == "private":
        return True

    return False


def extract_question(text: str, bot_username: str) -> str:
    """Remove bot mention from the question."""
    return text.replace(f"@{bot_username}", "").strip()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "Привет! Спрашивай про ивенты, бары, кино, погоду - что угодно по Таллинну.\n\n"
        "В группе тэгай меня или отвечай на мои сообщения."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "Спрашивай что угодно про Таллинн!\n\n"
        "Примеры:\n"
        "- какая сегодня погода?\n"
        "- где концерты на выходных?\n"
        "- есть крутые бары в старом городе?\n"
        "- что в кино?\n\n"
        "Пишу на русском и английском."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    if not should_respond(update, BOT_USERNAME):
        return

    user_id = update.effective_user.id
    username = update.effective_user.username
    user_name = USERNAME_TO_NAME.get(username) if username else None

    # Check rate limit
    if is_rate_limited(user_id):
        remaining = get_remaining_cooldown(user_id)
        await update.message.reply_text(
            f"Подожди {remaining} сек, не спеши)",
            reply_to_message_id=update.message.message_id,
        )
        return

    question = extract_question(update.message.text, BOT_USERNAME)

    if not question:
        await update.message.reply_text(
            "Чё спросить хотел?",
            reply_to_message_id=update.message.message_id,
        )
        return

    # Send typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    # Query Perplexity
    logger.info(f"Query from {user_id} ({username}): {question[:50]}...")
    answer = await query_perplexity(question, user_name)

    await update.message.reply_text(
        answer,
        reply_to_message_id=update.message.message_id,
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors."""
    logger.error(f"Exception while handling an update: {context.error}")


def main() -> None:
    """Start the bot."""
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required")
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY environment variable is required")
    if not BOT_USERNAME:
        raise ValueError("BOT_USERNAME environment variable is required")

    logger.info(f"Starting bot @{BOT_USERNAME}")

    # Build application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Error handler
    application.add_error_handler(error_handler)

    # Start polling (for local development)
    # For production on Render, we use webhooks
    if os.getenv("RENDER"):
        # Webhook mode for Render
        port = int(os.getenv("PORT", 10000))
        webhook_url = os.getenv("WEBHOOK_URL")  # e.g., https://your-app.onrender.com

        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=TELEGRAM_TOKEN,
            webhook_url=f"{webhook_url}/{TELEGRAM_TOKEN}",
        )
    else:
        # Polling mode for local development
        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
