import os
import re
import time
import json
import logging
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse

import httpx
import redis
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
BOT_USERNAME = os.getenv("BOT_USERNAME", "")
REDIS_URL = os.getenv("REDIS_URL")

# Rate limiting - only track AFTER successful query
user_last_query: dict[int, float] = defaultdict(float)
RATE_LIMIT_SECONDS = 60

# Conversation context: store last N messages per chat
CONTEXT_SIZE = 10
chat_context: dict[int, list[dict]] = defaultdict(list)

# Username to name mapping
USERNAME_TO_NAME = {
    "Vitalina_Bohaichuk": "Виталина",
    "hramus": "Миша",
    "I_lovet": "Полина",
    "Psychonauter": "Миша",
    "wimpex18": "Сергей",
}

# Redis connection for persistent memory
redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis for memory storage")
    except Exception as e:
        logger.warning(f"Redis connection failed, memory disabled: {e}")
        redis_client = None


# ============ MEMORY FUNCTIONS ============

def save_user_fact(user_id: int, fact: str) -> None:
    """Save a fact about a user to persistent memory."""
    if not redis_client:
        return
    try:
        key = f"user:{user_id}:facts"
        redis_client.sadd(key, fact)
        if redis_client.scard(key) > 20:
            facts = list(redis_client.smembers(key))
            redis_client.delete(key)
            for f in facts[-20:]:
                redis_client.sadd(key, f)
    except Exception as e:
        logger.error(f"Failed to save user fact: {e}")


def get_user_facts(user_id: int) -> list[str]:
    """Get all facts about a user from memory."""
    if not redis_client:
        return []
    try:
        return list(redis_client.smembers(f"user:{user_id}:facts"))
    except Exception as e:
        logger.error(f"Failed to get user facts: {e}")
        return []


def save_group_fact(chat_id: int, fact: str) -> None:
    """Save a fact about the group to persistent memory."""
    if not redis_client:
        return
    try:
        key = f"group:{chat_id}:facts"
        redis_client.sadd(key, fact)
        if redis_client.scard(key) > 30:
            facts = list(redis_client.smembers(key))
            redis_client.delete(key)
            for f in facts[-30:]:
                redis_client.sadd(key, f)
    except Exception as e:
        logger.error(f"Failed to save group fact: {e}")


def get_group_facts(chat_id: int) -> list[str]:
    """Get all facts about the group from memory."""
    if not redis_client:
        return []
    try:
        return list(redis_client.smembers(f"group:{chat_id}:facts"))
    except Exception as e:
        logger.error(f"Failed to get group facts: {e}")
        return []


# ============ CONTEXT FUNCTIONS ============

def add_to_context(chat_id: int, role: str, name: str, content: str) -> None:
    """Add a message to the chat context."""
    chat_context[chat_id].append({
        "role": role,
        "name": name,
        "content": content[:500],
        "time": time.time()
    })
    if len(chat_context[chat_id]) > CONTEXT_SIZE:
        chat_context[chat_id] = chat_context[chat_id][-CONTEXT_SIZE:]


def get_context_string(chat_id: int) -> str:
    """Get recent conversation context as a string."""
    if not chat_context[chat_id]:
        return ""
    context_lines = []
    for msg in chat_context[chat_id][-5:]:
        name = msg.get("name", "user")
        content = msg["content"]
        context_lines.append(f"{name}: {content}")
    return "\n".join(context_lines)


# ============ UTILITY FUNCTIONS ============

def check_rate_limit(user_id: int) -> tuple[bool, int]:
    """Check if user is rate limited. Returns (is_limited, seconds_remaining)."""
    now = time.time()
    last_query = user_last_query[user_id]
    if last_query and now - last_query < RATE_LIMIT_SECONDS:
        remaining = int(RATE_LIMIT_SECONDS - (now - last_query))
        return True, remaining
    return False, 0


def set_rate_limit(user_id: int) -> None:
    """Set rate limit timestamp after successful query."""
    user_last_query[user_id] = time.time()


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def get_message_content(message) -> str:
    """Extract text content from a message."""
    if message.text:
        return message.text
    if message.caption:
        return message.caption
    return ""


def is_forwarded_message(message) -> bool:
    """Check if message is forwarded (works with PTB v21+)."""
    if not message:
        return False
    # PTB v21+ uses forward_origin instead of forward_date
    return message.forward_origin is not None


def is_content_message(message) -> bool:
    """Check if message has analyzable content (forwarded, has links, etc.)."""
    if not message:
        return False
    content = get_message_content(message)
    # Has URLs
    if extract_urls(content):
        return True
    # Is forwarded
    if is_forwarded_message(message):
        return True
    # Has substantial text (more than just a few words)
    if len(content) > 100:
        return True
    return False


# ============ PERPLEXITY API ============

async def query_perplexity(
    question: str,
    referenced_content: str = None,
    user_name: str = None,
    context: str = None,
    user_facts: list[str] = None,
    group_facts: list[str] = None,
) -> str:
    """Query Perplexity API with context and memory."""

    system_prompt = """You are a casual friend helping out your buddies in Tallinn, Estonia.
When they ask about events, bars, restaurants, cinema, weather, or activities without specifying a location, assume they mean Tallinn.

Music/event preferences: your friends are into DIY, punk, rock, metal, hip-hop, trip-hop, underground stuff, and arthouse cinema - NOT mainstream pop, disco, or commercial events.

Keep responses VERY SHORT and casual - 1-2 sentences max. Write like texting a friend.
Use informal "ты" in Russian (never "вы"). Respond in the same language the user writes in.

IMPORTANT: NEVER use emojis. Instead use text emoticons: ) or )) for happy/funny things, ( or (( for sad things. Place emoticons directly after words WITHOUT space.

When user shares content (forwarded messages, links, articles) and asks about it:
- Answer their specific question about that content
- If they ask "кратко" or "о чём" - summarize briefly
- If they ask "правда?" or about facts - evaluate truthfulness
- If they ask "что думаешь?" - give your opinion
- If they ask about specific things in the content - answer specifically
- Understand the question from context, don't require exact keywords"""

    # Add memory context
    if user_facts:
        system_prompt += f"\n\nYou remember about this person: {', '.join(user_facts[:5])}"
    if group_facts:
        system_prompt += f"\n\nYou remember about this group: {', '.join(group_facts[:5])}"

    # Build the user message
    user_message = ""

    if referenced_content:
        user_message += f"[Content being discussed]:\n{referenced_content}\n\n"

    if context:
        user_message += f"[Recent conversation]:\n{context}\n\n"

    user_message += f"[User's question]: {question}"
    if user_name:
        user_message += f" (from {user_name})"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "sonar",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.3,
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
            answer = data["choices"][0]["message"]["content"]
            return clean_response(answer)
    except httpx.TimeoutException:
        return "Таймаут, попробуй ещё раз("
    except httpx.HTTPStatusError as e:
        logger.error(f"Perplexity API error: {e.response.status_code} - {e.response.text}")
        return "Ошибка API, попробуй позже("
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "Что-то пошло не так("


def clean_response(text: str) -> str:
    """Clean up response: remove citations [1][2] and fix emoticon spacing."""
    if not text:
        return text

    # Remove citation brackets like [1], [2], [6], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Fix emoticon spacing: " ))" or " (" should be "word))" or "word("
    # Match space followed by emoticons and move emoticons to previous word
    text = re.sub(r'\s+(\)+|\(+)', r'\1', text)

    # Clean up any double spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


async def extract_facts_from_response(question: str, answer: str, user_name: str) -> list[str]:
    """Extract memorable facts from a conversation."""
    facts = []
    patterns = [
        (r"люблю\s+(\w+)", "любит {}"),
        (r"нравится\s+(\w+)", "нравится {}"),
        (r"не люблю\s+(\w+)", "не любит {}"),
        (r"не ем\s+(\w+)", "не ест {}"),
        (r"работаю\s+(.+?)(?:\.|$)", "работает {}"),
        (r"живу\s+(.+?)(?:\.|$)", "живёт {}"),
    ]
    for pattern, fact_template in patterns:
        match = re.search(pattern, question.lower())
        if match:
            fact = fact_template.format(match.group(1))
            if user_name:
                fact = f"{user_name} {fact}"
            facts.append(fact)
    return facts


# ============ MESSAGE HANDLERS ============

def should_respond(update: Update, bot_username: str) -> bool:
    """Check if bot should respond to this message."""
    message = update.message
    if not message:
        return False

    # Must have some content
    if not message.text and not is_forwarded_message(message):
        return False

    # In private chats, always respond to messages with text
    if message.chat.type == "private" and message.text:
        return True

    # Respond if replying to bot's message
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.username == bot_username:
            return True

    # Respond if @mentioned
    if message.text and f"@{bot_username}" in message.text:
        return True

    return False


def extract_question(text: str, bot_username: str) -> str:
    """Remove bot mention from the question."""
    if not text:
        return ""
    return text.replace(f"@{bot_username}", "").strip()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "Привет! Спрашивай про ивенты, бары, кино, погоду - что угодно по Таллинну.\n\n"
        "Можешь пересылать посты или ссылки - ответь на них и спроси что угодно:\n"
        "- 'о чём это?'\n"
        "- 'какой фильм лучше?'\n"
        "- 'это правда?'\n\n"
        "В группе тэгай меня или отвечай на мои сообщения."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "Спрашивай что угодно про Таллинн!\n\n"
        "Анализ постов/ссылок:\n"
        "1. Перешли пост или скинь ссылку\n"
        "2. Ответь на него и спроси что хочешь\n\n"
        "Примеры вопросов:\n"
        "- 'о чём это?'\n"
        "- 'какой фильм из списка лучше?'\n"
        "- 'это правда?'\n"
        "- 'что думаешь?'\n\n"
        "Память:\n"
        "/remember <факт> - запомнить\n"
        "/forget - забыть всё"
    )


async def remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /remember command to save facts."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    username = update.effective_user.username
    user_name = USERNAME_TO_NAME.get(username, username)

    if not context.args:
        await update.message.reply_text(
            "Использование: /remember <факт>\n"
            "Например: /remember люблю IPA"
        )
        return

    fact = " ".join(context.args)
    if user_name:
        fact = f"{user_name}: {fact}"

    if update.effective_chat.type == "private":
        save_user_fact(user_id, fact)
    else:
        save_group_fact(chat_id, fact)

    await update.message.reply_text("Запомнил)")


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /forget command to clear memory."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if redis_client:
        try:
            if update.effective_chat.type == "private":
                redis_client.delete(f"user:{user_id}:facts")
            else:
                redis_client.delete(f"group:{chat_id}:facts")
            await update.message.reply_text("Забыл всё)")
        except Exception as e:
            logger.error(f"Failed to forget: {e}")
            await update.message.reply_text("Не получилось забыть(")
    else:
        await update.message.reply_text("Память не подключена(")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    message = update.message
    if not message:
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    username = update.effective_user.username
    user_name = USERNAME_TO_NAME.get(username) if username else None

    # Track context for all messages in groups (even if not responding)
    if message.text and update.effective_chat.type != "private":
        name = USERNAME_TO_NAME.get(username, username or "user")
        add_to_context(chat_id, "user", name, message.text)

    # Check if we should respond
    if not should_respond(update, BOT_USERNAME):
        return

    # Get the user's question
    question = extract_question(message.text or "", BOT_USERNAME)

    # Check for referenced content (reply to forwarded message, message with links, etc.)
    referenced_content = None
    reply_msg = message.reply_to_message

    # Case 1: User replies to another message (forwarded or with content)
    if reply_msg:
        reply_content = get_message_content(reply_msg)
        if reply_content:
            # Check if replied message is forwarded
            if is_forwarded_message(reply_msg):
                referenced_content = f"[Forwarded post]: {reply_content}"
            # Check if replied message has URLs
            elif extract_urls(reply_content):
                referenced_content = f"[Message with links]: {reply_content}"
            # Otherwise just include it as context
            elif len(reply_content) > 50:
                referenced_content = f"[Referenced message]: {reply_content}"

    # Case 2: Current message is forwarded (user forwarded + asked in same message or separately)
    if is_forwarded_message(message) and not referenced_content:
        content = get_message_content(message)
        if content:
            referenced_content = f"[Forwarded post]: {content}"
            # If no explicit question, default to analysis
            if not question:
                question = "расскажи об этом"

    # Case 3: Current message has URLs (no reply)
    if not referenced_content and question:
        urls = extract_urls(question)
        if urls:
            referenced_content = f"[Shared link]: {urls[0]}"

    # If still no question, prompt user
    if not question and not referenced_content:
        await message.reply_text(
            "Чё спросить хотел?",
            reply_to_message_id=message.message_id,
        )
        return

    # Default question if only content provided
    if not question and referenced_content:
        question = "о чём это?"

    # NOW check rate limit (after we know we will process)
    is_limited, remaining = check_rate_limit(user_id)
    if is_limited:
        await message.reply_text(
            f"Подожди {remaining} сек, не спеши)",
            reply_to_message_id=message.message_id,
        )
        return

    # Send typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Get context and memory
    conv_context = get_context_string(chat_id)
    user_facts = get_user_facts(user_id)
    group_facts = get_group_facts(chat_id) if chat_id != user_id else []

    # Query Perplexity
    logger.info(f"Query from {user_id} ({username}): {question[:50]}... [has_ref={referenced_content is not None}]")

    answer = await query_perplexity(
        question=question,
        referenced_content=referenced_content,
        user_name=user_name,
        context=conv_context,
        user_facts=user_facts,
        group_facts=group_facts,
    )

    # Set rate limit AFTER successful query
    set_rate_limit(user_id)

    # Add to context
    add_to_context(chat_id, "user", user_name or "user", question)
    add_to_context(chat_id, "assistant", "bot", answer)

    # Try to extract and save facts
    if not referenced_content:
        facts = await extract_facts_from_response(question, answer, user_name)
        for fact in facts:
            if chat_id == user_id:
                save_user_fact(user_id, fact)
            else:
                save_group_fact(chat_id, fact)

    await message.reply_text(answer, reply_to_message_id=message.message_id)


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
    logger.info(f"Redis connected: {redis_client is not None}")

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("remember", remember_command))
    application.add_handler(CommandHandler("forget", forget_command))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.FORWARDED) & ~filters.COMMAND,
        handle_message
    ))

    application.add_error_handler(error_handler)

    if os.getenv("RENDER"):
        port = int(os.getenv("PORT", 10000))
        webhook_url = os.getenv("WEBHOOK_URL")
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=TELEGRAM_TOKEN,
            webhook_url=f"{webhook_url}/{TELEGRAM_TOKEN}",
        )
    else:
        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
