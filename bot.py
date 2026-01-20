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

# Rate limiting
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
        # Keep only last 20 facts per user
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
        "content": content[:500],  # Limit message size
        "time": time.time()
    })
    # Keep only last N messages
    if len(chat_context[chat_id]) > CONTEXT_SIZE:
        chat_context[chat_id] = chat_context[chat_id][-CONTEXT_SIZE:]


def get_context_string(chat_id: int) -> str:
    """Get recent conversation context as a string."""
    if not chat_context[chat_id]:
        return ""

    context_lines = []
    for msg in chat_context[chat_id][-5:]:  # Last 5 messages for prompt
        name = msg.get("name", "user")
        content = msg["content"]
        context_lines.append(f"{name}: {content}")

    return "\n".join(context_lines)


# ============ UTILITY FUNCTIONS ============

def is_rate_limited(user_id: int) -> bool:
    """Check if user is rate limited."""
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


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def detect_intent(text: str) -> str:
    """Detect user intent for forwarded/link analysis."""
    text_lower = text.lower()

    if any(word in text_lower for word in ["кратко", "резюме", "summarize", "summary", "суть", "о чём", "о чем"]):
        return "summarize"
    if any(word in text_lower for word in ["правда", "true", "факт", "fact", "верно", "врёт", "врет", "ложь"]):
        return "factcheck"
    if any(word in text_lower for word in ["перевод", "translate", "переведи"]):
        return "translate"
    if any(word in text_lower for word in ["что думаешь", "мнение", "opinion", "анализ", "analyze"]):
        return "analyze"

    return "general"


# ============ PERPLEXITY API ============

async def query_perplexity(
    question: str,
    user_name: str = None,
    context: str = None,
    user_facts: list[str] = None,
    group_facts: list[str] = None,
    mode: str = "general"
) -> str:
    """Query Perplexity API with context and memory."""

    base_prompt = """You are a casual friend helping out your buddies in Tallinn, Estonia.
When they ask about events, bars, restaurants, cinema, weather, or activities without specifying a location, assume they mean Tallinn.

Music/event preferences: your friends are into DIY, punk, rock, metal, hip-hop, trip-hop, underground stuff, and arthouse cinema - NOT mainstream pop, disco, or commercial events.

Keep responses VERY SHORT and casual - 1-2 sentences max. Write like texting a friend.
Use informal "ты" in Russian (never "вы"). Respond in the same language the user writes in.

IMPORTANT: NEVER use emojis. Instead use text emoticons: ) or )) for happy/funny things, ( or (( for sad things. Place emoticons directly after words WITHOUT space."""

    # Add mode-specific instructions
    mode_instructions = {
        "summarize": "\n\nYour task: Summarize the provided content in 2-3 short sentences. Be concise.",
        "factcheck": "\n\nYour task: Fact-check the claims in the provided content. Be direct about what's true, questionable, or false.",
        "translate": "\n\nYour task: Translate the provided content. If it's in Russian, translate to English. If in English, translate to Russian.",
        "analyze": "\n\nYour task: Give your brief opinion/analysis of the provided content. Be honest and direct.",
        "link": "\n\nYour task: Summarize what this link/article is about in 2-3 sentences.",
        "general": ""
    }

    system_prompt = base_prompt + mode_instructions.get(mode, "")

    # Add memory context
    memory_context = ""
    if user_facts:
        memory_context += f"\n\nYou remember about this person: {', '.join(user_facts[:5])}"
    if group_facts:
        memory_context += f"\n\nYou remember about this group: {', '.join(group_facts[:5])}"

    if memory_context:
        system_prompt += memory_context

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation context if available
    if context:
        messages.append({
            "role": "user",
            "content": f"Recent conversation for context:\n{context}\n\n---\nCurrent question: {question}"
        })
    else:
        user_context = f" (from {user_name})" if user_name else ""
        messages.append({"role": "user", "content": f"{question}{user_context}"})

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
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "Таймаут, попробуй ещё раз("
    except httpx.HTTPStatusError as e:
        logger.error(f"Perplexity API error: {e.response.status_code} - {e.response.text}")
        return "Ошибка API, попробуй позже("
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "Что-то пошло не так("


async def extract_facts_from_response(question: str, answer: str, user_name: str) -> list[str]:
    """Extract memorable facts from a conversation using AI."""
    # Simple heuristic extraction - could be enhanced with AI
    facts = []

    # Look for preference patterns
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

    # Check for text or forwarded content
    has_content = message.text or message.forward_date
    if not has_content:
        return False

    # Respond if replying to bot's message
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.username == bot_username:
            return True

    # Respond if @mentioned
    if message.text and f"@{bot_username}" in message.text:
        return True

    # In private chats, always respond
    if message.chat.type == "private":
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
        "Можешь пересылать посты - я их проанализирую, скажу что думаю или кратко перескажу.\n\n"
        "В группе тэгай меня или отвечай на мои сообщения."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "Спрашивай что угодно про Таллинн!\n\n"
        "Примеры:\n"
        "- какая сегодня погода?\n"
        "- где концерты на выходных?\n"
        "- есть крутые бары в старом городе?\n\n"
        "Анализ постов:\n"
        "- перешли пост + 'кратко' = резюме\n"
        "- перешли пост + 'правда?' = фактчек\n"
        "- перешли пост + 'что думаешь?' = мнение\n\n"
        "Скинь ссылку - расскажу о чём статья."
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
    if not should_respond(update, BOT_USERNAME):
        # Still track context for group messages
        if update.message and update.message.text and update.effective_chat.type != "private":
            username = update.effective_user.username if update.effective_user else None
            name = USERNAME_TO_NAME.get(username, username or "user")
            add_to_context(update.effective_chat.id, "user", name, update.message.text)
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
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

    message = update.message
    question = extract_question(message.text or "", BOT_USERNAME)

    # Detect if this is a forwarded message
    is_forwarded = message.forward_date is not None
    forwarded_text = ""
    if is_forwarded:
        # Get forwarded content
        if message.text:
            forwarded_text = message.text
        elif message.caption:
            forwarded_text = message.caption

        # Determine intent
        intent = detect_intent(question) if question else "summarize"

        if forwarded_text:
            question = f"[Forwarded message]: {forwarded_text}\n\nUser request: {question or 'проанализируй'}"
            mode = intent
        else:
            await update.message.reply_text(
                "Не вижу текста в пересланном сообщении(",
                reply_to_message_id=message.message_id,
            )
            return

    # Detect URLs in message
    urls = extract_urls(question) if question else []
    mode = "general"

    if urls and not is_forwarded:
        mode = "link"
        intent = detect_intent(question)
        if intent != "general":
            mode = intent
        question = f"[Link shared]: {urls[0]}\n\nUser request: {question or 'о чём это?'}"

    if not question:
        await update.message.reply_text(
            "Чё спросить хотел?",
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
    logger.info(f"Query from {user_id} ({username}): {question[:50]}... [mode={mode}]")
    answer = await query_perplexity(
        question=question,
        user_name=user_name,
        context=conv_context,
        user_facts=user_facts,
        group_facts=group_facts,
        mode=mode
    )

    # Add to context
    add_to_context(chat_id, "user", user_name or "user", question)
    add_to_context(chat_id, "assistant", "bot", answer)

    # Try to extract and save facts (simple heuristic)
    if not is_forwarded and not urls:
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

    # Build application
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

    # Error handler
    application.add_error_handler(error_handler)

    # Start bot
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
