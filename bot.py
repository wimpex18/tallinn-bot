import os
import re
import time
import json
import logging
import base64
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse
from io import BytesIO

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


async def download_photo_as_base64(photo, bot) -> str:
    """Download a photo from Telegram and convert to base64."""
    try:
        # Get the file
        file = await bot.get_file(photo.file_id)

        # Download file bytes
        photo_bytes = await file.download_as_bytearray()

        # Convert to base64
        base64_string = base64.b64encode(photo_bytes).decode('utf-8')

        # Telegram photos are always JPEG, but check file path for PNG
        mime_type = "image/jpeg"
        if hasattr(file, 'file_path') and file.file_path:
            if file.file_path.endswith('.png'):
                mime_type = "image/png"
            logger.info(f"Photo MIME type: {mime_type}")

        return f"data:{mime_type};base64,{base64_string}"
    except Exception as e:
        logger.error(f"Failed to download photo: {e}")
        return None


def has_photo(message) -> bool:
    """Check if message has photo attachments."""
    if not message:
        return False
    return message.photo is not None and len(message.photo) > 0


async def send_typing(bot, chat_id: int) -> None:
    """Send typing action once."""
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass


# ============ PERPLEXITY API ============

async def query_perplexity(
    question: str,
    referenced_content: str = None,
    user_name: str = None,
    context: str = None,
    user_facts: list[str] = None,
    group_facts: list[str] = None,
    photo_urls: list[str] = None,
) -> str:
    """Query Perplexity API with context, memory, and photos."""

    system_prompt = """Ты друг в чате для русскоязычных в ТАЛЛИННЕ (Эстония). Отвечаешь КОРОТКО - 1-2 предложения.

КРИТИЧНО - ТОЛЬКО ТАЛЛИНН:
- ТЫ ЗНАЕШЬ ТОЛЬКО ПРО ТАЛЛИНН, ЭСТОНИЯ
- НИКОГДА не рекомендуй места в других городах (Москва, Питер и т.д.)
- Если спрашивают "куда сходить" - ТОЛЬКО места в Таллинне
- Ищи актуальные события в Таллинне на эту неделю

ВКУСЫ ГРУППЫ (учитывай при рекомендациях):
- Музыка: панк, рок, метал, хип-хоп, андеграунд, инди (НЕ поп, НЕ диско, НЕ мейнстрим)
- Бары: крафтовое пиво, коктейльные бары, dive bars (НЕ клубы, НЕ гламур)
- Кино: артхаус, фестивальное, авторское (НЕ блокбастеры)
- Общее: андеграунд, альтернатива, локальная сцена

ПОИСК СОБЫТИЙ - ОБЯЗАТЕЛЬНО ищи на сайтах:
- Концерты: sveta.ee, hall.ee, tfrec.com, kultuurikatel.ee, fotografiska.com/tallinn
- События: facebook.com/events (Tallinn), residentadvisor.net/events/ee, piletilevi.ee
- Кино: kinosoprus.ee, kino.artis.ee
- Типы: концерты, DJ сеты, vinyl nights, настолки, артхаус кино, выставки, DIY ивенты

Когда спрашивают "куда сходить" или "что делать":
- ИЩИ конкретные события на указанную дату
- Проверь сайты venue напрямую
- Укажи название события, место, время
- Если нашёл несколько - дай 2-3 варианта

МЕСТА В ТАЛЛИННЕ:
- Концерты: Sveta, Hall, Tapper, Kultuurikatel, Fotografiska
- Бары: Porogen, Tops, Pudel, St. Vitus, Koht, Labor
- Кино: Sõprus, Artis
- Районы: Telliskivi, Kalamaja, Rotermann, Noblessner

СТРОГИЕ ПРАВИЛА:
- Максимум 1-2 предложения
- Только "ты", никогда "вы"
- Без эмодзи. Только ) или ( после слова
- При рекомендации укажи название и район

Фото:
- Селфи/портрет: короткий комплимент
- Мем: короткая реакция
- Меню/афиша: ответь конкретно

Ссылки:
- ОТКРОЙ и проанализируй ссылки из контента
- Используй реальные данные со ссылок, не выдумывай
- Кратко изложи суть статьи/поста
- Если нашёл статью/источник - ПРИКРЕПИ ссылку на него

Если не знаешь про конкретное место в Таллинне - скажи что не знаешь."""

    # Add memory context
    if user_facts:
        system_prompt += f"\n\nТы помнишь про этого человека: {', '.join(user_facts[:5])}"
    if group_facts:
        system_prompt += f"\n\nТы помнишь про эту группу: {', '.join(group_facts[:5])}"

    # Build the user message
    user_message = ""

    # Extract URLs from referenced content for analysis
    extracted_urls = []
    if referenced_content:
        extracted_urls = extract_urls(referenced_content)
        user_message += f"[Content being discussed]:\n{referenced_content}\n\n"

        # Add URLs separately so Perplexity can fetch them
        if extracted_urls:
            user_message += f"[URLs to analyze]:\n"
            for url in extracted_urls[:5]:  # Limit to 5 URLs
                user_message += f"- {url}\n"
            user_message += "\n"

    if context:
        user_message += f"[Recent conversation]:\n{context}\n\n"

    user_message += f"[User's question]: {question}"
    if user_name:
        user_message += f" (from {user_name})"

    # Build user message content (with photos if provided)
    if photo_urls:
        # Format with images according to Perplexity API docs
        user_content = [{"type": "text", "text": user_message}]
        for photo_url in photo_urls[:3]:  # Limit to 3 photos
            user_content.append({
                "type": "image_url",
                "image_url": {"url": photo_url}
            })
        user_message_content = user_content
    else:
        user_message_content = user_message

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_content}
    ]

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "sonar",
        "messages": messages,
        "max_tokens": 150,
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
            try:
                answer = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                logger.error(f"Unexpected API response format: {data}")
                return "Не получил ответ от API("
            return clean_response(answer)
    except httpx.TimeoutException:
        return "Слишком долго думаю, попробуй ещё раз)"
    except httpx.HTTPStatusError as e:
        logger.error(f"Perplexity API error: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 429:
            return "Много запросов, подожди минутку)"
        return "Проблема с API, попробуй позже)"
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


async def smart_extract_facts(question: str, answer: str, user_name: str, chat_context: str = None) -> list[str]:
    """Use LLM to extract important facts from conversation."""
    if not question or len(question) < 10:
        return []

    prompt = f"""Извлеки важные факты о пользователе из этого диалога.
Пользователь: {user_name or 'unknown'}

Вопрос: {question}
Ответ: {answer}

{"Контекст чата: " + chat_context if chat_context else ""}

Выдай ТОЛЬКО факты о человеке (интересы, предпочтения, планы, работа, и т.д.)
Формат: один факт на строку, коротко (3-7 слов)
Если фактов нет - напиши "НЕТ"
Максимум 3 факта."""

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"].strip()

            if "НЕТ" in result.upper() or len(result) < 5:
                return []

            # Parse facts (one per line)
            facts = []
            for line in result.split("\n"):
                line = line.strip().lstrip("-•").strip()
                if line and len(line) > 3 and len(line) < 100:
                    if user_name and not line.startswith(user_name):
                        line = f"{user_name}: {line}"
                    facts.append(line)

            return facts[:3]  # Max 3 facts
    except Exception as e:
        logger.error(f"Failed to extract facts: {e}")
        return []


def save_user_interaction(user_id: int, user_name: str, username: str) -> None:
    """Save info about user who interacted with the bot."""
    if not redis_client or not user_name:
        return
    try:
        key = f"user:{user_id}:profile"
        redis_client.hset(key, mapping={
            "name": user_name,
            "username": username or "",
            "last_seen": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Failed to save user interaction: {e}")


# ============ MESSAGE HANDLERS ============

def should_respond(update: Update, bot_username: str) -> bool:
    """Check if bot should respond to this message."""
    message = update.message
    if not message:
        return False

    # Get text content (text or caption for photos)
    content = get_message_content(message)

    # Must have some content
    if not content and not is_forwarded_message(message) and not has_photo(message):
        return False

    # In private chats, always respond to messages with text/caption or photos
    if message.chat.type == "private" and (content or has_photo(message)):
        return True

    # Respond if replying to bot's message
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.username == bot_username:
            return True

    # Respond if @mentioned (check both text and caption)
    if content and f"@{bot_username}" in content:
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
        "Можешь пересылать посты, ссылки или фото:\n"
        "- 'о чём это?'\n"
        "- 'какой фильм лучше?'\n"
        "- 'это правда?'\n"
        "- 'что на фото?'\n\n"
        "В группе тэгай меня или отвечай на мои сообщения."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "Спрашивай что угодно про Таллинн!\n\n"
        "Анализ постов/ссылок:\n"
        "1. Перешли пост или скинь ссылку\n"
        "2. Ответь на него и спроси что хочешь\n\n"
        "Анализ фото:\n"
        "1. Скинь фото (меню, афиша, что угодно)\n"
        "2. Спроси что хочешь или просто жди ответ\n\n"
        "Анализ сообщений из чата:\n"
        "1. Сделай reply на любое сообщение\n"
        "2. Тэгни меня и спроси\n"
        "3. Я прочитаю сообщение + контекст разговора\n\n"
        "Примеры:\n"
        "- 'это правда?'\n"
        "- 'подробнее про это'\n"
        "- 'какой вариант лучше?'\n"
        "- 'что посоветуешь из меню?'\n\n"
        "Память:\n"
        "/memory - посмотреть что помню\n"
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

    # In group chats, check if user is admin
    if update.effective_chat.type != "private":
        member = await context.bot.get_chat_member(chat_id, user_id)
        if member.status not in ["creator", "administrator"]:
            await update.message.reply_text("Только админ может это делать)")
            return

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


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /memory command to view stored facts."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    username = update.effective_user.username
    user_name = USERNAME_TO_NAME.get(username, username) if username else "Ты"

    if not redis_client:
        await update.message.reply_text("Память не подключена(")
        return

    if update.effective_chat.type == "private":
        # Show user facts in private chat
        facts = get_user_facts(user_id)
        if facts:
            facts_text = "\n".join([f"- {fact}" for fact in facts])
            await update.message.reply_text(f"Что я помню про тебя:\n\n{facts_text}")
        else:
            await update.message.reply_text("Пока ничего не помню про тебя")
    else:
        # Show both user and group facts in group chat
        user_facts = get_user_facts(user_id)
        group_facts = get_group_facts(chat_id)

        response = ""
        if user_facts:
            facts_text = "\n".join([f"- {fact}" for fact in user_facts])
            response += f"Про {user_name}:\n{facts_text}\n\n"

        if group_facts:
            facts_text = "\n".join([f"- {fact}" for fact in group_facts])
            response += f"Про группу:\n{facts_text}"

        if not user_facts and not group_facts:
            response = "Пока ничего не помню"

        await update.message.reply_text(response.strip())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    message = update.message
    if not message:
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user = update.effective_user

    # Get display name with fallback: mapping -> first_name -> None
    if user.username and user.username in USERNAME_TO_NAME:
        user_name = USERNAME_TO_NAME[user.username]
    elif user.first_name:
        user_name = user.first_name
    else:
        user_name = None

    # Track context for all messages in groups (even if not responding)
    msg_content = get_message_content(message)
    if msg_content and update.effective_chat.type != "private":
        context_name = user_name or "user"
        add_to_context(chat_id, "user", context_name, msg_content)

    # Check if we should respond
    if not should_respond(update, BOT_USERNAME):
        return

    # Get the user's question
    # Get question from text OR caption (for photos with text)
    question = extract_question(get_message_content(message), BOT_USERNAME)

    # Check for referenced content (reply to forwarded message, message with links, etc.)
    referenced_content = None
    reply_msg = message.reply_to_message

    # Case 1: User replies to another message
    # When bot is tagged in a reply, ALWAYS analyze the replied message
    msg_text = get_message_content(message)
    if reply_msg and msg_text and f"@{BOT_USERNAME}" in msg_text:
        reply_content = get_message_content(reply_msg)
        if reply_content:
            # Get author info if available
            reply_author = "unknown"
            if reply_msg.from_user:
                reply_user = reply_msg.from_user
                if reply_user.username and reply_user.username in USERNAME_TO_NAME:
                    reply_author = USERNAME_TO_NAME[reply_user.username]
                elif reply_user.first_name:
                    reply_author = reply_user.first_name
                elif reply_user.username:
                    reply_author = reply_user.username

            # Check if replied message is forwarded
            if is_forwarded_message(reply_msg):
                referenced_content = f"[Forwarded post]: {reply_content}"
            # Check if replied message has URLs
            elif extract_urls(reply_content):
                referenced_content = f"[Message with links]: {reply_content}"
            # ANY other message - include it with author
            else:
                referenced_content = f"[Message from {reply_author}]: {reply_content}"

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

    # Check if photo without text
    has_current_photo = has_photo(message)
    has_reply_photo = reply_msg and has_photo(reply_msg)

    # If still no question, prompt user (unless there's a photo)
    if not question and not referenced_content and not has_current_photo and not has_reply_photo:
        await message.reply_text(
            "Чё спросить хотел?",
            reply_to_message_id=message.message_id,
        )
        return

    # Default question if only content provided
    if not question and referenced_content:
        question = "о чём это?"

    # Default question if only photo provided
    if not question and (has_current_photo or has_reply_photo):
        question = "что на фото?"

    # NOW check rate limit (after we know we will process)
    is_limited, remaining = check_rate_limit(user_id)
    if is_limited:
        await message.reply_text(
            f"Подожди {remaining} сек, не спеши)",
            reply_to_message_id=message.message_id,
        )
        return

    # Send typing indicator
    await send_typing(context.bot, chat_id)

    # Get context and memory
    conv_context = get_context_string(chat_id)
    user_facts = get_user_facts(user_id)
    group_facts = get_group_facts(chat_id) if chat_id != user_id else []

    # Check for photos to analyze
    photo_urls = []

    # Check current message for photos
    if has_photo(message):
        # Get the highest quality photo
        photo = message.photo[-1]
        photo_url = await download_photo_as_base64(photo, context.bot)
        if photo_url:
            photo_urls.append(photo_url)
            logger.info(f"Added photo from current message")

    # Check replied message for photos
    if reply_msg and has_photo(reply_msg):
        photo = reply_msg.photo[-1]
        photo_url = await download_photo_as_base64(photo, context.bot)
        if photo_url:
            photo_urls.append(photo_url)
            logger.info(f"Added photo from replied message")

    # Query Perplexity
    logger.info(f"Query from {user_id} ({user_name}): {question[:50]}... [has_ref={referenced_content is not None}, photos={len(photo_urls)}]")

    answer = await query_perplexity(
        question=question,
        referenced_content=referenced_content,
        user_name=user_name,
        context=conv_context,
        user_facts=user_facts,
        group_facts=group_facts,
        photo_urls=photo_urls if photo_urls else None,
    )

    # Set rate limit AFTER successful query
    set_rate_limit(user_id)

    # Add to context
    add_to_context(chat_id, "user", user_name or "user", question)
    add_to_context(chat_id, "assistant", "bot", answer)

    # Save user interaction (learn who talks to us)
    save_user_interaction(user_id, user_name, user.username)

    # Send response immediately, then learn in background
    await message.reply_text(answer, reply_to_message_id=message.message_id)

    # Smart fact extraction (runs after response sent)
    try:
        # Use LLM to extract facts from conversation
        facts = await smart_extract_facts(
            question=question,
            answer=answer,
            user_name=user_name,
            chat_context=conv_context
        )

        # Fallback to regex if LLM fails
        if not facts:
            facts = await extract_facts_from_response(question, answer, user_name)

        for fact in facts:
            if chat_id == user_id:
                save_user_fact(user_id, fact)
            else:
                save_group_fact(chat_id, fact)

        if facts:
            logger.info(f"Learned facts: {facts}")
    except Exception as e:
        logger.error(f"Learning failed: {e}")


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
    application.add_handler(CommandHandler("memory", memory_command))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.FORWARDED | filters.PHOTO) & ~filters.COMMAND,
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
