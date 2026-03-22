"""Mistral API client with streaming support."""

import re
import time
import logging

from mistralai.client import Mistral

from config import (
    MISTRAL_MODEL,
    MISTRAL_MAX_TOKENS,
    MISTRAL_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Module-level client — set by main.py post_init
mistral_client: Mistral = None

# Stream update interval: update Telegram message at most every N seconds
_STREAM_UPDATE_INTERVAL = 1.0

# Words that commonly follow prepositions (в/на/из) but are NOT location names.
_NON_LOCATION_WORDS = {
    # Time words
    "понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье",
    "неделю", "неделе", "месяц", "месяце", "году", "год", "выходные", "выходных",
    "утро", "утра", "вечер", "вечера", "ночь", "ночи", "день", "дня",
    "январе", "феврале", "марте", "апреле", "мае", "июне",
    "июле", "августе", "сентябре", "октябре", "ноябре", "декабре",
    # Demonstratives / pronouns
    "этом", "этой", "этих", "том", "той", "тех", "нём", "ней", "них",
    "каком", "какой", "каких", "нашем", "нашей", "любом", "любой",
    # Common abstract words
    "ближайшем", "ближайшей", "следующем", "следующей", "следующую",
    "прошлом", "прошлой", "прошлую",
    "центре", "районе", "городе", "стране", "округе", "области",
    "общем", "целом", "итоге", "основном", "принципе",
    "жизни", "работе", "школе", "деле", "сети", "интернете",
    "курсе", "группе", "чате", "теме", "наличии", "меню",
    # Place keywords (these are the venue types, not locations)
    "баре", "ресторане", "кафе", "клубе", "кинотеатре", "магазине",
    "музее", "театре", "галерее",
}


def _has_non_tallinn_location(text: str) -> bool:
    """Detect if the text mentions a specific non-Tallinn location."""
    for m in re.finditer(r'\b(?:в|во|на|из|про)\s+(\w{3,})', text):
        word = m.group(1).lower()
        if word in _NON_LOCATION_WORDS:
            continue
        if word in {
            "бар", "ресторан", "кафе", "клуб", "кино", "магазин",
            "музей", "театр", "галерею", "галерея",
        }:
            continue
        return True
    return False


def _parse_base64_image(data_url: str) -> dict | None:
    """Convert 'data:<mime>;base64,<data>' to a Mistral image content block."""
    try:
        prefix, _ = data_url.split(",", 1)
        if not prefix.startswith("data:") or ";base64" not in prefix:
            return None
        return {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
    except Exception as exc:
        logger.warning(f"Failed to parse base64 image: {exc}")
        return None


async def query_claude(
    question: str,
    referenced_content: str = None,
    user_name: str = None,
    context_messages: list[dict] = None,
    user_facts: list[str] = None,
    group_facts: list[str] = None,
    photo_urls: list[str] = None,
    user_style: str = None,
    telegram_bot=None,
    telegram_chat_id: int = None,
    telegram_message_id: int = None,
    thinking_budget: int = 0,
) -> str:
    """Query Mistral with multi-turn context, memory, and optional streaming.

    If telegram_bot / telegram_chat_id / telegram_message_id are provided the
    response is streamed live into the already-sent Telegram message.
    The final cleaned text is always returned so callers can store it in context.
    """
    t0 = time.monotonic()

    # ── System prompt ─────────────────────────────────────────────
    _STATIC_SYSTEM = (
        'Отвечай на русском. Используй "ты". Кратко, 2-4 предложения. Без эмодзи. '
        'Ты общаешься как живой человек в чате, НЕ как энциклопедия и НЕ как ассистент. '
        'На болтовню и простые вопросы (привет, как дела, как настроение, что делаешь) '
        'отвечай КОРОТКО и НЕФОРМАЛЬНО, как друг — 1-2 предложения максимум. '
        'НЕ давай определения, НЕ объясняй понятия, НЕ перечисляй варианты, '
        'если тебя просто спрашивают о чём-то бытовом. '
        'Пример: "как настроение?" → "у меня норм, а у тебя как?" '
        'а НЕ "Настроение — это общее эмоциональное состояние..."\n\n'
        'По умолчанию ты помогаешь с вопросами про Таллинн, Эстонию. '
        'Если в сообщении есть блок с данными о погоде, расписании или другой актуальной информацией — '
        'используй эти данные для ответа. '
        'Для вопросов о текущих событиях, расписаниях и ценах, по которым нет данных — '
        'честно скажи что у тебя нет актуальной информации и предложи проверить на сайте.\n\n'
        'КРИТИЧЕСКИ ВАЖНО — ГЕОГРАФИЯ ЗАПРОСА:\n'
        'Если пользователь спрашивает о КОНКРЕТНОМ городе или стране (Малага, Берлин, Москва, '
        'Барселона и т.д.) — отвечай ИМЕННО про тот город/страну. НЕ подменяй его Таллинном.\n\n'
        'При ответе на вопрос о погоде: один короткий ответ — температура + условие. '
        'Упоминай ветер только если сильный. Не копируй сырые данные и не пиши таблицы.\n\n'
        'КРИТИЧЕСКИ ВАЖНО — РАЗРЕШЕНИЕ МЕСТОИМЕНИЙ И ССЫЛОК:\n'
        'Когда в сообщении есть блок [Предыдущий ответ бота], пользователь отвечает '
        'на предыдущее сообщение бота. ВСЕ местоимения и указательные слова в вопросе '
        'пользователя (такие как «этот артист», «этот клуб», «там», «туда», «он», «она», '
        '«это место», «этот ресторан», «этого артиста», «на него» и т.д.) '
        'ССЫЛАЮТСЯ на конкретные названия из предыдущего ответа бота.\n'
        'ПЕРЕД формированием ответа ты ОБЯЗАН:\n'
        '1. Найти в предыдущем ответе бота конкретное название (артиста, клуба, места, ГОРОДА и т.д.)\n'
        '2. Заменить местоимение/неявную ссылку в вопросе этим конкретным названием\n\n'
        'НЕЯВНЫЕ ПРОДОЛЖЕНИЯ (без местоимений):\n'
        'Если пользователь задаёт уточняющий вопрос БЕЗ явного упоминания предмета, '
        'он относится к ТОМУ ЖЕ месту/теме/городу из предыдущего ответа бота.'
    )

    dynamic_parts = [_STATIC_SYSTEM]
    if user_facts:
        dynamic_parts.append(f"Ты помнишь про этого человека: {', '.join(user_facts[:5])}")
    if group_facts:
        dynamic_parts.append(f"Ты помнишь про эту группу: {', '.join(group_facts[:5])}")
    if user_style:
        dynamic_parts.append(user_style)
    system_text = "\n\n".join(dynamic_parts)

    # Auto-append Tallinn context for place/event queries
    if not referenced_content:
        question_lower = question.lower()
        place_keywords = [
            "бар", "ресторан", "кафе", "клуб", "кино", "магазин", "музей", "театр", "галерея",
            "концерт", "мероприятие", "событие", "фестиваль", "выставка", "вечеринка", "шоу",
            "ивент", "event", "афиша", "тусовка", "движ",
            "сегодня", "завтра", "выходные", "вечером", "weekend",
            "куда", "где", "посоветуй", "порекомендуй", "подскажи", "сходить", "пойти",
        ]
        location_keywords = ["таллин", "tallinn", "эстони", "estonia"]
        has_place_keyword = any(kw in question_lower for kw in place_keywords)
        has_tallinn_mention = any(loc in question_lower for loc in location_keywords)
        has_other_location = _has_non_tallinn_location(question_lower)
        if has_place_keyword and not has_tallinn_mention and not has_other_location:
            question = f"{question} (Tallinn, Estonia)"

    # Build the current user message text
    if referenced_content:
        user_message_text = f"{referenced_content}\n\nВопрос пользователя: {question}"
    else:
        user_message_text = question

    # Build user message content (text + optional images)
    if photo_urls:
        user_content: list = [{"type": "text", "text": user_message_text}]
        for photo_url in photo_urls[:3]:
            img_block = _parse_base64_image(photo_url)
            if img_block:
                user_content.append(img_block)
        user_message_content = user_content
    else:
        user_message_content = user_message_text

    # Build messages array — system goes as the first message
    messages: list[dict] = [{"role": "system", "content": system_text}]

    if context_messages:
        for msg in context_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Ensure alternating roles (Mistral requires alternating user/assistant after system)
    if messages and messages[-1]["role"] == "user":
        if referenced_content:
            messages.append({"role": "assistant", "content": "(другие сообщения в чате)"})
            messages.append({"role": "user", "content": user_message_content})
        else:
            prev_content = messages[-1]["content"]
            if isinstance(prev_content, str):
                combined_text = f"{prev_content}\n{user_message_text}"
                if photo_urls:
                    merged: list = [{"type": "text", "text": combined_text}]
                    for photo_url in photo_urls[:3]:
                        img_block = _parse_base64_image(photo_url)
                        if img_block:
                            merged.append(img_block)
                    messages[-1]["content"] = merged
                else:
                    messages[-1]["content"] = combined_text
            else:
                if isinstance(user_message_content, list):
                    messages[-1]["content"] = prev_content + user_message_content
                else:
                    messages[-1]["content"] = prev_content + [{"type": "text", "text": user_message_text}]
    else:
        messages.append({"role": "user", "content": user_message_content})

    # Log payload summary
    for i, msg in enumerate(messages):
        c = msg["content"]
        preview = c if isinstance(c, str) else "[multimodal]"
        if len(preview) > 300:
            preview = preview[:300] + "..."
        logger.info(f"Mistral msg[{i}] role={msg['role']}: {preview}")

    _client = mistral_client
    if _client is None:
        logger.error("mistral_client is not initialised — check main.py post_init")
        return "Бот не готов, попробуй чуть позже("

    streaming = bool(telegram_bot and telegram_chat_id and telegram_message_id)

    try:
        if streaming:
            answer = await _stream_response(
                _client, messages,
                telegram_bot, telegram_chat_id, telegram_message_id,
            )
        else:
            answer = await _blocking_response(_client, messages)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"Mistral responded in {elapsed_ms:.0f}ms ({len(answer)} chars)")
        return answer

    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status == 401:
            logger.error("Mistral API authentication failed (401)")
            err = "Ошибка авторизации API — проверь MISTRAL_API_KEY)"
        elif status == 429:
            logger.warning("Mistral API rate limit hit (429)")
            err = "Слишком много запросов, подожди минутку (429)"
        elif status == 400:
            logger.error(f"Mistral API bad request (400): {exc}")
            err = "Ошибка запроса к Mistral (400) — проверь логи Render"
        elif status and status >= 500:
            logger.warning(f"Mistral API server error ({status})")
            err = f"Сервер перегружен, попробуй через минуту ({status})"
        else:
            logger.error(f"Unexpected error querying Mistral: {exc}", exc_info=True)
            err = "Что-то пошло не так("
        await _safe_edit(telegram_bot, telegram_chat_id, telegram_message_id, err)
        return err


async def _blocking_response(client: Mistral, messages: list[dict]) -> str:
    """Non-streaming Mistral call — returns the full response text."""
    response = await client.chat.complete_async(
        model=MISTRAL_MODEL,
        max_tokens=MISTRAL_MAX_TOKENS,
        temperature=MISTRAL_TEMPERATURE,
        messages=messages,
    )
    text = response.choices[0].message.content or ""
    return _clean_response(text)


async def _stream_response(
    client: Mistral,
    messages: list[dict],
    telegram_bot,
    chat_id: int,
    message_id: int,
) -> str:
    """Stream Mistral response and pipe chunks into Telegram via editMessageText."""
    accumulated = ""
    last_edit_time = 0.0

    async with client.chat.stream_async(
        model=MISTRAL_MODEL,
        max_tokens=MISTRAL_MAX_TOKENS,
        temperature=MISTRAL_TEMPERATURE,
        messages=messages,
    ) as stream:
        async for event in stream:
            chunk = event.data.choices[0].delta.content
            if chunk:
                accumulated += chunk
                now = time.monotonic()
                if (now - last_edit_time) >= _STREAM_UPDATE_INTERVAL and accumulated.strip():
                    await _safe_edit(telegram_bot, chat_id, message_id, accumulated + "▌")
                    last_edit_time = now

    final_text = _clean_response(accumulated)
    await _safe_edit(telegram_bot, chat_id, message_id, final_text)
    return final_text


async def _safe_edit(telegram_bot, chat_id, message_id, text: str) -> None:
    """Edit a Telegram message, silently ignoring failures."""
    if not (telegram_bot and chat_id and message_id):
        return
    try:
        await telegram_bot.edit_message_text(
            text=text,
            chat_id=chat_id,
            message_id=message_id,
        )
    except Exception as exc:
        logger.debug(f"edit_message_text skipped: {exc}")


def _clean_response(text: str) -> str:
    """Remove citation markers and fix emoticon spacing."""
    if not text:
        return text
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+(\)+|\(+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
