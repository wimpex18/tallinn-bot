"""Anthropic Claude API client with streaming support."""

import re
import time
import logging

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Module-level client — set by main.py post_init
anthropic_client: anthropic.AsyncAnthropic = None

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
    """Detect if the text mentions a specific non-Tallinn location.

    Looks for patterns like "в Малаге", "в Берлине", "из Москвы" where the
    word after the preposition is a capitalized proper noun (checked via the
    original-case text) or an unknown word not in _NON_LOCATION_WORDS.
    """
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
    """Convert 'data:<mime>;base64,<data>' to an Anthropic image content block."""
    try:
        prefix, data = data_url.split(",", 1)
        media_type = prefix.split(":")[1].split(";")[0]
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
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
) -> str:
    """Query Claude with multi-turn context, memory, photos, and optional streaming.

    If telegram_bot / telegram_chat_id / telegram_message_id are provided the
    response is streamed live into the already-sent Telegram message via
    editMessageText, giving users a real-time typing effect.  The final cleaned
    text is always returned so callers can store it in conversation context.
    """
    t0 = time.monotonic()

    system_prompt = (
        'Отвечай на русском. Используй "ты". Кратко, 2-4 предложения. Без эмодзи. '
        'Ты общаешься как живой человек в чате, НЕ как энциклопедия и НЕ как ассистент. '
        'На болтовню и простые вопросы (привет, как дела, как настроение, что делаешь) '
        'отвечай КОРОТКО и НЕФОРМАЛЬНО, как друг — 1-2 предложения максимум. '
        'НЕ давай определения, НЕ объясняй понятия, НЕ перечисляй варианты, '
        'если тебя просто спрашивают о чём-то бытовом. '
        'Пример: "как настроение?" → "у меня норм, а у тебя как?" '
        'а НЕ "Настроение — это общее эмоциональное состояние..."\n\n'
        'По умолчанию ты помогаешь с вопросами про Таллинн, Эстонию. '
        'При поиске информации о местах, событиях и мероприятиях в Таллинне — '
        'ищи на АНГЛИЙСКОМ и ЭСТОНСКОМ языках (не на русском), так как большинство '
        'актуальной информации о Таллинне на этих языках. '
        'Хорошие источники: Facebook Events, visitestonia.com, tallinn.ee. '
        'Если не находишь на английском/эстонском — попробуй gloss.ee (русскоязычный сайт о Таллинне). '
        '\n\n'
        'КРИТИЧЕСКИ ВАЖНО — ГЕОГРАФИЯ ЗАПРОСА:\n'
        'Если пользователь спрашивает о КОНКРЕТНОМ городе или стране (Малага, Берлин, Москва, '
        'Барселона и т.д.) — отвечай ИМЕННО про тот город/страну. НЕ подменяй его Таллинном. '
        'Правила поиска для Таллинна (английский/эстонский) НЕ применяются к другим городам — '
        'для них ищи на языке, релевантном этому городу. '
        'Пример: «погода в Малаге» → ищи «Malaga weather», а НЕ «Tallinn weather».\n\n'
        'Если видишь "[PAGE NOT ACCESSIBLE]" — страница не загрузилась. '
        'СТРОГО ЗАПРЕЩЕНО угадывать содержание по URL-адресу или частям ссылки. '
        'Вместо этого ПОИЩИ информацию по этой ссылке или событию через веб-поиск. '
        'Если не нашёл — честно скажи что страница недоступна и ты не смог найти информацию. '
        'Если видишь "[PAYWALL]" — статья за пейволлом, доступен только превью. '
        'Расскажи что есть из превью и упомяни что полная статья доступна по подписке. '
        '\n\n'
        'КРИТИЧЕСКИ ВАЖНО — РАЗРЕШЕНИЕ МЕСТОИМЕНИЙ И ССЫЛОК:\n'
        'Когда в сообщении есть блок [Предыдущий ответ бота], пользователь отвечает '
        'на предыдущее сообщение бота. ВСЕ местоимения и указательные слова в вопросе '
        'пользователя (такие как «этот артист», «этот клуб», «там», «туда», «он», «она», '
        '«это место», «этот ресторан», «этого артиста», «на него» и т.д.) '
        'ССЫЛАЮТСЯ на конкретные названия из предыдущего ответа бота.\n'
        'ПЕРЕД формированием поискового запроса ты ОБЯЗАН:\n'
        '1. Найти в предыдущем ответе бота конкретное название (артиста, клуба, места, ГОРОДА и т.д.)\n'
        '2. Заменить местоимение/неявную ссылку в вопросе этим конкретным названием\n'
        '3. Сформулировать поисковый запрос на подходящем языке с конкретным названием\n'
        'Пример: предыдущий ответ упоминает «Ляпис Трубецкой», пользователь спрашивает '
        '«кто похож на этого артиста» → ищи «artists similar to Lyapis Trubetskoy».\n'
        'Пример: предыдущий ответ упоминает «клуб Privè», пользователь спрашивает '
        '«что ещё в этом клубе» → ищи «Privè Tallinn upcoming events».\n\n'
        'НЕЯВНЫЕ ПРОДОЛЖЕНИЯ (без местоимений):\n'
        'Если пользователь задаёт уточняющий вопрос БЕЗ явного упоминания предмета '
        '(например: «а на следующей неделе?», «а завтра?», «а цены?», «а вечером?»), '
        'он относится к ТОМУ ЖЕ месту/теме/городу из предыдущего ответа бота. '
        'Пример: предыдущий ответ — погода в Малаге, пользователь спрашивает «а на следующей неделе?» '
        '→ ищи «Malaga weather next week», а НЕ спрашивай где находится пользователь.'
    )

    if user_facts:
        system_prompt += f"\n\nТы помнишь про этого человека: {', '.join(user_facts[:5])}"
    if group_facts:
        system_prompt += f"\n\nТы помнишь про эту группу: {', '.join(group_facts[:5])}"
    if user_style:
        system_prompt += f"\n\n{user_style}"

    # Auto-append Tallinn context for place/event queries when not replying
    # to an existing bot message (which already carries location context).
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

    # Build user message content (text + optional images in Anthropic format)
    if photo_urls:
        user_content: list = [{"type": "text", "text": user_message_text}]
        for photo_url in photo_urls[:3]:
            img_block = _parse_base64_image(photo_url)
            if img_block:
                user_content.append(img_block)
        user_message_content = user_content
    else:
        user_message_content = user_message_text

    # Build messages array (Anthropic: system is a separate param, NOT in messages)
    messages: list[dict] = []

    if context_messages:
        for msg in context_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Ensure alternating roles (Anthropic requirement: messages must alternate user/assistant)
    if messages and messages[-1]["role"] == "user":
        if referenced_content:
            # Insert a minimal assistant separator to maintain role alternation
            # and avoid burying the referenced context inside a merged block.
            messages.append({"role": "assistant", "content": "(другие сообщения в чате)"})
            messages.append({"role": "user", "content": user_message_content})
        else:
            # Merge consecutive user messages
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
                # Existing content is already a list (multimodal) — append new text
                if isinstance(user_message_content, list):
                    messages[-1]["content"] = prev_content + user_message_content
                else:
                    messages[-1]["content"] = prev_content + [{"type": "text", "text": user_message_text}]
    else:
        messages.append({"role": "user", "content": user_message_content})

    # Log payload summary for debugging
    for i, msg in enumerate(messages):
        c = msg["content"]
        preview = c if isinstance(c, str) else "[multimodal]"
        if len(preview) > 300:
            preview = preview[:300] + "..."
        logger.info(f"Claude msg[{i}] role={msg['role']}: {preview}")

    _client = anthropic_client
    if _client is None:
        logger.error("anthropic_client is not initialised — check main.py post_init")
        return "Бот не готов, попробуй чуть позже("

    streaming = bool(telegram_bot and telegram_chat_id and telegram_message_id)

    try:
        if streaming:
            answer = await _stream_response(
                _client, system_prompt, messages,
                telegram_bot, telegram_chat_id, telegram_message_id,
            )
        else:
            answer = await _blocking_response(_client, system_prompt, messages)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"Claude responded in {elapsed_ms:.0f}ms ({len(answer)} chars)")
        return answer

    except anthropic.AuthenticationError:
        logger.error("Anthropic API authentication failed (401)")
        err = "Ошибка авторизации API — проверь ANTHROPIC_API_KEY)"
        await _safe_edit(telegram_bot, telegram_chat_id, telegram_message_id, err)
        return err

    except anthropic.RateLimitError:
        logger.warning("Anthropic API rate limit hit (429)")
        err = "Слишком много запросов, подожди минутку (429)"
        await _safe_edit(telegram_bot, telegram_chat_id, telegram_message_id, err)
        return err

    except anthropic.APIStatusError as exc:
        status = exc.status_code
        if status == 529:
            logger.warning("Anthropic API overloaded (529)")
            err = "Сервер перегружен, попробуй через минуту (529)"
        else:
            logger.error(f"Anthropic API error: {status} — {exc.message}")
            err = f"Проблема с API ({status}), попробуй позже)"
        await _safe_edit(telegram_bot, telegram_chat_id, telegram_message_id, err)
        return err

    except anthropic.APITimeoutError:
        logger.warning(f"Anthropic API timeout after {(time.monotonic() - t0)*1000:.0f}ms")
        err = "Слишком долго думаю, попробуй ещё раз)"
        await _safe_edit(telegram_bot, telegram_chat_id, telegram_message_id, err)
        return err

    except anthropic.APIConnectionError as exc:
        logger.error(f"Anthropic API connection error: {exc}")
        err = "Проблема с соединением, попробуй позже)"
        await _safe_edit(telegram_bot, telegram_chat_id, telegram_message_id, err)
        return err

    except Exception as exc:
        logger.error(f"Unexpected error querying Claude: {exc}")
        err = "Что-то пошло не так("
        await _safe_edit(telegram_bot, telegram_chat_id, telegram_message_id, err)
        return err


async def _blocking_response(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    messages: list[dict],
) -> str:
    """Non-streaming Claude call — returns the full response text."""
    response = await client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        temperature=ANTHROPIC_TEMPERATURE,
        system=system_prompt,
        messages=messages,
    )
    text = response.content[0].text if response.content else ""
    return _clean_response(text)


async def _stream_response(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    messages: list[dict],
    telegram_bot,
    chat_id: int,
    message_id: int,
) -> str:
    """Stream Claude response and pipe chunks into Telegram via editMessageText.

    Edits the message at most once per _STREAM_UPDATE_INTERVAL seconds to stay
    within Telegram rate limits, then does a clean final edit on completion.
    Returns the fully accumulated, cleaned response text.
    """
    accumulated = ""
    last_edit_time = 0.0

    async with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        temperature=ANTHROPIC_TEMPERATURE,
        system=system_prompt,
        messages=messages,
    ) as stream:
        async for chunk in stream.text_stream:
            accumulated += chunk
            now = time.monotonic()
            if (now - last_edit_time) >= _STREAM_UPDATE_INTERVAL and accumulated.strip():
                await _safe_edit(telegram_bot, chat_id, message_id, accumulated + "▌")
                last_edit_time = now

    final_text = _clean_response(accumulated)
    await _safe_edit(telegram_bot, chat_id, message_id, final_text)
    return final_text


async def _safe_edit(telegram_bot, chat_id, message_id, text: str) -> None:
    """Edit a Telegram message, silently ignoring failures (e.g. message unchanged)."""
    if not (telegram_bot and chat_id and message_id):
        return
    try:
        await telegram_bot.edit_message_text(
            text=text,
            chat_id=chat_id,
            message_id=message_id,
        )
    except Exception as exc:
        # "Message is not modified" errors are normal — ignore them
        logger.debug(f"edit_message_text skipped: {exc}")


def _clean_response(text: str) -> str:
    """Remove citation markers and fix emoticon spacing."""
    if not text:
        return text
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+(\)+|\(+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
