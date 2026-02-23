"""Perplexity Sonar API client."""

import re
import logging
import time

import httpx

from config import PERPLEXITY_API_KEY, PERPLEXITY_MAX_TOKENS, PERPLEXITY_TEMPERATURE

logger = logging.getLogger(__name__)

# Set by main.py post_init
http_client: httpx.AsyncClient = None

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
    # Check for prepositional phrases: в/во/на/из + word
    for m in re.finditer(r'\b(?:в|во|на|из|про)\s+(\w{3,})', text):
        word = m.group(1).lower()
        if word in _NON_LOCATION_WORDS:
            continue
        # Skip if it's one of the place_keywords (venue types like "бар", "клуб")
        if word in {
            "бар", "ресторан", "кафе", "клуб", "кино", "магазин",
            "музей", "театр", "галерею", "галерея",
        }:
            continue
        # Likely a location name (e.g., "малаге", "берлине", "москве")
        return True
    return False


async def query_perplexity(
    question: str,
    referenced_content: str = None,
    user_name: str = None,
    context_messages: list[dict] = None,
    user_facts: list[str] = None,
    group_facts: list[str] = None,
    photo_urls: list[str] = None,
    user_style: str = None,
) -> str:
    """Query Perplexity API with multi-turn context, memory, and photos.

    Uses proper alternating user/assistant messages so the model can
    resolve pronouns and follow-up references from conversation history.
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

    # Auto-append "Tallinn, Estonia" for place/event queries,
    # but ONLY if:
    #  - no other city/location is explicitly mentioned
    #  - NOT a reply to a previous bot message (referenced_content exists),
    #    because in that case the location context comes from the referenced
    #    answer, not the default Tallinn assumption
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

    # Build the current user message (question + any referenced content).
    # When there is referenced content (e.g. reply to bot), put the context
    # first so the model sees it before the question and can resolve pronouns.
    if referenced_content:
        user_message = f"{referenced_content}\n\nВопрос пользователя: {question}"
    else:
        user_message = question

    # Build message content (with photos if provided)
    if photo_urls:
        user_content = [{"type": "text", "text": user_message}]
        for photo_url in photo_urls[:3]:
            user_content.append({"type": "image_url", "image_url": {"url": photo_url}})
        user_message_content = user_content
    else:
        user_message_content = user_message

    # Build messages array: system + conversation history + current message
    messages = [{"role": "system", "content": system_prompt}]

    # Add multi-turn conversation history (proper alternating roles)
    if context_messages:
        # Ensure history starts with "user" role (API requirement)
        for msg in context_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Ensure alternating roles (API requirement).
    if len(messages) > 1 and messages[-1]["role"] == "user":
        if referenced_content:
            # When the user replies to the bot with context, don't merge with
            # unrelated previous user messages — that buries the context.
            # Insert a minimal assistant separator to maintain role alternation.
            messages.append({"role": "assistant", "content": "(другие сообщения в чате)"})
            messages.append({"role": "user", "content": user_message_content})
        else:
            # Regular consecutive user messages — merge them.
            prev_text = messages[-1]["content"]
            combined_text = f"{prev_text}\n{user_message}"
            if photo_urls:
                content = [{"type": "text", "text": combined_text}]
                for photo_url in photo_urls[:3]:
                    content.append({"type": "image_url", "image_url": {"url": photo_url}})
                messages[-1]["content"] = content
            else:
                messages[-1]["content"] = combined_text
    else:
        messages.append({"role": "user", "content": user_message_content})

    # Log the full message payload for debugging context resolution
    for i, msg in enumerate(messages):
        content_preview = msg["content"] if isinstance(msg["content"], str) else "[multimodal]"
        if len(content_preview) > 300:
            content_preview = content_preview[:300] + "..."
        logger.info(f"Perplexity msg[{i}] role={msg['role']}: {content_preview}")

    payload = {
        "model": "sonar",
        "messages": messages,
        "max_tokens": PERPLEXITY_MAX_TOKENS,
        "temperature": PERPLEXITY_TEMPERATURE,
    }

    try:
        _client = http_client or httpx.AsyncClient(timeout=30.0)
        try:
            response = await _client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        finally:
            if _client is not http_client:
                await _client.aclose()
        response.raise_for_status()
        data = response.json()
        elapsed_ms = (time.monotonic() - t0) * 1000
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.error(f"Unexpected API response format: {data}")
            return "Не получил ответ от API("
        logger.info(f"Perplexity responded in {elapsed_ms:.0f}ms ({len(answer)} chars)")
        return _clean_response(answer)
    except httpx.TimeoutException:
        logger.warning(f"Perplexity timeout after {(time.monotonic() - t0)*1000:.0f}ms")
        return "Слишком долго думаю, попробуй ещё раз)"
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        logger.error(f"Perplexity API error: {status} - {e.response.text}")
        if status == 401:
            return "Ошибка авторизации API (401) — проверь PERPLEXITY_API_KEY)"
        if status == 402:
            return "Закончились кредиты Perplexity API (402) — пополни баланс)"
        if status == 422:
            return f"Ошибка запроса к API (422) — возможно, неверная модель или параметры)"
        if status == 429:
            return "Много запросов, подожди минутку)"
        if status >= 500:
            return f"Сервер Perplexity временно недоступен ({status}), попробуй позже)"
        return f"Проблема с API ({status}), попробуй позже)"
    except Exception as e:
        logger.error(f"Unexpected error querying Perplexity: {e}")
        return "Что-то пошло не так("


def _clean_response(text: str) -> str:
    """Remove citations and fix emoticon spacing."""
    if not text:
        return text
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+(\)+|\(+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
