"""Perplexity Sonar API client."""

import re
import logging
import time

import httpx

from config import PERPLEXITY_API_KEY, PERPLEXITY_MAX_TOKENS, PERPLEXITY_TEMPERATURE

logger = logging.getLogger(__name__)

# Set by main.py post_init
http_client: httpx.AsyncClient = None


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
        'ВАЖНО: При поиске информации о местах, событиях и мероприятиях в Таллинне — '
        'ищи на АНГЛИЙСКОМ и ЭСТОНСКОМ языках (не на русском), так как большинство '
        'актуальной информации о Таллинне на этих языках. '
        'Хорошие источники: Facebook Events, visitestonia.com, tallinn.ee. '
        'Если не находишь на английском/эстонском — попробуй gloss.ee (русскоязычный сайт о Таллинне). '
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
        '1. Найти в предыдущем ответе бота конкретное название (артиста, клуба, места и т.д.)\n'
        '2. Заменить местоимение в вопросе этим конкретным названием\n'
        '3. Сформулировать поисковый запрос на АНГЛИЙСКОМ с конкретным названием\n'
        'Пример: предыдущий ответ упоминает «Ляпис Трубецкой», пользователь спрашивает '
        '«кто похож на этого артиста» → ищи «artists similar to Lyapis Trubetskoy».\n'
        'Пример: предыдущий ответ упоминает «клуб Privè», пользователь спрашивает '
        '«что ещё в этом клубе» → ищи «Privè Tallinn upcoming events».'
    )

    if user_facts:
        system_prompt += f"\n\nТы помнишь про этого человека: {', '.join(user_facts[:5])}"
    if group_facts:
        system_prompt += f"\n\nТы помнишь про эту группу: {', '.join(group_facts[:5])}"
    if user_style:
        system_prompt += f"\n\n{user_style}"

    # Auto-append "Tallinn, Estonia" for place/event queries
    question_lower = question.lower()
    place_keywords = [
        "бар", "ресторан", "кафе", "клуб", "кино", "магазин", "музей", "театр", "галерея",
        "концерт", "мероприятие", "событие", "фестиваль", "выставка", "вечеринка", "шоу",
        "ивент", "event", "афиша", "тусовка", "движ",
        "сегодня", "завтра", "выходные", "вечером", "weekend",
        "куда", "где", "посоветуй", "порекомендуй", "подскажи", "сходить", "пойти",
    ]
    location_keywords = ["таллин", "tallinn", "эстони", "estonia"]

    if any(kw in question_lower for kw in place_keywords) and not any(
        loc in question_lower for loc in location_keywords
    ):
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
        client = http_client or httpx.AsyncClient(timeout=30.0)
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
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
        logger.error(f"Perplexity API error: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 429:
            return "Много запросов, подожди минутку)"
        return "Проблема с API, попробуй позже)"
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
