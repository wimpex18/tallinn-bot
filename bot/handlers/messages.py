"""Main message handler — the core bot pipeline."""

import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from config import BOT_USERNAME
from bot.middleware.timing import Timer
from bot.utils.context import add_to_context, get_context_string, evict_stale_data
from bot.utils.helpers import (
    get_message_content, get_all_urls, extract_urls, extract_question,
    is_forwarded_message, has_photo, download_photo_as_base64,
    send_typing, check_rate_limit, set_rate_limit, get_display_name,
)
from bot.services.url_fetcher import fetch_url_content
from bot.services.perplexity import query_perplexity
from bot.services.memory import (
    get_user_facts, get_group_facts,
    save_user_fact, save_group_fact, save_user_interaction,
    smart_extract_facts, extract_facts_from_response,
)

logger = logging.getLogger(__name__)


# ── Routing ──────────────────────────────────────────────────────────

def should_respond(update: Update, bot_username: str) -> bool:
    message = update.message
    if not message:
        return False

    content = get_message_content(message)
    if not content and not is_forwarded_message(message) and not has_photo(message):
        return False

    if message.chat.type == "private" and (content or has_photo(message)):
        return True

    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.username == bot_username:
            return True

    if content and f"@{bot_username}" in content:
        return True

    return False


# ── Background fact extraction ───────────────────────────────────────

async def _extract_and_save_facts(
    question: str, answer: str, user_name: str,
    conv_context: str, chat_id: int, user_id: int,
) -> None:
    try:
        facts = await smart_extract_facts(
            question=question, answer=answer,
            user_name=user_name, chat_context=conv_context,
        )
        if not facts:
            facts = extract_facts_from_response(question, answer, user_name)

        for fact in facts:
            if chat_id == user_id:
                await save_user_fact(user_id, fact)
            else:
                await save_group_fact(chat_id, fact)

        if facts:
            logger.info(f"Learned facts: {facts}")
    except Exception as e:
        logger.error(f"Background fact extraction failed: {e}")


# ── Main handler ─────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages with full timing instrumentation."""
    message = update.message
    if not message:
        return

    timer = Timer(update)

    # Periodic cleanup
    evict_stale_data()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user = update.effective_user
    user_name = get_display_name(user)

    # Track context for all group messages (even if not responding)
    msg_content = get_message_content(message)
    if msg_content and update.effective_chat.type != "private":
        add_to_context(chat_id, "user", user_name or "user", msg_content)

    if not should_respond(update, BOT_USERNAME):
        return

    timer.checkpoint("routing")

    # ── Extract question ─────────────────────────────────────────
    question = extract_question(get_message_content(message), BOT_USERNAME)
    referenced_content = None
    reply_msg = message.reply_to_message

    # Case 1: User replies to another message
    msg_text = get_message_content(message)
    if reply_msg and msg_text and f"@{BOT_USERNAME}" in msg_text:
        reply_content = get_message_content(reply_msg)
        if reply_content:
            reply_author = get_display_name(reply_msg.from_user) if reply_msg.from_user else "unknown"
            if not reply_author:
                reply_author = reply_msg.from_user.username if reply_msg.from_user else "unknown"

            reply_urls = get_all_urls(reply_msg)

            if is_forwarded_message(reply_msg):
                referenced_content = f"[Forwarded post]: {reply_content}"
                if reply_urls:
                    referenced_content += f"\n[URLs in post]: {', '.join(reply_urls[:5])}"
            elif reply_urls:
                referenced_content = f"[Message with links]: {reply_content}"
                referenced_content += f"\n[URLs]: {', '.join(reply_urls[:5])}"
            else:
                referenced_content = f"[Message from {reply_author}]: {reply_content}"

    # Case 2: Current message is forwarded
    if is_forwarded_message(message) and not referenced_content:
        content = get_message_content(message)
        if content:
            msg_urls = get_all_urls(message)
            referenced_content = f"[Forwarded post]: {content}"
            if msg_urls:
                referenced_content += f"\n[URLs in post]: {', '.join(msg_urls[:5])}"
            if not question:
                question = "расскажи об этом"

    # Case 3: Current message has URLs (no reply)
    if not referenced_content and question:
        urls = get_all_urls(message) or extract_urls(question)
        if urls:
            referenced_content = f"[Shared link]: {urls[0]}"

    timer.checkpoint("parse")

    # ── Fetch URL content ────────────────────────────────────────
    urls_to_fetch = []
    if reply_msg:
        urls_to_fetch = get_all_urls(reply_msg)
    if not urls_to_fetch:
        urls_to_fetch = get_all_urls(message) or extract_urls(question or "")

    if urls_to_fetch and referenced_content:
        first_url = urls_to_fetch[0]
        logger.info(f"Fetching URL content: {first_url}")
        url_content = await fetch_url_content(first_url)
        if url_content and len(url_content) > 100:
            referenced_content += f"\n\n[Article content]:\n{url_content}"

    timer.checkpoint("url_fetch")

    # ── Photo handling ───────────────────────────────────────────
    has_current_photo = has_photo(message)
    has_reply_photo = reply_msg and has_photo(reply_msg)

    if not question and not referenced_content and not has_current_photo and not has_reply_photo:
        await message.reply_text("Чё спросить хотел?", reply_to_message_id=message.message_id)
        return

    if not question and referenced_content:
        question = "о чём это?"
    if not question and (has_current_photo or has_reply_photo):
        question = "что на фото?"

    # Rate limit (checked after we know we will process)
    is_limited, remaining = check_rate_limit(user_id)
    if is_limited:
        await message.reply_text(
            f"Подожди {remaining} сек, не спеши)", reply_to_message_id=message.message_id,
        )
        return

    await send_typing(context.bot, chat_id)

    # ── Gather context + memory in parallel ──────────────────────
    conv_context = get_context_string(chat_id)

    async def _empty_list():
        return []

    user_facts_coro = get_user_facts(user_id)
    group_facts_coro = get_group_facts(chat_id) if chat_id != user_id else _empty_list()

    user_facts, group_facts = await asyncio.gather(user_facts_coro, group_facts_coro)

    timer.checkpoint("memory")

    # ── Photos ───────────────────────────────────────────────────
    photo_urls = []
    if has_photo(message):
        photo = message.photo[-1]
        photo_url = await download_photo_as_base64(photo, context.bot)
        if photo_url:
            photo_urls.append(photo_url)

    if reply_msg and has_photo(reply_msg):
        photo = reply_msg.photo[-1]
        photo_url = await download_photo_as_base64(photo, context.bot)
        if photo_url:
            photo_urls.append(photo_url)

    timer.checkpoint("photos")

    # ── Query Perplexity ─────────────────────────────────────────
    logger.info(
        f"Query from {user_id} ({user_name}): {question[:80]}... "
        f"[ref={referenced_content is not None}, photos={len(photo_urls)}]"
    )

    answer = await query_perplexity(
        question=question,
        referenced_content=referenced_content,
        user_name=user_name,
        context=conv_context,
        user_facts=user_facts,
        group_facts=group_facts,
        photo_urls=photo_urls if photo_urls else None,
    )

    timer.checkpoint("perplexity")

    # ── Post-processing ──────────────────────────────────────────
    set_rate_limit(user_id)
    add_to_context(chat_id, "user", user_name or "user", question)
    add_to_context(chat_id, "assistant", "bot", answer)
    await save_user_interaction(user_id, user_name, user.username)

    await message.reply_text(answer, reply_to_message_id=message.message_id)

    timer.checkpoint("reply_sent")
    timer.done()

    # Fire-and-forget: background fact extraction
    asyncio.create_task(_extract_and_save_facts(
        question=question, answer=answer, user_name=user_name,
        conv_context=conv_context, chat_id=chat_id, user_id=user_id,
    ))
