"""Main message handler — the core bot pipeline."""

import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from config import BOT_USERNAME
from bot.middleware.timing import Timer
from bot.utils.context import add_to_context, get_context_messages, evict_stale_data
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
    get_recent_chat_messages,
    redis_client,
)
from bot.services.style import get_style_summary
from bot.handlers.observer import record_bot_replied

logger = logging.getLogger(__name__)


# ── Routing ──────────────────────────────────────────────────────────

def should_respond(update: Update, bot_username: str, bot_id: int = None) -> bool:
    message = update.message
    if not message:
        return False

    content = get_message_content(message)
    if not content and not is_forwarded_message(message) and not has_photo(message):
        return False

    if message.chat.type == "private" and (content or has_photo(message)):
        return True

    if message.reply_to_message and message.reply_to_message.from_user:
        reply_from = message.reply_to_message.from_user
        # Match by username (case-insensitive) or by bot ID as fallback
        if reply_from.username and reply_from.username.lower() == bot_username.lower():
            return True
        if bot_id and reply_from.id == bot_id:
            logger.info(
                f"Reply matched by bot_id={bot_id} (username was "
                f"'{reply_from.username}' vs expected '{bot_username}')"
            )
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

    msg_content = get_message_content(message)

    # For messages the bot won't respond to: track in context and exit early.
    # For messages the bot WILL respond to: defer add_to_context until AFTER
    # get_context_messages() to avoid the current message appearing in the
    # conversation history sent to the API (which caused duplication).
    if not should_respond(update, BOT_USERNAME, bot_id=context.bot.id):
        if msg_content and update.effective_chat.type != "private":
            add_to_context(chat_id, "user", user_name or "user", msg_content)
        return

    timer.checkpoint("routing")

    # ── Extract question ─────────────────────────────────────────
    question = extract_question(get_message_content(message), BOT_USERNAME)
    referenced_content = None
    reply_msg = message.reply_to_message

    # Case 1: User replies to another message (reply to bot OR @mention in reply)
    if reply_msg:
        reply_content = get_message_content(reply_msg)
        logger.info(
            f"Reply detected: reply_to_message exists, "
            f"from_user={reply_msg.from_user.username if reply_msg.from_user else 'None'}, "
            f"has_text={bool(reply_msg.text)}, has_caption={bool(reply_msg.caption)}, "
            f"content_len={len(reply_content) if reply_content else 0}"
        )
        if reply_content:
            reply_author = get_display_name(reply_msg.from_user) if reply_msg.from_user else "unknown"
            if not reply_author:
                reply_author = reply_msg.from_user.username if reply_msg.from_user else "unknown"

            reply_urls = get_all_urls(reply_msg)

            # Determine if this is a reply to the bot's own message
            is_reply_to_bot = (
                reply_msg.from_user
                and reply_msg.from_user.username == BOT_USERNAME
            )

            if is_forwarded_message(reply_msg):
                referenced_content = f"[Forwarded post]: {reply_content}"
                if reply_urls:
                    referenced_content += f"\n[URLs in post]: {', '.join(reply_urls[:5])}"
            elif reply_urls:
                referenced_content = f"[Message with links]: {reply_content}"
                referenced_content += f"\n[URLs]: {', '.join(reply_urls[:5])}"
            elif is_reply_to_bot:
                # User is replying to the bot's own message — include full text
                # so the model can resolve pronouns ("этот артист", "там", etc.)
                referenced_content = (
                    f"[Предыдущий ответ бота, на который пользователь отвечает]:\n"
                    f"«{reply_content}»"
                )
                logger.info(
                    f"Reply-to-bot context captured ({len(reply_content)} chars): "
                    f"{reply_content[:150]}..."
                )
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
        # Still track in context so future replies have full history
        if msg_content and update.effective_chat.type != "private":
            add_to_context(chat_id, "user", user_name or "user", msg_content)
        await message.reply_text(
            f"Подожди {remaining} сек, не спеши)", reply_to_message_id=message.message_id,
        )
        return

    await send_typing(context.bot, chat_id)

    # ── Gather context + memory in parallel ──────────────────────
    # IMPORTANT: get context BEFORE adding the current message, so the
    # current question is not duplicated in the conversation history
    # that we send to the API.
    conv_context_msgs = get_context_messages(chat_id)

    # Redis fallback: after a restart, in-memory context is empty.
    # Load recent messages from Redis so the bot still has chat history.
    if not conv_context_msgs and update.effective_chat.type != "private":
        try:
            recent = await get_recent_chat_messages(chat_id, 15)
            if recent:
                # Redis stores newest-first; reverse to oldest-first
                for entry in reversed(recent):
                    conv_context_msgs.append({"role": "user", "content": entry})
                logger.info(
                    f"Loaded {len(conv_context_msgs)} messages from Redis "
                    f"(in-memory context was empty after restart)"
                )
        except Exception as e:
            logger.warning(f"Redis context fallback failed: {e}")

    # Now add the current user message to context (for future queries).
    if update.effective_chat.type != "private":
        if msg_content:
            add_to_context(chat_id, "user", user_name or "user", msg_content)
    else:
        add_to_context(chat_id, "user", user_name or "user", question)

    async def _empty_list():
        return []

    user_facts_coro = get_user_facts(user_id)
    group_facts_coro = get_group_facts(chat_id) if chat_id != user_id else _empty_list()

    user_facts, group_facts = await asyncio.gather(user_facts_coro, group_facts_coro)

    # Fetch per-user communication style (for tone adaptation)
    from bot.services import memory as mem_svc
    user_style = await get_style_summary(mem_svc.redis_client, user_id)

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
        f"Query from {user_id} ({user_name}): {question[:120]}... "
        f"[ref={'yes('+str(len(referenced_content))+'chars)' if referenced_content else 'no'}, "
        f"ctx_msgs={len(conv_context_msgs)}, photos={len(photo_urls)}]"
    )
    if referenced_content:
        logger.info(f"Referenced content preview: {referenced_content[:200]}...")

    answer = await query_perplexity(
        question=question,
        referenced_content=referenced_content,
        user_name=user_name,
        context_messages=conv_context_msgs,
        user_facts=user_facts,
        group_facts=group_facts,
        photo_urls=photo_urls if photo_urls else None,
        user_style=user_style,
    )

    timer.checkpoint("perplexity")

    # ── Post-processing ──────────────────────────────────────────
    set_rate_limit(user_id)
    # User message was already added to context before the API call.
    add_to_context(chat_id, "assistant", "bot", answer)
    await save_user_interaction(user_id, user_name, user.username)

    await message.reply_text(answer, reply_to_message_id=message.message_id)
    record_bot_replied(chat_id)

    timer.checkpoint("reply_sent")
    timer.done()

    # Fire-and-forget: background fact extraction
    # Build a flat string for fact extraction (doesn't need multi-turn)
    conv_context_str = "\n".join(
        f"{m['role']}: {m['content']}" for m in conv_context_msgs
    ) if conv_context_msgs else ""
    asyncio.create_task(_extract_and_save_facts(
        question=question, answer=answer, user_name=user_name,
        conv_context=conv_context_str, chat_id=chat_id, user_id=user_id,
    ))
