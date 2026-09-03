"""Shared action execution for /summary, /debate, /factcheck, /poll suggest, and /quiz.

Used both by the literal slash commands (bot/handlers/commands.py, which parse
context.args/reply_to_message and delegate here) and by natural-language
triggers (bot/handlers/messages.py, via bot/services/intent.py) — the logic
exists exactly once regardless of how it was invoked.
"""

import logging

from telegram import ReplyParameters, Update
from telegram.ext import ContextTypes

from bot.utils.context import get_context_string
from bot.utils.helpers import send_typing
from config import DEBATE_MODE_TTL, FACTCHECK_CONTEXT_MESSAGES

logger = logging.getLogger(__name__)


async def do_summary(update: Update, context: ContextTypes.DEFAULT_TYPE, count: int = 30) -> None:
    """Summarize the last `count` buffered messages in this chat/thread."""
    chat_id = update.effective_chat.id
    thread_id = update.message.message_thread_id

    from bot.services.ai import summarize_conversation
    from bot.services.memory import get_recent_chat_messages

    messages = await get_recent_chat_messages(chat_id, count, thread_id=thread_id)
    if not messages:
        await update.message.reply_text("Пока нечего суммировать — в чате было тихо.")
        return

    await send_typing(context.bot, chat_id)
    summary = await summarize_conversation(messages)
    await update.message.reply_text(summary)


async def do_debate(update: Update, context: ContextTypes.DEFAULT_TYPE, topic: str) -> None:
    """Activate debate mode on `topic` for this chat/thread."""
    chat_id = update.effective_chat.id
    thread_id = update.message.message_thread_id
    topic = topic[:300]

    from bot.services.memory import set_debate_mode
    await set_debate_mode(chat_id, topic, thread_id=thread_id, ttl=DEBATE_MODE_TTL)

    minutes = DEBATE_MODE_TTL // 60
    await update.message.reply_text(
        f"Режим дебатов включён на {minutes} мин. Тема: «{topic}». "
        f"Обращайся ко мне — буду топить за другую сторону) /clear чтобы выключить раньше."
    )


async def do_factcheck(update: Update, context: ContextTypes.DEFAULT_TYPE, claim: str) -> None:
    """Verify `claim` via live web search and reply with a verdict + sources.

    When triggered as a reply, a claim discussed across several messages (not
    just the one replied to) can lose its full picture if only that one
    message is checked — so the last few chat messages are pulled in as
    extra context for both the search and the verdict.
    """
    message = update.message
    chat_id = update.effective_chat.id
    await send_typing(context.bot, chat_id)

    from bot.services.ai import query_ai
    from bot.services.memory import get_recent_chat_messages
    from bot.services.search import search_web

    surrounding = ""
    if message.reply_to_message:
        thread_id = message.message_thread_id
        recent = await get_recent_chat_messages(chat_id, FACTCHECK_CONTEXT_MESSAGES, thread_id=thread_id)
        if recent:
            surrounding = "\n".join(reversed(recent))

    search_query = f"Проверь факт: {claim}"
    question = f"Проверь это утверждение на достоверность и дай короткий вердикт: {claim}"
    if surrounding:
        context_block = f"Контекст обсуждения в чате перед этим сообщением:\n{surrounding}"
        search_query = f"{search_query}\n\n{context_block}"
        question = f"{question}\n\n{context_block}"

    search_result = await search_web(search_query)
    if not search_result:
        await message.reply_text("Не получилось проверить — попробуй позже(")
        return

    answer = await query_ai(
        question=question,
        referenced_content=search_result,
        reasoning_effort="high",
    )
    await message.reply_text(answer, reply_parameters=ReplyParameters(message_id=message.message_id))


async def do_poll_suggest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Propose and send a poll based on the recent conversation."""
    message = update.message
    chat_id = update.effective_chat.id
    thread_id = message.message_thread_id

    from bot.services.ai import suggest_poll

    conv_context = get_context_string(chat_id, thread_id)
    if not conv_context:
        await message.reply_text("Пока не с чем работать — обсудите что-нибудь, и я предложу опрос)")
        return

    await send_typing(context.bot, chat_id)
    suggestion = await suggest_poll(conv_context)
    if not suggestion:
        await message.reply_text("Не придумал опрос из последнего обсуждения(")
        return

    await context.bot.send_poll(
        chat_id=chat_id,
        question=suggestion["question"],
        options=suggestion["options"],
        is_anonymous=False,
        allows_revoting=True,
        message_thread_id=thread_id,
    )


async def do_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE, topic: str = None) -> None:
    """Generate and send a native Telegram quiz question, optionally on `topic`."""
    message = update.message
    chat_id = update.effective_chat.id
    thread_id = message.message_thread_id

    from bot.services.ai import suggest_quiz

    await send_typing(context.bot, chat_id)
    quiz = await suggest_quiz(topic)
    if not quiz:
        await message.reply_text("Не придумал вопрос для викторины, попробуй ещё раз)")
        return

    await context.bot.send_poll(
        chat_id=chat_id,
        question=quiz["question"],
        options=quiz["options"],
        type="quiz",
        correct_option_id=quiz["correct_option_id"],
        is_anonymous=False,
        message_thread_id=thread_id,
    )
