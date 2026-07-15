"""Voice/audio message handler.

Transcribes the message via Voxtral, then answers the transcript through the
same query_ai() pipeline used for typed questions. Only responds in private
chats or when the voice note is a reply to the bot — voice messages carry no
text, so there's no @mention to gate on, and transcribing every voice note in
a group by default would be noisy (and cost money) for messages never meant
for the bot.
"""

import asyncio
import logging

from telegram import ReplyParameters, Update
from telegram.ext import ContextTypes

from bot.handlers.messages import _extract_and_save_facts
from bot.handlers.observer import record_bot_replied
from bot.services.ai import query_ai
from bot.services.memory import (
    get_debate_topic,
    get_group_facts,
    get_user_facts,
    save_user_interaction,
)
from bot.services.style import get_style_summary
from bot.services.transcription import transcribe_audio
from bot.utils.context import add_to_context, get_context_messages, trim_context_for_api
from bot.utils.helpers import check_rate_limit, get_display_name, send_typing, set_rate_limit

logger = logging.getLogger(__name__)


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Transcribe a voice/audio message and answer it like a typed question."""
    message = update.message
    if not message:
        return

    voice = message.voice or message.audio
    if not voice:
        return

    is_private = update.effective_chat.type == "private"
    reply_to = message.reply_to_message
    is_reply_to_bot = bool(
        reply_to and reply_to.from_user and reply_to.from_user.id == context.bot.id
    )
    if not is_private and not is_reply_to_bot:
        return

    chat_id = update.effective_chat.id
    thread_id = message.message_thread_id
    user = update.effective_user
    user_id = user.id
    user_name = get_display_name(user)

    is_limited, remaining = check_rate_limit(user_id)
    if is_limited:
        await message.reply_text(
            f"Подожди {remaining} сек, не спеши)",
            reply_parameters=ReplyParameters(message_id=message.message_id),
        )
        return

    await send_typing(context.bot, chat_id)

    file = await context.bot.get_file(voice.file_id)
    audio_bytes = await file.download_as_bytearray()
    transcript = await transcribe_audio(bytes(audio_bytes), file_name=f"{voice.file_unique_id}.ogg")
    if not transcript:
        await message.reply_text(
            "Не получилось распознать голосовое(",
            reply_parameters=ReplyParameters(message_id=message.message_id),
        )
        return

    logger.info(f"Transcribed voice from {user_id} ({user_name}): {transcript[:120]}")

    conv_context_msgs = trim_context_for_api(get_context_messages(chat_id, thread_id))
    add_to_context(chat_id, "user", user_name or "user", transcript, thread_id=thread_id)

    async def _empty_list():
        return []

    user_facts, group_facts, debate_topic = await asyncio.gather(
        get_user_facts(user_id),
        get_group_facts(chat_id) if chat_id != user_id else _empty_list(),
        get_debate_topic(chat_id, thread_id),
    )
    from bot.services import memory as mem_svc
    user_style = await get_style_summary(mem_svc.redis_client, user_id)

    placeholder = await message.reply_text(
        "...", reply_parameters=ReplyParameters(message_id=message.message_id),
    )
    answer = await query_ai(
        question=transcript,
        user_name=user_name,
        context_messages=conv_context_msgs,
        user_facts=user_facts,
        group_facts=group_facts,
        user_style=user_style,
        telegram_bot=context.bot,
        telegram_chat_id=chat_id,
        telegram_message_id=placeholder.message_id,
        debate_topic=debate_topic,
    )

    set_rate_limit(user_id)
    add_to_context(chat_id, "assistant", "bot", answer, thread_id=thread_id)
    await save_user_interaction(user_id, user_name, user.username)
    record_bot_replied(chat_id)

    conv_context_str = "\n".join(
        f"{m['role']}: {m['content']}" for m in conv_context_msgs
    ) if conv_context_msgs else ""
    asyncio.create_task(_extract_and_save_facts(
        question=transcript, answer=answer, user_name=user_name,
        conv_context=conv_context_str, chat_id=chat_id, user_id=user_id,
    ))
