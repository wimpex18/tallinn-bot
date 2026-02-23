"""Telegram command handlers: /start, /help, /remember, /forget, /memory."""

import logging

from telegram import Update
from telegram.ext import ContextTypes

from config import USERNAME_TO_NAME

logger = logging.getLogger(__name__)
from bot.services.memory import (
    save_user_fact, get_user_facts,
    save_group_fact, get_group_facts,
    redis_client,
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    username = update.effective_user.username
    user_name = USERNAME_TO_NAME.get(username, username)

    if not context.args:
        await update.message.reply_text(
            "Использование: /remember <факт>\nНапример: /remember люблю IPA"
        )
        return

    fact = " ".join(context.args)
    if len(fact) > 500:
        await update.message.reply_text("Слишком длинно, напиши покороче (до 500 символов)")
        return
    if user_name:
        fact = f"{user_name}: {fact}"

    if update.effective_chat.type == "private":
        await save_user_fact(user_id, fact)
    else:
        await save_group_fact(chat_id, fact)

    await update.message.reply_text("Запомнил)")


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if update.effective_chat.type != "private":
        member = await context.bot.get_chat_member(chat_id, user_id)
        if member.status not in ["creator", "administrator"]:
            await update.message.reply_text("Только админ может это делать)")
            return

    from bot.services import memory
    if memory.redis_client:
        try:
            if update.effective_chat.type == "private":
                await memory.redis_client.delete(f"user:{user_id}:facts")
            else:
                await memory.redis_client.delete(f"group:{chat_id}:facts")
            await update.message.reply_text("Забыл всё)")
        except Exception as e:
            logger.error(f"Failed to forget: {e}")
            await update.message.reply_text("Не получилось забыть(")
    else:
        await update.message.reply_text("Память не подключена(")


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    username = update.effective_user.username
    user_name = USERNAME_TO_NAME.get(username, username) if username else "Ты"

    from bot.services import memory
    if not memory.redis_client:
        await update.message.reply_text("Память не подключена(")
        return

    if update.effective_chat.type == "private":
        facts = await get_user_facts(user_id)
        if facts:
            facts_text = "\n".join([f"- {fact}" for fact in facts])
            await update.message.reply_text(f"Что я помню про тебя:\n\n{facts_text}")
        else:
            await update.message.reply_text("Пока ничего не помню про тебя")
    else:
        user_facts = await get_user_facts(user_id)
        group_facts = await get_group_facts(chat_id)

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


async def cleanup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cleanup — remove stale Redis keys (admin only)."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Only allow in private chat or by group admin
    if update.effective_chat.type != "private":
        member = await context.bot.get_chat_member(chat_id, user_id)
        if member.status not in ["creator", "administrator"]:
            await update.message.reply_text("Только админ может это делать)")
            return

    from bot.services.memory import cleanup_stale_redis_keys
    await update.message.reply_text("Чищу старые данные...")
    stats = await cleanup_stale_redis_keys(max_age_days=90)
    await update.message.reply_text(
        f"Готово! Просканировано: {stats.get('scanned', 0)}, "
        f"удалено: {stats.get('deleted', 0)}"
    )


async def quiet_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /quiet — toggle proactive/spontaneous messages in this chat."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if update.effective_chat.type == "private":
        await update.message.reply_text("Эта команда для групповых чатов)")
        return

    member = await context.bot.get_chat_member(chat_id, user_id)
    if member.status not in ["creator", "administrator"]:
        await update.message.reply_text("Только админ может это делать)")
        return

    from bot.services.memory import is_quiet_mode, set_quiet_mode
    currently_quiet = await is_quiet_mode(chat_id)
    await set_quiet_mode(chat_id, not currently_quiet)

    if currently_quiet:
        await update.message.reply_text("Включил спонтанные сообщения)")
    else:
        await update.message.reply_text("Выключил спонтанные сообщения. /quiet чтобы вернуть)")
