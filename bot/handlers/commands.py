"""Telegram command handlers: /start, /help, /remember, /forget, /memory, and more."""

import logging

from telegram import Update
from telegram.ext import ContextTypes

from bot.services.memory import (
    get_group_facts,
    get_user_facts,
    save_group_fact,
    save_user_fact,
)
from bot.utils.context import clear_context
from bot.utils.helpers import get_message_content
from config import USERNAME_TO_NAME

logger = logging.getLogger(__name__)


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
        "Поиск в интернете: просто спроси \"найди...\", \"погугли...\" и т.д.\n\n"
        "Примеры:\n"
        "- 'это правда?'\n"
        "- 'подробнее про это'\n"
        "- 'какой вариант лучше?'\n"
        "- 'что посоветуешь из меню?'\n\n"
        "Групповые фишки (командой или просто по-человечески, без /):\n"
        "/summary или /tldr - краткое саммари обсуждения (\"о чём тут говорили?\")\n"
        "/debate <тема> - режим дебатов на 30 мин (\"давай поспорим про удалёнку\")\n"
        "/factcheck - проверить факт (\"проверь, правда ли что...\")\n"
        "/poll Вопрос | Вариант 1 | Вариант 2 - создать опрос вручную\n"
        "/poll suggest - предложить опрос по обсуждению (\"сделай опрос\")\n\n"
        "Память:\n"
        "/memory - посмотреть что помню\n"
        "/remember <факт> - запомнить\n"
        "/forget - забыть всё\n"
        "/clear - очистить историю разговора"
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


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear — wipe in-memory conversation context (and debate mode) for this chat."""
    chat_id = update.effective_chat.id
    thread_id = update.message.message_thread_id
    clear_context(chat_id, thread_id)
    from bot.services.memory import clear_debate_mode
    await clear_debate_mode(chat_id, thread_id)
    await update.message.reply_text("Контекст разговора очищен. Начинаем с чистого листа)")


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


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /summary and /tldr — summarize recent buffered chat messages."""
    count = 30
    if context.args:
        try:
            count = max(5, min(int(context.args[0]), 100))
        except ValueError:
            pass

    from bot.handlers.actions import do_summary
    await do_summary(update, context, count=count)


async def debate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /debate <topic> — bot takes an adversarial stance for a limited time."""
    if not context.args:
        await update.message.reply_text(
            "Использование: /debate <тема>\nНапример: /debate удалёнка лучше офиса"
        )
        return

    topic = " ".join(context.args)
    from bot.handlers.actions import do_debate
    await do_debate(update, context, topic)


async def factcheck_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /factcheck — verify a claim via live web search (reply or inline text)."""
    message = update.message
    reply_msg = message.reply_to_message

    claim = " ".join(context.args) if context.args else None
    if not claim and reply_msg:
        claim = get_message_content(reply_msg)

    if not claim:
        await message.reply_text(
            "Использование: ответь на сообщение командой /factcheck, "
            "или напиши /factcheck <утверждение>"
        )
        return

    from bot.handlers.actions import do_factcheck
    await do_factcheck(update, context, claim)


async def poll_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /poll — manual poll, or /poll suggest for an LLM-proposed one."""
    message = update.message
    chat_id = update.effective_chat.id
    thread_id = message.message_thread_id
    raw = " ".join(context.args) if context.args else ""

    if raw.strip().lower() == "suggest":
        from bot.handlers.actions import do_poll_suggest
        await do_poll_suggest(update, context)
        return

    parts = [p.strip() for p in raw.split("|") if p.strip()]
    if len(parts) < 3:
        await message.reply_text(
            "Использование: /poll Вопрос | Вариант 1 | Вариант 2 [| Вариант 3 ...]\n"
            "Или: /poll suggest — предложу опрос по недавнему обсуждению"
        )
        return

    question, options = parts[0][:300], [o[:100] for o in parts[1:10]]
    await context.bot.send_poll(
        chat_id=chat_id,
        question=question,
        options=options,
        is_anonymous=False,
        message_thread_id=thread_id,
    )
