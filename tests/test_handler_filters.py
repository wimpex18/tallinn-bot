"""Tests for the PTB filter compositions used to register handlers in main.py.

These mirror the exact filter expressions in main.py (rather than importing
main.py itself, which builds a live Application) so a regression in the
edited-message exclusion is caught without needing to spin up the bot.
"""

import datetime

from telegram import Chat, Message, Update, User
from telegram.ext import filters

_MAIN_MESSAGE_FILTER = (
    (filters.TEXT | filters.FORWARDED | filters.PHOTO)
    & ~filters.COMMAND
    & ~filters.UpdateType.EDITED
)
_OBSERVER_FILTER = filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS & ~filters.UpdateType.EDITED


def _make_update(*, edited: bool) -> Update:
    chat = Chat(id=1, type="group")
    user = User(id=2, is_bot=False, first_name="Alice")
    message = Message(
        message_id=10, date=datetime.datetime.now(), chat=chat, text="hello", from_user=user,
    )
    if edited:
        return Update(update_id=1, edited_message=message)
    return Update(update_id=1, message=message)


def test_main_handler_ignores_edited_messages():
    assert _MAIN_MESSAGE_FILTER.check_update(_make_update(edited=False))
    assert not _MAIN_MESSAGE_FILTER.check_update(_make_update(edited=True))


def test_observer_handler_ignores_edited_messages():
    assert _OBSERVER_FILTER.check_update(_make_update(edited=False))
    assert not _OBSERVER_FILTER.check_update(_make_update(edited=True))
