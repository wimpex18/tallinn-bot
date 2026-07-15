import os

# Set before any project module is imported so config.py doesn't see missing
# required env vars during test collection.
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("BOT_USERNAME", "test_bot")
