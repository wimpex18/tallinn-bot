import os
from unittest.mock import AsyncMock

import pytest

# Set before any project module is imported so config.py doesn't see missing
# required env vars during test collection.
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("BOT_USERNAME", "test_bot")


@pytest.fixture(autouse=True)
def _no_real_throttle_sleep(monkeypatch):
    """Every Mistral call site now awaits ai.throttle_call() first, which can
    await asyncio.sleep() to stay under the free tier's ~1 req/s cap. Without
    this, the shared module-level throttle state would make the test suite
    actually sleep for real across the many tests that call it. Tests that
    check the throttle's own timing logic re-patch asyncio.sleep locally."""
    monkeypatch.setattr("bot.services.ai.asyncio.sleep", AsyncMock())
