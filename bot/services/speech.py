"""Text-to-speech via Mistral's Voxtral TTS — experimental, opt-in voice replies.

Unlike the rest of this codebase, the request/response shapes here are built
from the official Mistral docs and SDK introspection, not a verified call
against a live API key (this development environment has no network access
to Mistral's API). Treat the first real use in production as the actual
test. Priced per minute (exact rate not published at the time this was
written) — this is opt-in and off by default via VOXTRAL_TTS_VOICE_ID being
unset, same pattern as the other cost-bearing feature (Voxtral STT).
"""

import base64
import logging
import re

from bot.utils.helpers import mask_quoted_spans
from config import VOICE_REPLY_TRIGGER_KEYWORDS, VOXTRAL_TTS_MODEL, VOXTRAL_TTS_VOICE_ID

logger = logging.getLogger(__name__)

_MAX_TTS_CHARS = 500  # keep synthesized replies short — this is a novelty, not the primary UX


def is_voice_reply_trigger(text: str) -> bool:
    """Return True if the user explicitly asked for a spoken reply."""
    if not text or not VOXTRAL_TTS_VOICE_ID:
        return False
    text_lower = mask_quoted_spans(text.lower())
    return any(re.search(r'\b' + re.escape(kw), text_lower) for kw in VOICE_REPLY_TRIGGER_KEYWORDS)


async def synthesize_speech(text: str) -> bytes | None:
    """Synthesize `text` to Opus-encoded audio bytes, or None on failure/disabled."""
    if not VOXTRAL_TTS_VOICE_ID or not text:
        return None

    from bot.services import ai as ai_service
    client = ai_service.mistral_client
    if client is None:
        return None

    try:
        await ai_service.throttle_call()
        response = await client.audio.speech.complete_async(
            model=VOXTRAL_TTS_MODEL,
            input=text[:_MAX_TTS_CHARS],
            voice_id=VOXTRAL_TTS_VOICE_ID,
            response_format="opus",
        )
        await ai_service.record_call()
        return base64.b64decode(response.audio_data)
    except Exception as exc:
        logger.warning(f"Speech synthesis failed, falling back to text-only: {exc}")
        return None
