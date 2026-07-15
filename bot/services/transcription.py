"""Voice/audio message transcription via Mistral's Voxtral speech-to-text.

The one feature in this codebase with non-zero (if trivial, ~$0.001/min)
marginal cost — everything else stays on Mistral's free tier. See README.
"""

import logging

from mistralai.client.models.file import File

from config import VOXTRAL_MODEL

logger = logging.getLogger(__name__)


async def transcribe_audio(audio_bytes: bytes, file_name: str = "voice.ogg") -> str | None:
    """Transcribe a voice/audio message. Returns the transcript text, or None on failure."""
    from bot.services import ai as ai_service
    client = ai_service.mistral_client
    if client is None or not audio_bytes:
        return None

    try:
        response = await client.audio.transcriptions.complete_async(
            model=VOXTRAL_MODEL,
            file=File(file_name=file_name, content=audio_bytes),
        )
        await ai_service.record_call()
        text = (response.text or "").strip()
        return text or None
    except Exception as exc:
        logger.warning(f"Voice transcription failed: {exc}")
        return None
