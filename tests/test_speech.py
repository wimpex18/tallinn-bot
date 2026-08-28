import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.services import ai as ai_module
from bot.services import speech


def test_is_voice_reply_trigger_false_when_voice_id_unset(monkeypatch):
    monkeypatch.setattr(speech, "VOXTRAL_TTS_VOICE_ID", "")
    assert speech.is_voice_reply_trigger("ответь голосом, где бар?") is False


def test_is_voice_reply_trigger_matches_phrase(monkeypatch):
    monkeypatch.setattr(speech, "VOXTRAL_TTS_VOICE_ID", "voice-123")
    assert speech.is_voice_reply_trigger("ответь голосом, где бар?") is True


def test_is_voice_reply_trigger_no_match(monkeypatch):
    monkeypatch.setattr(speech, "VOXTRAL_TTS_VOICE_ID", "voice-123")
    assert speech.is_voice_reply_trigger("где хороший бар?") is False


def test_is_voice_reply_trigger_ignores_quoted_example(monkeypatch):
    monkeypatch.setattr(speech, "VOXTRAL_TTS_VOICE_ID", "voice-123")
    assert speech.is_voice_reply_trigger('можно написать "ответь голосом"') is False


def test_is_voice_reply_trigger_empty_text(monkeypatch):
    monkeypatch.setattr(speech, "VOXTRAL_TTS_VOICE_ID", "voice-123")
    assert speech.is_voice_reply_trigger("") is False
    assert speech.is_voice_reply_trigger(None) is False


@pytest.mark.asyncio
async def test_synthesize_speech_disabled_without_voice_id(monkeypatch):
    monkeypatch.setattr(speech, "VOXTRAL_TTS_VOICE_ID", "")
    client = MagicMock()
    monkeypatch.setattr(ai_module, "mistral_client", client)

    result = await speech.synthesize_speech("привет")

    assert result is None
    client.audio.speech.complete_async.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_speech_no_client(monkeypatch):
    monkeypatch.setattr(speech, "VOXTRAL_TTS_VOICE_ID", "voice-123")
    monkeypatch.setattr(ai_module, "mistral_client", None)

    assert await speech.synthesize_speech("привет") is None


@pytest.mark.asyncio
async def test_synthesize_speech_returns_decoded_audio(monkeypatch):
    monkeypatch.setattr(speech, "VOXTRAL_TTS_VOICE_ID", "voice-123")
    audio_bytes = b"fake-opus-audio"
    encoded = base64.b64encode(audio_bytes).decode()
    response = MagicMock(audio_data=encoded)
    client = MagicMock()
    client.audio.speech.complete_async = AsyncMock(return_value=response)
    monkeypatch.setattr(ai_module, "mistral_client", client)

    result = await speech.synthesize_speech("привет")

    assert result == audio_bytes
    kwargs = client.audio.speech.complete_async.call_args.kwargs
    assert kwargs["voice_id"] == "voice-123"
    assert kwargs["response_format"] == "opus"


@pytest.mark.asyncio
async def test_synthesize_speech_failure_returns_none(monkeypatch):
    monkeypatch.setattr(speech, "VOXTRAL_TTS_VOICE_ID", "voice-123")
    client = MagicMock()
    client.audio.speech.complete_async = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(ai_module, "mistral_client", client)

    assert await speech.synthesize_speech("привет") is None
