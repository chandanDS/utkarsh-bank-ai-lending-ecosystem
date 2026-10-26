"""Multilingual voice helpers for the Analyze conversational BI experience."""

from __future__ import annotations

import json
from typing import Any


SUPPORTED_LANGUAGES = {
    "English": {"code": "en", "locale": "en-IN"},
    "Hindi": {"code": "hi", "locale": "hi-IN"},
    "Bengali": {"code": "bn", "locale": "bn-IN"},
    "Kannada": {"code": "kn", "locale": "kn-IN"},
}

TRANSCRIPTION_MODEL = "whisper-large-v3"


def transcribe_voice_note(audio_bytes: bytes, api_key: str, language: str | None = None) -> tuple[str, str]:
    """Return the transcript and detected ISO language code for a recorded question."""
    from groq import Groq

    request: dict[str, Any] = {
        "file": ("analyze-question.wav", audio_bytes),
        "model": TRANSCRIPTION_MODEL,
        "response_format": "verbose_json",
        "temperature": 0,
    }
    if language:
        request["language"] = language
    result = Groq(api_key=api_key).audio.transcriptions.create(**request)
    return (result.text or "").strip(), (getattr(result, "language", "") or "").lower()


def _chat_text(client: Any, model: str, system: str, text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
        temperature=0,
        max_completion_tokens=400,
        reasoning_effort="low",
    )
    return (response.choices[0].message.content or "").strip()


def translate_question_to_english(text: str, api_key: str, model: str, known_language: str | None = None) -> tuple[str, str]:
    """Translate a BI question to English and identify its language for the spoken reply."""
    if known_language == "en":
        return text.strip(), "en"
    if not api_key and known_language is None:
        # Preserve the demo's existing English path when optional Groq services are not configured.
        return text.strip(), "en"
    from groq import Groq

    language_names = ", ".join(SUPPORTED_LANGUAGES)
    system = (
        "You prepare multilingual banking analytics questions for a governed SQL system. "
        "Detect the user's language and translate the question faithfully into English. "
        f"Recognize these supported output languages: {language_names}. "
        "Do not answer the question or add filters, metrics, dates, or assumptions. "
        "Return only JSON with keys language (one of English, Hindi, Bengali, Kannada, or Other) "
        "and english_question (the faithful English translation)."
    )
    raw = _chat_text(Groq(api_key=api_key), model, system, text)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].removesuffix("```").strip()
        parsed = json.loads(cleaned)
        language = str(parsed.get("language", "Other"))
        translated = str(parsed.get("english_question", "")).strip()
        code = SUPPORTED_LANGUAGES.get(language, {}).get("code", known_language or "en")
        return translated or text.strip(), code
    except (json.JSONDecodeError, AttributeError, TypeError):
        return text.strip(), known_language or "en"


def translate_answer(text: str, api_key: str, model: str, language_code: str) -> str:
    """Translate an English BI answer to one of the supported spoken languages."""
    language_name = next(
        (name for name, values in SUPPORTED_LANGUAGES.items() if values["code"] == language_code),
        "English",
    )
    if language_name == "English":
        return text
    from groq import Groq

    return _chat_text(
        Groq(api_key=api_key),
        model,
        f"Translate the banking analytics response into natural {language_name}. "
        "Preserve all numbers, units, product names, branch names and caveats. "
        "Do not add facts or commentary. Return only the translation.",
        text,
    )
