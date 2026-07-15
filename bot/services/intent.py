"""Hybrid natural-language intent routing for /summary, /debate, /factcheck, /poll.

Three tiers, in order — see README's "Architecture notes" for the rationale:

1. Strong phrase match (free, regex/substring only) — a specific phrase like
   "о чём говорили" or "сделай опрос" resolves immediately, no LLM call.
2. No strong match and no loose signal word either (the common case — plain
   chat) — returns None immediately, same cost as today.
3. No strong match, but a loose "this might be about summarizing/debating/
   fact-checking/polling" stem is present — one extra Mistral call
   (`ai.classify_intent`) disambiguates and can also fill in a topic/claim
   that a strong match found the trigger for but not the parameter.
"""

import re
from dataclasses import dataclass

from config import (
    INTENT_STRONG_DEBATE,
    INTENT_STRONG_FACTCHECK,
    INTENT_STRONG_POLL,
    INTENT_STRONG_SUMMARY,
    INTENT_WEAK_DEBATE,
    INTENT_WEAK_FACTCHECK,
    INTENT_WEAK_POLL,
    INTENT_WEAK_SUMMARY,
)

_TOPIC_CONNECTORS = ("про ", "о ", "на тему ")

# A phrase inside quotes is being mentioned/quoted, not issued as a command —
# e.g. an announcement listing example phrases like «сделай саммари» shouldn't
# itself trigger a summary. Blank quoted spans out before phrase-matching
# (same length, so index math in _extract_remainder stays valid).
_QUOTE_SPAN_RE = re.compile(r'«[^»]+»|"[^"]+"|“[^”]+”|„[^“]+“')


def _mask_quoted_spans(text: str) -> str:
    return _QUOTE_SPAN_RE.sub(lambda m: " " * len(m.group(0)), text)


@dataclass
class Intent:
    action: str
    topic: str | None = None
    claim: str | None = None


def _matches_any(text_lower: str, phrases: list[str]) -> bool:
    return any(re.search(r'\b' + re.escape(p), text_lower) for p in phrases)


def _first_match(text_lower: str, phrases: list[str]) -> str | None:
    """Return the longest phrase from `phrases` found at a word boundary in
    text_lower, or None.

    A leading \\b (not a trailing one) is intentional: several entries are
    stems meant to match any inflected suffix — "опрос" should match "опросе"/
    "опросы" — but a plain substring check also matched "опрос" inside totally
    unrelated words like "попросить" or "вопрос", since it happens to appear
    mid-word there. Requiring a word boundary immediately before the phrase
    keeps the stem-matching while ruling out those accidental substrings.
    """
    for p in sorted(phrases, key=len, reverse=True):
        if re.search(r'\b' + re.escape(p), text_lower):
            return p
    return None


def _extract_remainder(text_lower: str, original_text: str, trigger: str) -> str:
    """Whatever comes after `trigger` in the original-cased text, with a
    leading connector word ("про"/"о"/"на тему") stripped."""
    idx = text_lower.find(trigger)
    if idx == -1:
        return ""
    remainder = original_text[idx + len(trigger):].strip(" ,:—-")
    remainder_lower = remainder.lower()
    for conn in _TOPIC_CONNECTORS:
        if remainder_lower.startswith(conn):
            remainder = remainder[len(conn):].strip()
            break
    return remainder


async def detect_action_intent(question: str, conv_context: str = None) -> Intent | None:
    """Decide whether `question` is a request to run one of the four actions.

    Returns None for ordinary chat (the overwhelming majority of messages,
    resolved for free via tiers 1-2) or an `Intent` naming the action —
    `topic`/`claim` may still be None even for a resolved debate/factcheck
    intent if neither the phrase nor tier 3 could pin one down; callers
    should fall back to normal conversation in that case rather than guess.
    """
    if not question:
        return None
    text_lower = _mask_quoted_spans(question.lower())

    # ── Tier 1: strong phrase matches (free) ──────────────────────────
    if _matches_any(text_lower, INTENT_STRONG_SUMMARY):
        return Intent("summary")

    if _matches_any(text_lower, INTENT_STRONG_POLL):
        return Intent("poll")

    debate_trigger = _first_match(text_lower, INTENT_STRONG_DEBATE)
    if debate_trigger:
        topic = _extract_remainder(text_lower, question, debate_trigger)
        if topic:
            return Intent("debate", topic=topic)
        # Trigger phrase present but no topic in the sentence ("давай поспорим") —
        # fall through to tier 3, which can try to infer a topic from context
        # instead of giving up outright.

    factcheck_trigger = _first_match(text_lower, INTENT_STRONG_FACTCHECK)
    if factcheck_trigger:
        claim = _extract_remainder(text_lower, question, factcheck_trigger)
        if claim:
            return Intent("factcheck", claim=claim)
        # Bare "фактчек"/"проверь факт" with nothing after it — likely meant
        # as a reply-based factcheck; the caller falls back to reply content,
        # or (if it's not a reply either) tier 3 below gets a shot at it.

    # ── Tier 2: no strong match — only tier 3 if there's a loose signal ───
    has_weak_signal = (
        _matches_any(text_lower, INTENT_WEAK_SUMMARY)
        or _matches_any(text_lower, INTENT_WEAK_DEBATE)
        or _matches_any(text_lower, INTENT_WEAK_FACTCHECK)
        or _matches_any(text_lower, INTENT_WEAK_POLL)
        or debate_trigger is not None
        or factcheck_trigger is not None
    )
    if not has_weak_signal:
        return None

    # ── Tier 3: ask the model to disambiguate (~1 extra Mistral call) ────
    from bot.services.ai import classify_intent

    result = await classify_intent(question, conv_context)
    if not result:
        return None
    return Intent(result["action"], topic=result.get("topic"), claim=result.get("claim"))
