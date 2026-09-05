"""Live web search via Mistral's Conversations API (web_search connector).

Chat Completions (chat.complete/chat.stream — used everywhere else in this
codebase) does NOT support the web_search tool; only the Conversations/Agents
API does. This is a standalone, non-streaming call whose result is fed back
into the normal query_ai() pipeline as referenced_content, exactly like
weather data, so search results still go through the bot's persona/formatting
instead of being returned raw.
"""

import logging
import re

from mistralai.client.models.websearchtool import WebSearchTool

from bot.utils.helpers import current_tallinn_date_context, mask_quoted_spans
from config import MISTRAL_MODEL, SEARCH_TRIGGER_KEYWORDS

logger = logging.getLogger(__name__)


def is_search_trigger(text: str) -> bool:
    """Return True if the text explicitly asks the bot to search the web.

    Quoted spans are masked first (a quoted "найди" is an example being
    mentioned, not a command — e.g. an announcement listing usage examples),
    and matches require a leading word boundary so a keyword only fires as
    a real word/stem, not as a substring buried inside an unrelated word.
    """
    if not text:
        return False
    text_lower = mask_quoted_spans(text.lower())
    return any(re.search(r'\b' + re.escape(kw), text_lower) for kw in SEARCH_TRIGGER_KEYWORDS)


async def search_web(query: str) -> str | None:
    """Run a live web search and return a compact, cited answer string.

    Returns None on failure so callers can fall back to answering without it.
    """
    from bot.services import ai as ai_service
    client = ai_service.mistral_client
    if client is None or not query:
        return None

    try:
        await ai_service.throttle_call()
        response = await client.beta.conversations.start_async(
            inputs=query,
            model=MISTRAL_MODEL,
            tools=[WebSearchTool()],
            # Without this, the search agent gets no system prompt at all and
            # has no idea what "today" is — a claim like "Завтра солнце в
            # 6:47" was being checked with "завтра" left unanchored to any
            # actual date (confirmed live: the agent hedged with "у меня нет
            # актуальных данных" instead of resolving it).
            instructions=current_tallinn_date_context(),
        )
        await ai_service.record_call()

        answer_parts: list[str] = []
        sources: list[str] = []
        for entry in response.outputs:
            content = getattr(entry, "content", None)
            if content is None:
                continue
            if isinstance(content, str):
                answer_parts.append(content)
                continue
            for chunk in content:
                chunk_type = getattr(chunk, "type", None)
                if chunk_type == "text":
                    answer_parts.append(getattr(chunk, "text", ""))
                elif chunk_type == "tool_reference":
                    url = getattr(chunk, "url", None)
                    if url and url not in sources:
                        sources.append(url)

        answer = " ".join(p.strip() for p in answer_parts if p and p.strip())
        if not answer:
            return None

        result = f"[WEB SEARCH: {query}] {answer}"
        if sources:
            result += f"\nSources: {', '.join(sources[:5])}"
        return result
    except Exception as exc:
        logger.warning(f"Web search failed for '{query}': {exc}")
        return None
