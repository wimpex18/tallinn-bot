# Tallinn Bot

A Telegram bot for a private friends' group centered on life in Tallinn, Estonia — Q&A, link/photo
analysis, weather, debate facilitation, and persistent memory of the group's running conversation.

## How it works

**Stack:** [python-telegram-bot](https://docs.python-telegram-bot.org/) 22.8 (async, Bot API 10.0),
Python 3.14, [Mistral AI](https://mistral.ai/) (`mistral-small-latest`, currently Mistral Small 4 —
vision-capable, 256k context), Redis for persistent memory, deployed on [Render](https://render.com/)
(webhook in production, long polling locally).

### Persona & tone

The system prompt (`bot/services/ai.py::_STATIC_SYSTEM`) defines a specific character, not a generic
assistant: short, informal replies in Russian, witty and warm rather than encyclopedic — a sharp
friend in the chat, not a customer-service bot. Two rules sit above everything else and can't be
overridden by per-user style adaptation:
- **No hostility** — no anger, contempt, or insults toward anyone in the chat.
- **No bravado** — no tough-guy posturing, no mock-threats (physical or sexual) even as jokes.

Per-user tone adaptation (`bot/services/style.py`) tracks each person's own casual register — slang,
mild profanity, message length — and nudges the bot to match it, but explicitly scoped to *manner of
speech*, never as license for the two rules above. As a second layer, every finished reply also gets
checked by Mistral's Moderation API (`mistral-moderation-2603`) before being sent, scoped to the
categories behind those two rules (`hate_and_discrimination`, `violence_and_threats`, `sexual`,
`self_harm`) — not a blanket filter, so normal group-chat banter isn't flagged. A flagged reply gets
replaced with a short, honest fallback instead. The check fails open: a moderation-API error lets the
original reply through rather than going silent.

### Memory

Redis holds long-lived state: per-user facts (global to the person, not scoped to a chat — the same
`user:{id}:facts` bucket whether they're in a DM or any group), per-group facts, per-user style
profiles, a saved-quotes book per group (`/quote`/`/quotes`), and quiet-mode/debate-mode flags. Every
write refreshes a 90-day TTL on its own key, so Redis expires stale data on its own — no cleanup job.
An in-memory, per-process cache holds the active conversation window for low-latency replies; Redis is
the fallback that survives restarts.

### Handler groups

- **Group 0** (default): commands and the main message handler, which only reacts to @-mentions,
  replies to the bot, plain-text mentions of the bot's name (`BOT_DISPLAY_NAMES` in `config.py` —
  "Sam"/"Сэм" by default), or private-chat messages.
- **Group 1** (silent observer): runs on every group message regardless of whether the bot was
  addressed — stores it into Redis, updates the sender's style profile, and at a small default
  probability reacts with an emoji. A spontaneous-text-reply path exists but is disabled by default
  (too noisy for this group).

### Scheduled jobs

- **Proactive memory** (~every 8h) and **style refresh** (daily) run quietly in the background.
- **Daily icebreaker prompt** exists but is **disabled by default** — this bot only speaks when
  spoken to (or reacts with an emoji).

### Group chat reliability

- Edited messages don't re-trigger a reply (Telegram routes edits through the same handlers as new
  messages by default; explicitly filtered out).
- An unqualified follow-up ("а это точно так?") only continues the bot's *previous* reply if it's
  from the same person that reply was addressed to — the conversation window is shared across
  everyone in a chat, so this prevents cross-talk in a busy group.
- Content from web pages, search results, and forwarded posts is explicitly framed as untrusted
  reference material, not instructions — a defense against prompt injection via a forwarded link.

## Commands

The bot pushes this list to Telegram's `/` command menu itself on every startup
(`set_my_commands` in `main.py`), so a new command shows up in clients automatically
after deploy — no manual BotFather edit needed.

| Command | Description | Also works by just asking |
|---|---|---|
| `/start`, `/help` | Intro and usage guide | — |
| `/summary`, `/tldr [N]` | Summarize the last N (default 30) buffered messages | "о чём тут говорили?" |
| `/debate <topic>` | Bot takes an adversarial stance on `<topic>` for 30 minutes | "давай поспорим про X" |
| `/factcheck` | Verify a claim (reply to a message, or `/factcheck <claim>`) via live web search | "проверь факт: X" |
| `/poll Q \| A \| B \| C` | Send a native, revotable Telegram poll (manual only) | — |
| `/poll suggest` | Propose a poll from the recent discussion | "сделай опрос" |
| `/quiz [topic]` | Native Telegram quiz question with a marked correct answer | "устрой викторину про Таллинн" |
| `/quote` | Reply to a message with this to save it in the group's quote book | — |
| `/quotes` | Show a random saved quote (`/quotes list` for the recent ones) | — |
| `/remember <fact>` | Save a fact (per-user in DM, per-group in groups) | — |
| `/forget` | Wipe saved facts — own in a DM, *shared group* facts in a group (admin-only) | — |
| `/forget me` | Wipe only your own remembered facts, anywhere, no admin needed | — |
| `/memory` | Show what the bot remembers | — |
| `/clear` | Reset conversation context (and end debate mode) for this chat/thread | — |
| `/quiet` | Toggle emoji-reaction engagement for this chat (admin-only) | — |

The five natural-language triggers above are handled by `bot/services/intent.py` (see Architecture
notes). The bot also: describes its own features conversationally ("что ты умеешь?"); reads forwarded
posts/photos (including from channels, citing the source), shared links, and images; transcribes
voice messages (in DMs or when replied to); answers natural-language search triggers ("найди...");
and can reply with a synthesized voice message on request ("ответь голосом") — see "Voice replies" in
Setup, off by default.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `TELEGRAM_TOKEN` — from [@BotFather](https://t.me/BotFather)
   - `MISTRAL_API_KEY` — from [console.mistral.ai](https://console.mistral.ai/api-keys). The free
     "Experiment" tier is enough for a small group; consider opting out of data-training use for
     your account in the Mistral Admin Console → Privacy (opt-out is more manual on the free tier
     than paid, but available).
   - `BOT_USERNAME` — your bot's username, without `@`
   - `REDIS_URL` — see below
2. `pip install -r requirements.txt` (or `requirements-dev.txt` to include test/lint tools)
3. `python main.py` — runs with long polling locally; set `RENDER=true` + `WEBHOOK_URL` +
   `WEBHOOK_PATH` + `WEBHOOK_SECRET` for webhook mode (see `render.yaml`).

### Voice replies (experimental, paid, off by default)

This is the only feature in the codebase that costs money and isn't already in use — it's off unless
you explicitly opt in. Voxtral TTS is **not** on Mistral's free tier: $0.016 per 1,000 characters
(a typical short reply costs a fraction of a cent, but it needs billing enabled on your Mistral
account). To turn it on, set `VOXTRAL_TTS_VOICE_ID` to a voice id from your Mistral account
(console.mistral.ai → Voices — a preset voice works, no cloning required); the bot then sends a
synthesized voice message alongside the text reply whenever someone asks "ответь голосом". Unlike the
rest of this codebase, `bot/services/speech.py` hasn't been verified against a live API key (no
network access to Mistral from the environment this was built in) — it fails safely to text-only on
any error, but treat the first real use as the actual test.

### Redis hosting

Any Redis 7.4+ instance works (whole-key `EXPIRE` refreshes, no advanced features needed). Two free
options that fit this bot's tiny footprint (no images stored, only short text):

- **[Render Key Value](https://render.com/docs/key-value)** — same platform as the bot, 25MB free.
- **[Upstash](https://upstash.com/)** — 256MB / 500K commands per month free, works from anywhere.

### Webhook security

`WEBHOOK_PATH` is a random value independent of `TELEGRAM_TOKEN`, so the token never ends up in
Render's request logs. Generate both:

```bash
python -c "import uuid; print(uuid.uuid4())"                  # WEBHOOK_PATH
python -c "import secrets; print(secrets.token_urlsafe(32))"  # WEBHOOK_SECRET
```

Both are optional (falls back to a per-process random path with a warning), but should be set for
any real deployment — a fixed `WEBHOOK_PATH` is also required for the spin-down fix below.

### Free-tier spin-down (messages silently lost)

Render's free plan stops the process after ~15 minutes idle; waking it back up takes 20-30+ seconds.
Because the bot runs in **webhook** mode with `drop_pending_updates=True`, a message that arrives
while it's asleep (or mid-restart) can be lost outright rather than delayed — from the outside this
looks exactly like "the bot stopped responding." Fix, free:
1. Set a **fixed** `WEBHOOK_PATH`/`WEBHOOK_SECRET` (above) — without a fixed path there's nothing
   stable to target.
2. Point a free uptime monitor ([UptimeRobot](https://uptimerobot.com/),
   [cron-job.org](https://cron-job.org/)) at `https://<your-app>.onrender.com/<WEBHOOK_PATH>` every
   ~10 minutes, accepting any response code — Telegram webhooks only accept `POST`, so a `GET` check
   gets a harmless `405`, which is still enough traffic to keep Render from spinning the service down.

Only a paid Render plan (e.g. Starter) avoids spin-down entirely.

## Deployment

Render is the only deployment tool — no separate CD pipeline. `render.yaml` is a
[Render Blueprint](https://render.com/docs/blueprint-spec) declaring the web service and the env var
names it expects. Once connected to this repo (via Render's GitHub App), Render **auto-deploys on
every push to `master`** — merging a PR is what ships it. `.github/workflows/ci.yml` (pytest + ruff)
is a separate GitHub-only check; it doesn't deploy anything unless branch protection requires it.

## Costs

Everything runs on Mistral's free tier except voice-message transcription (~$0.001/minute, already
in use) and, if enabled, voice replies (~$0.016/1000 characters, off by default — see Setup). Every
response also costs one extra free-tier moderation call. `bot/services/ai.py` logs a warning if daily
call volume gets high enough to be worth a look at the Mistral console — no hard cutoff, just
visibility (`QUOTA_WARN_THRESHOLD` in `config.py`).

## Development

```bash
pip install -r requirements-dev.txt
pytest              # run the test suite
ruff check .         # lint
```

Tests cover the pure/near-pure logic (URL handling, context windowing, style-signal extraction, HTML
metadata parsing, weather-query parsing) plus the AI/search/speech services with a mocked Mistral
client — nothing hits a live Telegram or Mistral connection.

## Architecture notes for future changes

- `config.py` — every tunable constant lives here (rate limits, TTLs, prompts, keyword lists).
- `bot/services/ai.py` — the Mistral client and `query_ai()`, the main entry point (streaming or
  blocking), plus `summarize_conversation()`, `suggest_poll()`, `suggest_quiz()`, and
  `classify_intent()`. Injects the current date/day-of-week (Europe/Tallinn) into the system prompt
  on every call. All JSON-returning calls pass `response_format={"type": "json_object"}` (Mistral's
  structured-output mode) rather than only asking for JSON in the prompt text. Takes a
  `reasoning_effort` param — Mistral Small 4 unifies fast chat and deep reasoning via this one
  parameter. The SDK's type hints suggest a 5-level scale, but the live model currently only accepts
  `"none"` or `"high"` (confirmed via a real 400 in production when `"low"` was tried — don't reuse
  the wider scale without checking against a live key first). Defaults to `"none"` for quick casual
  replies, `"high"` for debate mode and fact-checking.
- `bot/services/intent.py` — natural-language routing for summary/debate/factcheck/poll/quiz, in
  three tiers: (1) a specific phrase match resolves for free, no LLM call; (2) no phrase match and no
  loose signal word — the common case, plain chat — returns `None` for free; (3) a loose signal word
  without a clean phrase match spends one extra Mistral call (`ai.classify_intent`) to disambiguate
  and fill in a topic/claim. Phrase matching runs `bot/utils/helpers.py::mask_quoted_spans()` first
  (a quoted phrase is being mentioned, not commanded) and requires a word boundary so a keyword can't
  false-fire as a substring inside an unrelated word.
- `bot/handlers/actions.py` — the actual work behind `/summary`, `/debate`, `/factcheck`,
  `/poll suggest`, and `/quiz`, called both by `commands.py`'s slash-command handlers and by
  `messages.py`'s natural-language routing — the logic exists exactly once regardless of trigger.
- `bot/services/search.py` — live web search via Mistral's Conversations/Agents API
  (`beta.conversations.start_async` + `WebSearchTool`), not Chat Completions — the `web_search`
  connector is only available there. Result flows back into `query_ai()` as `referenced_content`,
  same pattern as weather data. `is_search_trigger()` uses the same quote-masking/word-boundary
  matching as `intent.py`.
- `bot/services/speech.py` — Voxtral TTS voice replies, see Setup. Off unless
  `VOXTRAL_TTS_VOICE_ID` is set.
- `bot/services/url_fetcher.py` — fetches shared links in three tiers: (1) `curl_cffi` with browser
  TLS/JA3 impersonation, tried in parallel; (2) on failure or a detected Cloudflare block, a
  best-effort fallback through `search.py`'s Mistral-mediated web search (different infra/IP, clears
  a different class of bot-protection, not guaranteed); (3) a non-fetching URL-heuristic string as
  the last resort.
- `bot/services/memory.py` — all Redis reads/writes; every write refreshes its own key's TTL.
- `bot/utils/context.py` — in-memory conversation window per `(chat_id, thread_id)`, with
  role-merging and age/turn-count-based compaction. Also tracks `last_bot_reply_target` so
  `messages.py` can scope the "implicit continuation" prompt rule to the right person.
- `bot/handlers/messages.py` — the main pipeline: routing, natural-language action-intent short
  circuit, reply/forward/URL/photo parsing, weather and web-search pre-fetch, context assembly, the
  `query_ai()` call, and post-processing.
- `bot/handlers/observer.py` — handler group 1, runs on every group message unconditionally.
