# Tallinn Bot

A Telegram bot for a private friends' group centered on life in Tallinn, Estonia — Q&A, link/photo
analysis, weather, debate facilitation, and persistent memory of the group's running conversation.

## How it works

**Framework:** [python-telegram-bot](https://docs.python-telegram-bot.org/) 22.8 (async), running on
Python 3.14, deployed on [Render](https://render.com/) (webhook in production, long polling locally).

**AI:** [Mistral AI](https://mistral.ai/) (`mistral-small-latest`, currently Mistral Small 4 —
vision-capable). The bot uses Mistral for everything: answering questions, extracting facts,
summarizing, debate mode, poll suggestions, fact-checking, and speech-to-text (Voxtral).

**Memory:** Redis holds long-lived state (facts about people/groups, per-user communication style,
a rolling buffer of recent messages, quiet-mode/debate-mode flags). Every write refreshes a 90-day
TTL on its own key, so Redis expires stale data on its own — there's no separate cleanup job.
An in-memory, per-process cache holds the active conversation window for low-latency replies; Redis
is the fallback that survives restarts.

### Handler groups

- **Group 0** (default): commands (`/start`, `/summary`, `/debate`, ...) and the main message handler,
  which only reacts to messages that are @-mentions, replies to the bot, or private-chat messages.
- **Group 1** (silent observer): runs on *every* group message regardless of whether the bot was
  addressed. It stores the message into Redis, updates the sender's style profile, and — at a small
  default probability — reacts with an emoji to keep the bot feeling present without being spammy.
  A full spontaneous text-reply path also exists but is disabled by default (`SPONTANEOUS_REPLY_PROBABILITY = 0`
  in `config.py`) since the group found it too noisy.

### Scheduled jobs (JobQueue)

- **Proactive memory** (~every 8h): reviews each chat's recent-message buffer and extracts facts the
  per-message pipeline missed.
- **Style refresh** (daily, 14:00 Tallinn time): regenerates natural-language style summaries for
  active users so the bot's tone stays adapted to how each person actually talks.
- **Daily prompt** (daily, 18:00 Tallinn time): posts a random icebreaker/trivia question to
  recently-active, non-quiet chats.

## Commands

| Command | Description |
|---|---|
| `/start`, `/help` | Intro and usage guide |
| `/summary`, `/tldr [N]` | Summarize the last N (default 30) buffered messages |
| `/debate <topic>` | Bot takes an adversarial stance on `<topic>` for 30 minutes |
| `/factcheck` | Verify a claim (reply to a message, or `/factcheck <claim>`) via live web search |
| `/poll Q \| A \| B \| C` | Send a native Telegram poll |
| `/poll suggest` | Ask the bot to propose a poll from the recent discussion |
| `/remember <fact>` | Save a fact (per-user in DM, per-group in groups) |
| `/forget` | Wipe saved facts (group chats: admin-only) |
| `/memory` | Show what the bot remembers |
| `/clear` | Reset conversation context (and end debate mode) for this chat/thread |
| `/quiet` | Toggle emoji-reaction engagement for this chat (admin-only) |

The bot also responds to forwarded posts/photos (including forwards from channels — it captures and
cites the original source), shared links, images (menus, flyers, screenshots — anything), voice
messages (in DMs or when replied to), and natural-language search triggers ("найди...", "search for...").

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

### Redis hosting

Any Redis 7.4+ instance works (the code uses whole-key `EXPIRE` refreshes, compatible everywhere;
7.4+ is only needed if you extend it with per-field hash TTLs). Two free options that fit this
bot's tiny footprint (no images are ever stored in Redis, only short text):

- **[Render Key Value](https://render.com/docs/key-value)** — same platform as the bot, 25MB free.
- **[Upstash](https://upstash.com/)** — 256MB / 500K commands per month free, works from anywhere.

### Webhook security

The webhook path is a random value (`WEBHOOK_PATH`) independent of `TELEGRAM_TOKEN`, so the bot
token never ends up in Render's request logs. Generate one with:

```bash
python -c "import uuid; print(uuid.uuid4())"       # WEBHOOK_PATH
python -c "import secrets; print(secrets.token_urlsafe(32))"  # WEBHOOK_SECRET
```

Both are optional (the bot falls back to a per-process random path and logs a warning), but should
be set for any real deployment.

## Costs

Everything runs on Mistral's free tier except voice-message transcription (Voxtral,
~$0.001/minute) — trivial for a small group, but worth knowing it's the one part of the bot that
isn't strictly free. `bot/services/ai.py` logs a warning if daily API call volume climbs high
enough to be worth a look at the Mistral console (`QUOTA_WARN_THRESHOLD` in `config.py`) — there's
no hard cutoff, just visibility.

## Development

```bash
pip install -r requirements-dev.txt
pytest              # run the test suite
ruff check .         # lint
```

Tests cover the pure/near-pure logic (URL handling, context windowing, style-signal extraction,
HTML metadata parsing, weather-query parsing) plus the AI/search services with a mocked Mistral
client — nothing hits a live Telegram or Mistral connection.

## Architecture notes for future changes

- `config.py` — every tunable constant lives here (rate limits, TTLs, prompts, keyword lists).
- `bot/services/ai.py` — the Mistral client, main `query_ai()` entry point (streaming + blocking),
  plus `summarize_conversation()` and `suggest_poll()`.
- `bot/services/search.py` — live web search. Uses Mistral's **Conversations/Agents API**
  (`beta.conversations.start_async` with a `WebSearchTool`), *not* Chat Completions — Mistral's
  `web_search` connector is only available there, which is why it's a separate call whose result
  gets fed back into `query_ai()` as `referenced_content`, the same pattern used for weather data.
- `bot/services/memory.py` — all Redis reads/writes. Every write path refreshes its own key's TTL.
- `bot/utils/context.py` — in-memory conversation window per `(chat_id, thread_id)`, with
  role-merging and a simple age/turn-count-based compaction pass before sending to the API.
- `bot/handlers/messages.py` — the main pipeline: routing, reply/forward/URL/photo parsing, weather
  and web-search pre-fetch, context assembly, the `query_ai()` call, and post-processing.
- `bot/handlers/observer.py` — handler group 1, runs on every group message unconditionally.
