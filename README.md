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
  which only reacts to messages that are @-mentions, replies to the bot, plain-text mentions of the
  bot's name (`BOT_DISPLAY_NAMES` in `config.py` — "Sam"/"Сэм" by default, whole-word/case-insensitive
  via `mentions_bot_by_name()` in `bot/utils/helpers.py`), or private-chat messages.
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
  recently-active, non-quiet chats. **Disabled by default** (`DAILY_PROMPT_ENABLED = False` in
  `config.py`) — the bot should only speak when spoken to (or, occasionally, react with an emoji).

### Group chat robustness

A handful of behaviors worth knowing about for a busy multi-person chat:

- **Editing a message doesn't re-trigger the bot.** Telegram routes edited messages through the
  same handlers as new ones by default; `main.py`'s handlers explicitly exclude
  `filters.UpdateType.EDITED` so fixing a typo in your question doesn't produce a second reply
  (or double-count style/reaction stats in the silent observer).
- **Follow-up questions are scoped to the person who asked them.** The in-memory conversation
  window (`bot/utils/context.py`) is shared per `(chat_id, thread_id)` across everyone in the
  chat, but the "answer an unqualified follow-up as a continuation of the bot's last reply" prompt
  rule only applies when the current asker is the same person the bot's last reply (in that
  chat/thread) was addressed to — tracked via `set_last_bot_reply_target()`/
  `get_last_bot_reply_target()`. Without this, two people asking the bot unrelated things back to
  back could get a reply that's accidentally about the other person's topic.
- **External content is framed as untrusted data, not instructions.** Fetched web pages, search
  results, and forwarded posts get fed to the model as reference material — the system prompt
  explicitly tells it not to follow anything that reads like a command inside that content (a
  defense against a forwarded link or page containing a prompt-injection attempt).

## Commands

| Command | Description | Also works by just asking |
|---|---|---|
| `/start`, `/help` | Intro and usage guide | — |
| `/summary`, `/tldr [N]` | Summarize the last N (default 30) buffered messages | "о чём тут говорили?" |
| `/debate <topic>` | Bot takes an adversarial stance on `<topic>` for 30 minutes | "давай поспорим про X" |
| `/factcheck` | Verify a claim (reply to a message, or `/factcheck <claim>`) via live web search | "проверь факт: X" |
| `/poll Q \| A \| B \| C` | Send a native Telegram poll — revotable — (manual only, no natural-language equivalent) | — |
| `/poll suggest` | Ask the bot to propose a poll from the recent discussion | "сделай опрос" |
| `/remember <fact>` | Save a fact (per-user in DM, per-group in groups) | — |
| `/forget` | Wipe saved facts. In a DM, wipes your own. In a group, wipes the *shared group* facts — admin-only | — |
| `/forget me` | Wipe only your own remembered facts — works for anyone, in DMs or groups, no admin needed | — |
| `/memory` | Show what the bot remembers | — |
| `/clear` | Reset conversation context (and end debate mode) for this chat/thread | — |
| `/quiet` | Toggle emoji-reaction engagement for this chat (admin-only) | — |

The four natural-language triggers above are handled by `bot/services/intent.py` — see
"Architecture notes" below for how it decides when a plain sentence means one of these. The bot can
also describe its own features conversationally (e.g. "что ты умеешь?") since `bot/services/ai.py`'s
system prompt includes a short capabilities summary.

The bot also responds to forwarded posts/photos (including forwards from channels — it captures and
cites the original source), shared links, images (menus, flyers, screenshots — anything), voice
messages (in DMs or when replied to), and natural-language search triggers ("найди...", "search for...").

**Voice replies** ("ответь голосом", "скажи голосом") get a synthesized voice message alongside the
normal text reply — see "Voice replies (experimental)" under Setup; off unless configured.

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

### Voice replies (experimental)

Off by default. To turn it on, set `VOXTRAL_TTS_VOICE_ID` to a voice id from your Mistral account
(console.mistral.ai → Voices — a preset voice works, no cloning required) and the bot will send a
synthesized voice message alongside the text reply whenever someone asks "ответь голосом" /
"скажи голосом". Unlike everything else in this codebase, `bot/services/speech.py` hasn't been
verified against a live API key (no network access to Mistral from the development environment this
was built in) — the request/response shapes are built from the official docs and SDK introspection.
It fails safely either way (falls back to text-only on any error), but treat the first real use as
the actual test. Priced per minute like transcription, exact rate not published at time of writing.

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

### Free-tier spin-down (messages silently lost)

Render's free web-service plan stops the process after ~15 minutes with no incoming HTTP traffic,
and cold-starting it back up takes 20-30+ seconds. Because the bot runs in **webhook** mode on
Render, a message that arrives while it's asleep (or during that boot window) can fail to be
delivered at all — `run_webhook(..., drop_pending_updates=True)` means anything that piles up while
the process is down gets discarded on restart rather than replayed, and Telegram's own webhook
retry window is short. From the outside this looks exactly like "the bot stopped responding":
Telegram still marks your message delivered (it reached the webhook endpoint or was queued for
retry), but nothing ever processes it.

Fix (free, a few minutes of setup):
1. Set a **fixed** `WEBHOOK_PATH` and `WEBHOOK_SECRET` as real Render env vars (see above) — without
   a fixed path, it's a fresh random value every restart, so nothing external can reliably target it.
2. Point a free uptime monitor (e.g. [UptimeRobot](https://uptimerobot.com/),
   [cron-job.org](https://cron-job.org/)) at `https://<your-app>.onrender.com/<WEBHOOK_PATH>` on a
   ~10 minute interval, configured to treat **any** response as "up" (not just 200). Telegram
   webhooks only accept `POST`, so a monitor's `GET`/`HEAD` check will get back a `405` — that's
   expected and fine, it's still real HTTP traffic reaching the port, which is all Render checks for
   to keep a free service warm. (There's no supported way to bolt a separate `/health` route onto
   python-telegram-bot's webhook server — it's a fixed single-route Tornado app internally — so
   pinging the existing webhook path is the simplest option that doesn't touch PTB internals.)

If losing an occasional message is unacceptable, the only complete fix is a paid Render plan
(e.g. Starter) — those don't spin down.

## Deployment

Render is the only deployment tool this project uses — there's no separate CD pipeline. `render.yaml`
in the repo root is a [Render Blueprint](https://render.com/docs/blueprint-spec): it declares one
`web` service (`runtime: python`, `buildCommand: pip install -r requirements.txt`,
`startCommand: python main.py`) plus the env var names it expects (values with `sync: false` are
secrets you set once in the Render dashboard, not stored in the repo).

Once the Render service is connected to this GitHub repo (via Render's GitHub App, set up from the
Render dashboard when the service was first created), Render watches the branch it's configured to
deploy from — normally `master` — and **auto-deploys on every push to that branch**: it pulls the
new commit, re-runs `buildCommand`, and restarts the service with `startCommand`. Nothing needs to
run on this side to trigger it; merging a PR into `master` is what ships it. `.github/workflows/ci.yml`
(pytest + ruff) is a separate, unrelated check that runs on GitHub and does not deploy anything —
it only gates PRs if you turn on branch protection requiring it to pass.

If you ever need to check *which* commit is actually live, or force a redeploy without a new
commit, that's done from the Render dashboard (Manual Deploy) or the Render CLI — not from GitHub.

## Costs

Everything runs on Mistral's free tier except voice-message transcription (Voxtral,
~$0.001/minute) and, if you turn it on, voice replies (Voxtral TTS, priced per minute, exact rate
not published) — trivial for a small group, but worth knowing these are the parts of the bot that
aren't strictly free. Every response also gets one extra Mistral call for output moderation (see
"Output safety check" below) — still free-tier, but it does count toward `QUOTA_WARN_THRESHOLD`.
`bot/services/ai.py` logs a warning if daily API call volume climbs high enough to be worth a look
at the Mistral console — there's no hard cutoff, just visibility.

## Output safety check

After the incidents documented in the git history (the bot going hostile, then swaggering — both
patched via the system prompt), `bot/services/ai.py::query_ai()` now also runs the bot's own
finished reply through Mistral's Moderation API (`client.classifiers.moderate_async`,
`mistral-moderation-2603`) before treating it as final. It only checks categories relevant to those
two failure modes (`hate_and_discrimination`, `violence_and_threats`, `sexual`, `self_harm`) —
deliberately not a blanket filter, so normal colorful group-chat banter doesn't get flagged. A
flagged reply gets replaced with a short, honest fallback instead of being sent. The check fails
open: if the moderation call itself errors, the original reply goes through unchanged rather than
the whole bot going silent over a moderation-API hiccup. This is on top of the system-prompt tone
rules, not instead of them — prompting is still the first line of defense.

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
  plus `summarize_conversation()`, `suggest_poll()`, and `classify_intent()`. Every call injects the
  current date/day-of-week (Europe/Tallinn) into the system prompt, so date-relative reasoning
  ("сегодня"/"завтра", how recent something actually is) is grounded instead of guessed. Every
  JSON-returning call (`suggest_poll`, `classify_intent`, and `memory.py`'s two fact-extraction
  functions) passes `response_format={"type": "json_object"}` — Mistral's real structured-output
  mode — instead of only asking for JSON in the prompt text and hoping; before this, the model would
  occasionally wrap its JSON in a markdown code fence and break the naive `json.loads()` call (this
  was an actual, observed production warning, not a hypothetical). The manual `json.loads()` +
  except stays as a defensive fallback, but should rarely trigger now.
- `bot/services/intent.py` — decides whether a plain-language message means "run /summary" (etc.)
  in three tiers: (1) a specific phrase match ("о чём говорили", "сделай опрос") resolves for free,
  no LLM call; (2) no phrase match and no loose signal word either — the common case, plain chat —
  returns `None` for free, same cost as today; (3) a loose signal word is present but no clean
  phrase match (e.g. "дебат" appears but not in a recognized trigger) — spends one extra Mistral
  call (`ai.classify_intent`) to disambiguate and to fill in a topic/claim a phrase match found the
  trigger for but not the parameter (e.g. "давай поспорим" with no topic in the sentence). Phrase
  matching first runs `bot/utils/helpers.py::mask_quoted_spans()` — a phrase in quotes is being
  mentioned, not issued as a command (e.g. an announcement listing usage examples like «сделай
  саммари» shouldn't itself trigger one) — and matches require a leading word boundary so a keyword
  can't false-fire as a substring buried inside an unrelated word.
- `bot/handlers/actions.py` — `do_summary()`/`do_debate()`/`do_factcheck()`/`do_poll_suggest()`, the
  actual work behind those four commands, called both by `bot/handlers/commands.py`'s slash-command
  handlers (parsing `context.args`) and by `messages.py`'s natural-language routing (parsing
  `intent.py`'s output) — the logic exists exactly once regardless of how it was triggered.
- `bot/services/search.py` — live web search. Uses Mistral's **Conversations/Agents API**
  (`beta.conversations.start_async` with a `WebSearchTool`), *not* Chat Completions — Mistral's
  `web_search` connector is only available there, which is why it's a separate call whose result
  gets fed back into `query_ai()` as `referenced_content`, the same pattern used for weather data.
  `is_search_trigger()` uses the same `mask_quoted_spans()` + word-boundary matching as
  `intent.py`, for the same reason — a quoted example shouldn't fire a real (costly) search.
- `bot/services/speech.py` — experimental opt-in voice replies via Voxtral TTS, see "Voice replies
  (experimental)" under Setup. Off unless `VOXTRAL_TTS_VOICE_ID` is set; `is_voice_reply_trigger()`
  uses the same quote-masking pattern as intent/search matching.
- `bot/services/url_fetcher.py` — fetches shared links in three tiers: (1) `curl_cffi` with browser
  TLS/JA3 impersonation (`IMPERSONATE_PROFILES` in `config.py`, tried in parallel, generic
  `"chrome"`/`"safari"` aliases so they always track the latest supported browser fingerprint); (2) if
  every profile fails or gets Cloudflare-block-detected, a best-effort fallback through
  `bot/services/search.py`'s Mistral-mediated web search, asking it to open and summarize the URL
  directly — different infra/IP than this process, so it clears a different class of bot-protection
  (not guaranteed; paywalls and heavily JS-gated sites can still beat both); (3) a non-fetching
  URL-heuristic string as the last resort so the model at least knows the URL exists.
- `bot/services/memory.py` — all Redis reads/writes. Every write path refreshes its own key's TTL.
  Per-user facts (`user:{id}:facts`) are global to the person, not scoped to a chat — the same
  bucket is used whether they're in a DM or any group — which is why `/forget me` (any group
  member, self-service) and admin-only bare `/forget` (the group's shared `group:{chat_id}:facts`
  bucket) in `commands.py::forget_command` are two genuinely different operations.
- `bot/utils/context.py` — in-memory conversation window per `(chat_id, thread_id)`, with
  role-merging and a simple age/turn-count-based compaction pass before sending to the API. Also
  tracks `last_bot_reply_target` per `(chat_id, thread_id)` — who the bot's most recent reply there
  was addressed to — so `messages.py` can tell `ai.py` when an unqualified follow-up is from a
  *different* person than the last exchange, scoping the "implicit continuation" prompt rule
  correctly instead of letting it bleed across unrelated conversations in a busy group.
- `bot/handlers/messages.py` — the main pipeline: routing, natural-language action-intent routing
  (`intent.py` → `actions.py`, short-circuits the rest of the pipeline on a match),
  reply/forward/URL/photo parsing, weather and web-search pre-fetch, context assembly, the
  `query_ai()` call, and post-processing.
- `bot/handlers/observer.py` — handler group 1, runs on every group message unconditionally.
