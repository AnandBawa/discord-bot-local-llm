# Discord Bot with Local LLM

A Discord mention-based chatbot that connects to a local OpenAI-compatible LLM endpoint (for example, LM Studio), supports multimodal prompts (text, images, stickers, and PDFs), ingests content from URLs, keeps per-server chat memory, and performs autonomous web search when needed.

## What It Does

- Responds when the bot is mentioned in a server channel
- Uses an OpenAI-compatible chat API via configurable base URL
- Supports image and sticker understanding in prompts
- Supports PDF attachments (extracts text and adds it to model context)
- Ingests URLs found in messages through Jina Reader (`https://r.jina.ai/`):
  - Direct image URLs are treated like image attachments
  - PDF URLs are parsed and added as text context
  - Other content is decoded as text and truncated safely
- Maintains per-server chat history in SQLite
- Extracts and stores per-user core memories per server
- Supports server-level personality prompts (`role` command)
- Exposes diagnostic and memory management commands
- Allows owner-only administrative data wipe commands
- Ignores DMs by design (server-only operation)

## Tech Stack

- Python 3.12+
- `discord.py`
- `openai` (async client, OpenAI-compatible endpoint)
- `aiosqlite`
- `python-dotenv`
- `ddgs` (DuckDuckGo search)
- `Pillow`
- `aiohttp`
- `PyMuPDF`

## Project Structure

- `bot.py`: Main bot implementation
- `.env`: Runtime secrets and model endpoint config (not committed)
- `bot_database.db`: SQLite data store (created at runtime)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install discord.py openai aiosqlite python-dotenv ddgs pillow aiohttp pymupdf
```

3. Create a `.env` file:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=local-model
```

4. Run the bot:

```bash
python bot.py
```

## Required Discord Configuration

- Enable **Message Content Intent** for the bot application.
- Invite the bot with permissions to read/send messages in target channels.
- Use the bot in servers (DMs are intentionally ignored).

## Command Reference

Use commands by mentioning the bot, for example: `@BotName help`

- `help`: Show command help
- `status`: Show latency, model status, and history metrics
- `role`: Show current server personality
- `role <prompt>`: Set server personality and reset chat history
- `role clear`: Reset to default personality and clear server history
- `clear`: Clear server conversation history
- `memory`: List known users with stored core memories
- `memory <name>`: Show stored facts for a user
- `memory clear`: Delete your own memory and chat history in this server

Owner-only commands:

- `force-forget @user`: Delete a specific user's data in the current server
- `wipe-server-memories`: Delete all memory and chat history for the current server

## Data and Memory Behavior

- Chat history is stored per server in SQLite.
- Core memories are keyed by `(server_id, user_id)`.
- History capacity is managed with chunk eviction:
  - When history reaches 50 records, the oldest 20 are extracted.
  - Extracted messages are summarized into core memories asynchronously.
- Memory and wipe operations include safeguards to prevent stale background tasks from restoring deleted data.
- Large extracted context from PDFs and URLs is treated as ephemeral runtime context:
  - It is sent to the model for the current response.
  - Lightweight message content is stored in the database instead of full scraped blobs.
- Historical image payloads are scrubbed from replay context to reduce KV-cache and VRAM pressure.

## Runtime Pipeline

The message flow is organized into explicit pipeline stages:

1. Command interception (`help`, `role`, `clear`, `status`, `memory`, admin commands)
2. Context extraction (attachments, reply context, URL fetches)
3. Payload building (text/media packaging for API and DB)
4. AI context assembly (system prompt, memory injection, history cleanup)
5. Generation and tool loop execution (up to 3 tool iterations)
6. Persistence and response delivery (DB save + chunked Discord send)

Concurrency and safety controls:

- LLM call concurrency is capped with a semaphore (`llm_queue = 3`).
- Memory writes are serialized with `memory_lock`.
- Deletion requests are timestamped and stale entries are pruned.
- Typing indicators are wrapped with a safe context manager to avoid permission/HTTP failures.

## Input Limits and Safety Guards

- 10MB limit for attached images and PDFs.
- 10MB limit for URL-fetched files.
- PDF extraction is capped to the first 15 pages.
- Extracted webpage and PDF text is truncated to 40,000 characters.
- URL fetch timeout is 15 seconds.
- URL fetches use `X-Return-Format: markdown` and `X-No-Cache: true` headers via Jina Reader.

## Deployment Tuning Note

Several runtime parameters are intentionally conservative defaults and should be tuned for your specific hardware, model size, and latency targets.

In particular, review and adjust values such as:

- LLM concurrency (`llm_queue` semaphore)
- File size limits (`MAX_FILE_SIZE`)
- PDF extraction page cap (currently first 15 pages)
- Text truncation threshold (currently 40,000 characters)
- Generation settings (`temperature`, `max_tokens`)
- Tool loop limits (max iterations)
- History retention policy (`MAX_HISTORY_LENGTH`, and eviction thresholds such as `>= 50` with oldest `LIMIT 20`)
- Image preprocessing (`thumbnail((1024, 1024))`, JPEG `quality=85`)

If your model server has lower VRAM or slower throughput, reduce concurrency and token/context-related limits. If you have stronger hardware and stable latency, you can increase them cautiously while monitoring response time, memory pressure, and failure rates.

## Notes

- The bot exposes one tool (`web_search`) and runs a tool-calling loop for up to 3 iterations.
- Large images are resized to fit within 1024x1024 and JPEG-compressed before sending to the LLM.
- Sticker bytes are base64-encoded and sent as image payloads.
- If a non-vision model is loaded and media is sent, the bot replies with a compatibility warning.
- If the local model endpoint is offline, `status` reports it as `Offline / Unreachable`.
- Response chunking is Markdown-aware for fenced code blocks to avoid broken formatting across Discord message splits.

## License

MIT. See `LICENSE`.
