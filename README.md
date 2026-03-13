# Discord Bot with Local LLM

A Discord mention-based chatbot that connects to a local OpenAI-compatible LLM endpoint (for example, LM Studio), supports multimodal prompts (text, images, stickers, and PDFs), ingests content from URLs, keeps per-server chat memory, and performs autonomous web search when needed.

## What It Does

- Responds when the bot is mentioned in a server channel
- Uses an OpenAI-compatible chat API via configurable base URL
- Supports image and sticker understanding in prompts
- Supports PDF attachments (extracts text and adds it to model context)
- Ingests URLs found in messages:
  - Web pages and plain text pages are fetched and cleaned
  - Direct image URLs are treated like image attachments
  - Direct PDF URLs are parsed and added as text context
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
- `beautifulsoup4`
- `PyMuPDF`

## Project Structure

- `bot.py`: Main bot implementation
- `.env`: Runtime secrets and model endpoint config (not committed)
- `bot_database.db`: SQLite data store (created at runtime)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install discord.py openai aiosqlite python-dotenv ddgs pillow aiohttp beautifulsoup4 pymupdf
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

## Input Limits and Safety Guards

- 10MB limit for attached images and PDFs.
- 10MB limit for URL-fetched files.
- PDF extraction is capped to the first 16 pages (index 0 through 15).
- Extracted webpage and PDF text is truncated to 40,000 characters.
- Unsupported URL content types are skipped with a system note.

## Notes

- The bot uses a tool-calling loop for web search via DuckDuckGo when the model requests it.
- Large images are resized to fit within 1024x1024 and JPEG-compressed before sending to the LLM.
- If a non-vision model is loaded and media is sent, the bot replies with a compatibility warning.
- If the local model endpoint is offline, `status` reports it as `Offline / Unreachable`.

## License

MIT. See `LICENSE`.
