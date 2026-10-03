# Discord AI Bot

A Discord server bot with a shared conversation and persona, plus remembered facts associated with members. It connects to local OpenAI-compatible chat and embedding endpoints, with optional cloud fallback, web search, PDF extraction, and image analysis.

## Key Features

- **Shared Server Memory:** SQLite stores chronological conversation history as plain text and retains input awaiting memory extraction. Attachment notes and image descriptions survive in history; image bytes are sent only with the current model request. ChromaDB stores extracted text facts for semantic retrieval. Failed extraction is retried, including after restart.
- **Explicit Memory:** `/remember` saves a fact immediately in SQLite while the background worker indexes it in Chroma. Members can view saved facts or clear their own memory through `/memory`; facts remain shared within the server.
- **Ordered Conversations:** Chat turns run in arrival order within each server, including across channels. Different servers can progress concurrently, subject to the existing global limit of three chat/memory-extraction LLM tasks.
- **Cloud Fallback:** Chat and memory extraction share local/cloud routing. Embeddings have an independent failure cooldown. A chat turn stays on its selected fallback throughout tool calls. Both chat SDK clients use a 2-second connection timeout, a 120-second read timeout, and no automatic SDK retries.
- **Image Analysis:** Passes images and supported Discord stickers to a vision-capable chat model. Images use Pillow resizing; `VISION_ENABLED` controls whether visual input is sent to the model.
- **Autonomous Web Search:** Uses `ddgs` to find missing information, including when an uploaded document is insufficient. Search terms and dates are preserved. Results include source URLs; the bot requests citations and appends up to three search source links if omitted from the answer.
- **URL and Document Parsing:** Extracts text from uploaded PDF files using PyMuPDF (`pymupdf`) and converts public URLs into readable Markdown using the Jina Reader API (`r.jina.ai`). URL downloads reject internal addresses, including redirect destinations.
- **Logging:** Writes logs to `bot.log` and truncates long console messages. Logging limitations and retention concerns are recorded in the audit.
- **Slash Commands:** Provides commands for the shared persona, history, memory, and status. Members can change the server persona; force-forget and server-wipe commands require administrator or bot-owner access.
- **Permission-Aware Replies:** Uses native Discord replies when permitted and ordinary messages mentioning the requester otherwise. Reply context uses content already delivered or cached; fetching older messages requires Read Message History.

## Prerequisites

- Python 3.12. The dependency snapshot is verified on Python 3.12 on Linux.
- A Discord Bot Token (with the **Message Content Intent** enabled in the Discord Developer Portal). When creating the OAuth2 URL for bot invite, select **bot** and **application.commands** under **Scopes**, and **View Channels** and **Send Messages** under **Bot Permissions**. For threads/posts, also grant **Send Messages in Threads**. **Read Message History** enables native replies and fetching referenced messages; the bot falls back to ordinary messages when it is missing. Effective permissions include channel overrides. See [Discord's message permissions](https://docs.discord.com/developers/resources/message#create-message).
- An active LLM API endpoint (defaults to a local instance running on `http://localhost:1234/v1`).
- A Text Generation model and a separate Text Embedding model (e.g., `jina-embeddings-v5-text-small`) loaded in your local inference server.

## Installation

1. **Set up the project directory and virtual environment:**

```bash
python3.12 -m venv venv_bot
source venv_bot/bin/activate # On Windows use: venv_bot\Scripts\activate
```

2. **Install dependencies:**
   Install the checked package versions and their resolved dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` pins the latest stable application packages checked on October 3, 2026. `constraints.txt` pins their resolved dependencies. Some supporting packages have upper version limits imposed by the application packages; retain compatible versions when updating the snapshot.

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and populate it with your credentials:

```env
# Core Discord Setup

DISCORD_BOT_TOKEN=your_discord_bot_token_here
BOT_OWNER_ID=your_discord_user_id_here

# Primary Local LLM Setup

LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=local-model
# Use the exact embedding model identifier shown by your local server.
EMB_MODEL_NAME=embedding-model
VISION_ENABLED=True

# Secondary Cloud Fallback Setup (Optional)

FALLBACK_BASE_URL=
FALLBACK_API_KEY=
FALLBACK_MODEL_NAME=
FALLBACK_EMB_API_KEY=

# Memory Tuning (Optional)

MEMORY_DISTANCE_THRESHOLD=0.45
```

4. **Run the Bot:**

```bash
python bot.py
```

Startup loads `.env`, configures logging, and creates the provider clients and storage connections. Importing `bot.py` registers the bot's handlers without reading `.env`, creating a log/database, constructing model clients, or connecting to Discord. Tests can supply configuration, model clients, and temporary storage paths directly.

## Dependency Verification and Audit

Run the isolated compatibility and regression checks:

```bash
python scripts/check_dependencies.py
python scripts/check_regressions.py
python scripts/check_features.py
python scripts/check_refactor.py
```

The check uses synthetic messages, mocked API responses, and temporary databases. It verifies command registration, SDK timeouts and tool calls, SQLite history, Chroma memory, image processing, and PDF extraction without loading `.env` or logging in to Discord.

The regression checks cover URL restrictions, forgetting during background work, and retaining failed extraction input. They use synthetic data, temporary databases, and a controlled local HTTP server; external socket connections are blocked.

The feature checks cover missing Discord permissions, search queries and sources, explicit memory saves and retries, scoped clearing, and conversation ordering across servers. The dependency check also exercises saving and retry after a failed database acknowledgement with real temporary SQLite and Chroma storage.

The refactor checks cover import/startup behavior, literal JSON history, image/sticker payloads, direct/replied attachment limits, failed downloads, persona recovery, transaction rollback/cancellation, shutdown cleanup, independent provider cooldowns, and tool-loop limits. The plain-text history format follows the owner's data reset; no decoder or migration for the former mixed text/JSON history is included.

History removed by pruning, `/clear`, or `/role` is retained in a `pending_memories` table until extraction succeeds. The bot retries in the background and after restart. Forget/wipe commands also remove this retained input and invalidate unfinished responses. The table is created automatically in the existing SQLite database.

Explicit facts use an `explicit_memories` SQLite table, created automatically at startup. Individual memory editing/deletion has been removed. Its former override and suppression tables are no longer used or migrated; the owner chose to reset the saved data. Restart the bot to load the updated code and sync the slash-command options; this does not require re-inviting it to servers where its slash commands already work.

The [audit and improvement plan](docs/AUDIT.md) records findings, accepted server-wide behavior, implemented fixes, and deferred improvements.

## Bot Commands

The bot features two distinct ways to interact: standard conversational tagging, and native Slash Commands (`/`).

### General Chat

- **`@BotName [message]`**: Chat or ask questions in the channel. The bot can analyze supported attachments and public links.
- **Reply to the Bot**: A reply whose referenced message is delivered or cached is recognised without an extra tag. Tag the bot if the reference is unavailable. Without Read Message History, it can use its saved conversation but cannot fetch missing Discord messages or their attachments.

Unsupported file attachments and oversized images/PDFs are skipped, with a note passed to the chat model; remaining text and supported attachments are processed normally. The explanation to the user depends on the model, so there is no guaranteed rejection message. Unsupported animated stickers are silently skipped; a mention with only such a sticker receives the generic `/help` greeting.

### Slash Commands (`/`)

- **`/help`**: Display the command guide (Ephemeral - only visible to you).
- **`/status`**: Show ping, server history, chat/memory models, supported inputs, and key limits. Model lines reflect each service's most recent successful request across the bot; fallback appears on the relevant model line only after successful use. Before first use, the configured primary models are shown. The command makes no provider requests. Images and web search require a compatible chat model.
- **`/role`**: View the shared persona, or change it and start a fresh server conversation. Extraction input from the previous conversation is retained until processing succeeds. Type `clear` to restore the neutral default.
- **`/remember fact:...`**: Save a fact about yourself immediately, up to 500 characters. The confirmation is private; the saved fact is shared server memory.
- **`/memory`**: List tracked users, read facts (your own by default), or clear your own saved facts and conversation data in the current server. `target_user` applies only to reading.
- **`/clear`**: Reset the visible server conversation and queue its text for memory extraction. Existing ChromaDB facts are retained.
- **`/force-forget`**: _(Admin/Owner Only)_ Delete a user's attributed history, retained extraction input, and ChromaDB facts in the current server.
- **`/admin_wipe_server`**: _(Admin/Owner Only)_ Delete vector memories, chat history, and retained extraction input for the current server. The stored persona is retained.

Memory command examples:

```text
/remember fact:I prefer Python for small scripts.
/memory action:read
/memory action:list
```

`/remember` adds a fact; it does not edit or replace existing facts. Indexing retries if the embedding service or Chroma is unavailable. Up to 20 recent pending explicit facts are included directly in chat context while waiting for indexing. `/memory action:clear` removes your saved facts, retained extraction input, and saved conversation in the current server. Individual fact editing and deletion are not available.

The per-server conversation queue also covers context loading and saving replies. Slow requests delay later turns in that server. Clear, forget, persona changes, and explicit memory changes invalidate older running/queued turns before they save their answers. Already dispatched Discord messages are not retracted. The queue is in-process; waiting chat turns are not resumed after a restart.

## Embedding Models

The local embedding model is selected by `EMB_MODEL_NAME`. Its model family, dimensionality, and preprocessing must match the API adapter when both providers share an index. The [audit's embedding note](docs/AUDIT.md#f04-embedding-fallback-can-mix-incompatible-vectors) records the task/preprocessing concern.

Jina v5 Omni Small adds image, audio, video, and PDF embeddings. Jina documents matching text embeddings between corresponding Text and Omni models, so a model-name change alone is not expected to improve this bot's text-fact retrieval. Using Omni's media capabilities would require indexing the attachments themselves and retaining their references. The bot continues to use Text Small. See [Jina's model documentation](https://jina.ai/embeddings/).

As checked on October 3, 2026, Omni is available through Jina's API. Full multimodal support in standard LM Studio has not been verified; Jina's [GGUF instructions](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval-GGUF#install-llamacpp-with-multimodal-patches) require a patched llama.cpp build. Task settings, dimensions, and the local quantization should be checked before treating providers as interchangeable.

## Advanced Configuration

You can adjust constants directly in the `GLOBAL STATE & CONFIGURATION` section of `bot.py`. Key settings include:

**Model & Context Limits:**

- `MAX_HISTORY_LENGTH` (Default: 100) - Maximum messages kept in active SQLite chat history before vector summarization.
- `MAX_TOOL_ITERATIONS` (Default: 3) - Maximum tool-call rounds in one turn; each round may contain multiple searches.
- `LLM_TEMPERATURE` (Default: 1.0) - Controls the creativity and randomness of standard chat responses.
- `LLM_MAX_TOKENS` (Default: 4096) - Maximum token length for standard chat responses.
- `MEMORY_TEMPERATURE` (Default: 0.1) - Creativity for fact extraction (kept low to ensure strict factual JSON output).
- `MEMORY_MAX_TOKENS` (Default: 500) - Maximum token length when the AI is generating memory JSON arrays.
- `MEMORY_MAX_MSG_CHARS` (Default: 2000) - Max characters per message fed into the background memory extractor.
- `MEMORY_DEDUPLICATION_THRESHOLD` (Default: 0.15) - Strict cosine distance threshold used to prevent the bot from saving nearly identical facts into the database.

**Hardware & Parsing Limits:**

- `MAX_FILE_SIZE` (Default: 10MB) - Size limit for direct and replied-to image/PDF attachments and URL downloads.
- `MAX_PDF_PAGES` (Default: 15) - Maximum pages read from a PDF.
- `MAX_TEXT_EXTRACTION_LENGTH` (Default: 40000) - Character limit for text extracted from URLs or PDFs.
- `MAX_IMAGE_DIMENSION` (Default: 1024) - Images are resized to this maximum width/height to save VRAM.
- `IMAGE_COMPRESSION_QUALITY` (Default: 85) - Pillow JPEG compression quality.
- `SCRAPER_TIMEOUT` (Default: 15) - Seconds to wait for web scraping or large native file downloads.
- `WEB_SEARCH_MAX_RESULTS` (Default: 3) - Number of search result snippets pulled from DuckDuckGo.

**Discord & System Limits:**

- `DISCORD_CHUNK_LIMIT` (Default: 1980) - Max character limit per Discord message chunk.
- `CHUNK_MESSAGE_DELAY` (Default: 1.5) - Seconds to wait between sending chunks to avoid rate limits.
- `MEMORY_RETRY_INTERVAL` (Default: 60) - Seconds to wait before retrying retained memory extraction input.
- `MANUAL_MEMORY_CONTEXT_LIMIT` (Default: 20) - Maximum pending explicit facts included directly in chat context before indexing completes.
- `DEFAULT_PERSONA` - Fallback system prompt if no custom role is set for a server.
- `CIRCUIT_BREAKER_COOLDOWN` (Default: 60) - Seconds to bypass a failed local service when its fallback is configured. Chat/extraction and embedding cooldowns are independent.
