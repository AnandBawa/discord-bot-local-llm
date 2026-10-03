# Discord AI Bot

A Discord server bot with a shared conversation and persona, plus remembered facts associated with members. It connects to local OpenAI-compatible chat and embedding endpoints, with optional cloud fallback, web search, PDF extraction, and image analysis.

## Key Features

- **Shared Server Memory:** SQLite stores chronological conversation history and retains input awaiting memory extraction. ChromaDB stores extracted text facts for semantic retrieval. Failed extraction is retried, including after restart.
- **Cloud Fallback:** Tries local chat and embedding endpoints and can use configured cloud providers when requests fail.
- **Image Analysis:** Passes images and Discord stickers to a vision-capable chat model. Images use Pillow resizing; `VISION_ENABLED` controls whether visual input is sent to the model.
- **Autonomous Web Search:** Integrates the DuckDuckGo search engine (`ddgs`) as an automated tool. The AI can independently query the web to answer questions about current events or missing facts.
- **URL and Document Parsing:** Extracts text from uploaded PDF files using PyMuPDF (`pymupdf`) and converts public URLs into readable Markdown using the Jina Reader API (`r.jina.ai`). URL downloads reject internal addresses, including redirect destinations.
- **Logging:** Writes logs to `bot.log` and truncates long console messages. Logging limitations and retention concerns are recorded in the audit.
- **Slash Commands:** Provides commands for the shared persona, history, memory, and status. Members can change the server persona; force-forget and server-wipe commands require administrator or bot-owner access.

## Prerequisites

- Python 3.12. The dependency snapshot is verified on Python 3.12 on Linux.
- A Discord Bot Token (with the **Message Content Intent** enabled in the Discord Developer Portal). When creating the OAuth2 URL for bot invite, select **bot** and **application.commands** under **Scopes**, and **View Channels** and **Send Messages** under **Bot Permissions**.
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

## Dependency Verification and Audit

Run the isolated compatibility and regression checks:

```bash
python scripts/check_dependencies.py
python scripts/check_regressions.py
```

The check uses synthetic messages, mocked API responses, and temporary databases. It verifies command registration, SDK timeouts and tool calls, SQLite history, Chroma memory, image processing, and PDF extraction without loading `.env` or logging in to Discord.

The regression checks cover URL restrictions, forgetting during background work, and retaining failed extraction input. They use synthetic data, temporary databases, and a controlled local HTTP server; external socket connections are blocked.

History removed by pruning, `/clear`, or `/role` is retained in a `pending_memories` table until extraction succeeds. The bot retries in the background and after restart. Forget/wipe commands also remove this retained input and invalidate unfinished responses. The table is created automatically in the existing SQLite database.

The [audit and improvement plan](docs/AUDIT.md) records findings, accepted server-wide behavior, implemented fixes, and deferred improvements.

## Bot Commands

The bot features two distinct ways to interact: standard conversational tagging, and native Slash Commands (`/`).

### General Chat

- **`@BotName [message]`**: Chat or ask questions natively in the channel. The bot will automatically analyze any attached files or links.
- **Reply to the Bot**: Reply directly to one of the bot's messages and tag it to seamlessly continue an exact train of thought.

### Slash Commands (`/`)

- **`/help`**: Display the interactive guide and view current system limits (Ephemeral - only visible to you).
- **`/status`**: Check bot diagnostics, ping, active primary/fallback AI node status, and current chat history capacity.
- **`/role`**: View the shared persona, or change it and start a fresh server conversation. Extraction input from the previous conversation is retained until processing succeeds. Type `clear` to restore the neutral default.
- **`/memory`**: Opens an interactive menu to list tracked users, read the permanent vector facts the AI has learned about a specific user from ChromaDB, or securely delete your own data.
- **`/clear`**: Reset the visible server conversation and queue its text for memory extraction. Existing ChromaDB facts are retained.
- **`/force-forget`**: _(Admin/Owner Only)_ Delete a user's attributed history, retained extraction input, and ChromaDB facts in the current server.
- **`/admin_wipe_server`**: _(Admin/Owner Only)_ Delete vector memories, chat history, and retained extraction input for the current server. The stored persona is retained.

## Embedding Models

The local embedding model is selected by `EMB_MODEL_NAME`. Its model family, dimensionality, and preprocessing must match the API adapter when both providers share an index. The [audit's embedding note](docs/AUDIT.md#f04-embedding-fallback-can-mix-incompatible-vectors) records the task/preprocessing concern.

Jina v5 Omni Small adds image, audio, video, and PDF embeddings. Jina documents matching text embeddings between corresponding Text and Omni models, so a model-name change alone is not expected to improve this bot's text-fact retrieval. Using Omni's media capabilities would require indexing the attachments themselves and retaining their references. The bot continues to use Text Small. See [Jina's model documentation](https://jina.ai/embeddings/).

As checked on October 3, 2026, Omni is available through Jina's API. Full multimodal support in standard LM Studio has not been verified; Jina's [GGUF instructions](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval-GGUF#install-llamacpp-with-multimodal-patches) require a patched llama.cpp build. Task settings, dimensions, and the local quantization should be checked before treating providers as interchangeable.

## Advanced Configuration

You can adjust constants directly in the `GLOBAL STATE & CONFIGURATION` section of `bot.py`. Key settings include:

**Model & Context Limits:**

- `MAX_HISTORY_LENGTH` (Default: 100) - Maximum messages kept in active SQLite chat history before vector summarization.
- `MAX_TOOL_ITERATIONS` (Default: 3) - Maximum consecutive tool calls (e.g., searches) the AI can make in one turn.
- `LLM_TEMPERATURE` (Default: 1.0) - Controls the creativity and randomness of standard chat responses.
- `LLM_MAX_TOKENS` (Default: 4096) - Maximum token length for standard chat responses.
- `MEMORY_TEMPERATURE` (Default: 0.1) - Creativity for fact extraction (kept low to ensure strict factual JSON output).
- `MEMORY_MAX_TOKENS` (Default: 500) - Maximum token length when the AI is generating memory JSON arrays.
- `MEMORY_MAX_MSG_CHARS` (Default: 2000) - Max characters per message fed into the background memory extractor.
- `MEMORY_DEDUPLICATION_THRESHOLD` (Default: 0.15) - Strict cosine distance threshold used to prevent the bot from saving nearly identical facts into the database.

**Hardware & Parsing Limits:**

- `MAX_FILE_SIZE` (Default: 10MB) - Size limit used for direct attachment checks and URL downloads.
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
- `DEFAULT_PERSONA` - Fallback system prompt if no custom role is set for a server.
- `CIRCUIT_BREAKER_COOLDOWN` (Default: 60) - Seconds to automatically bypass the local node and route straight to the cloud fallback after a local failure is detected.
