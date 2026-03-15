# Discord AI Bot

A highly scalable, multimodal, and autonomous Discord AI bot. Designed to interface seamlessly with local LLMs (via LM Studio, Ollama, or similar OpenAI-compatible APIs), this agent features a robust dual-database architecture for long-term semantic memory (RAG), autonomous web searching, document parsing, native Slash Commands (`/`), image analysis capabilities, and an advanced failover routing system.

## Key Features

- **Dual-Database Memory Architecture (RAG):** Uses a highly optimized two-tier memory system. **SQLite** (`aiosqlite` in WAL mode) acts as the bot's short-term memory to maintain chronological chat history. **ChromaDB** acts as a permanent, searchable Vector Database. Older conversations are automatically distilled into core facts and mathematically retrieved (RAG) when contextually relevant.
- **Cloud Failover Routing:** Implements a strict fail-fast connection timeout. If your local LLM or Embedding node goes offline, the bot seamlessly routes requests to a configured cloud fallback API (like OpenAI or Jina) to ensure zero downtime.
- **Multimodal Capabilities (with Vision Toggle):** Safely processes user-uploaded images and Discord stickers using `PIL` (Pillow), downscaling them to conserve VRAM. Vision can be globally toggled on or off via environment variables. Unsupported files are elegantly intercepted.
- **Autonomous Web Search:** Integrates the DuckDuckGo search engine (`ddgs`) as an automated tool. The AI can independently query the web to answer questions about current events or missing facts.
- **URL and Document Parsing:** Extracts text from uploaded PDF files using PyMuPDF (`fitz`) and seamlessly converts shared URLs into readable Markdown using the Jina Reader API (`r.jina.ai`) with full browser spoofing to bypass firewalls.
- **Advanced Logging:** Dual-stream logging system saves complete tracebacks to `bot.log` while intelligently truncating long AI responses in the terminal to keep your screen clean.
- **Native Slash Commands:** Utilizes Discord's modern UI (`/commands`) for clean, spam-free interactions, memory management, and role-based access control.

## Prerequisites

- Python 3.8 or higher.
- A Discord Bot Token (with the **Message Content Intent** enabled in the Discord Developer Portal). When creating the OAuth2 URL for bot invite, select **bot** and **application.commands** under **Scopes**, and **View Channels** and **Send Messages** under **Bot Permissions**.
- An active LLM API endpoint (defaults to a local instance running on `http://localhost:1234/v1`).
- A Text Generation model and a separate Text Embedding model (e.g., `jina-embeddings-v5-text-small`) loaded in your local inference server.

## Installation

1. **Set up the project directory and virtual environment:**

```bash
python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate
```

2. **Install dependencies:**
   Run the following pip command to install the required libraries:

```bash
pip install discord.py openai python-dotenv aiosqlite PyMuPDF Pillow aiohttp ddgs chromadb httpx requests
```

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
EMB_MODEL_NAME=local-model
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

## Bot Commands

The bot features two distinct ways to interact: standard conversational tagging, and native Slash Commands (`/`).

### General Chat

- **`@BotName [message]`**: Chat or ask questions natively in the channel. The bot will automatically analyze any attached files or links.
- **Reply to the Bot**: Reply directly to one of the bot's messages and tag it to seamlessly continue an exact train of thought.

### Slash Commands (`/`)

- **`/help`**: Display the interactive guide and view current system limits (Ephemeral - only visible to you).
- **`/status`**: Check bot diagnostics, ping, active primary/fallback AI node status, and current chat history capacity.
- **`/role`**: View the active AI personality, or assign a new personality and start a fresh conversation. Type `clear` to restore the neutral default.
- **`/memory`**: Opens an interactive menu to list tracked users, read the permanent vector facts the AI has learned about a specific user from ChromaDB, or securely delete your own data.
- **`/clear`**: Clear the current server's temporary conversation history in SQLite (ChromaDB core facts are safely retained).
- **`/force-forget`**: _(Admin/Owner Only)_ Purge all stored data for a specific user across both SQLite and ChromaDB.
- **`/admin_wipe_server`**: _(Admin/Owner Only)_ Complete factory reset of all vector memories and chat history for the entire server.

## Advanced Configuration

Hardware limits, API parameters, and system behaviors are entirely modular. You can adjust constants directly in the `GLOBAL STATE & CONFIGURATION` section at the top of `bot.py` to match your specific hardware constraints. Key configurable parameters include:

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

- `MAX_FILE_SIZE` (Default: 10MB) - Hard limit for Discord attachments and Jina URL scraping.
- `MAX_PDF_PAGES` (Default: 15) - Maximum pages read from a PDF to prevent context window overflow.
- `MAX_TEXT_EXTRACTION_LENGTH` (Default: 40000) - Character limit for text extracted from URLs or PDFs.
- `MAX_IMAGE_DIMENSION` (Default: 1024) - Images are resized to this maximum width/height to save VRAM.
- `IMAGE_COMPRESSION_QUALITY` (Default: 85) - Pillow JPEG compression quality.
- `SCRAPER_TIMEOUT` (Default: 15) - Seconds to wait for web scraping or large native file downloads.
- `WEB_SEARCH_MAX_RESULTS` (Default: 3) - Number of search result snippets pulled from DuckDuckGo.

**Discord & System Limits:**

- `DISCORD_CHUNK_LIMIT` (Default: 1980) - Max character limit per Discord message chunk.
- `CHUNK_MESSAGE_DELAY` (Default: 1.5) - Seconds to wait between sending chunks to avoid rate limits.
- `WIPE_REQUEST_EXPIRY` (Default: 3600) - Seconds before a pending memory deletion request expires.
- `DEFAULT_PERSONA` - Fallback system prompt if no custom role is set for a server.
- `CIRCUIT_BREAKER_COOLDOWN` (Default: 60) - Seconds to automatically bypass the local node and route straight to the cloud fallback after a local failure is detected.
