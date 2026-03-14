# Discord AI Agent

A highly scalable, multimodal, and autonomous Discord AI bot. Designed to interface seamlessly with local LLMs (via LM Studio or similar OpenAI-compatible APIs), this agent features long-term memory management, autonomous web searching, document parsing, native Slash Commands (`/`), and image analysis capabilities.

## Key Features

- **Intelligent Memory Management:** Uses SQLite (`aiosqlite`) to maintain a rolling chat history and automatically summarizes older conversations into permanent "core memories" for each user, preventing LLM context window overflow.
- **Multimodal Capabilities:** Safely processes user-uploaded images and Discord stickers using `PIL` (Pillow), downscaling them to conserve VRAM. Unsupported file formats are elegantly intercepted, prompting the AI to politely request supported formats (Images/PDFs).
- **Autonomous Web Search:** Integrates the DuckDuckGo search engine (`ddgs`) as an automated tool. The AI can independently query the web to answer questions about current events or missing facts.
- **URL and Document Parsing:** Extracts text from uploaded PDF files using PyMuPDF (`fitz`) and seamlessly converts shared URLs into readable Markdown using the Jina Reader API (`r.jina.ai`). It features strict context-locking to prevent AI hallucinations when processing documents, and safely intercepts unsupported binary files (like audio or zip archives) to conserve bandwidth.
- **Native Slash Commands:** Utilizes Discord's modern UI (`/commands`) for clean, spam-free interactions and role-based access control.
- **Markdown-Aware Message Chunking:** Safely bypasses Discord's 2,000-character limit by intelligently splitting long responses without breaking Markdown code blocks.

## Prerequisites

- Python 3.8 or higher.
- A Discord Bot Token (with the **Message Content Intent** enabled in the Discord Developer Portal).
- An active LLM API endpoint (defaults to a local instance running on `http://localhost:1234/v1` but can be configured for cloud providers).

## Installation

1. **Set up the project directory and virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. **Install dependencies:**
   Create a `requirements.txt` file or run the following pip command directly:

```bash
pip install discord.py openai python-dotenv aiosqlite PyMuPDF Pillow aiohttp ddgs
```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and populate it with your credentials:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token_here
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=local-model
BOT_OWNER_ID=your_discord_user_id_here
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
- **`/status`**: Check bot diagnostics, ping, active AI model, and current chat history capacity.
- **`/role`**: View the active AI personality, or assign a new personality and start a fresh conversation. Type `clear` to restore the neutral default.
- **`/memory`**: Opens an interactive menu to list tracked users, read the permanent facts the AI has learned about a specific user, or securely delete your own data.
- **`/clear`**: Clear the current server's temporary conversation history (core facts are safely retained).
- **`/force-forget`**: _(Admin/Owner Only)_ Purge all stored date for a specific user.
- **`/admin_wipe_server`**: _(Admin/Owner Only)_ Complete factory reset of all core memories and chat history for the entire server.

## Advanced Configuration

Hardware limits, API parameters, and system behaviors are entirely modular. You can adjust constants directly in the `GLOBAL STATE & CONFIGURATION` section at the top of `bot.py` to match your specific hardware constraints. Key configurable parameters include:

**Model & Context Limits:**

- `MAX_HISTORY_LENGTH` (Default: 50) - Maximum messages kept in active chat history before summarization.
- `MAX_TOOL_ITERATIONS` (Default: 3) - Maximum consecutive tool calls (e.g., searches) the AI can make in one turn.
- `LLM_TEMPERATURE` (Default: 1.0) - Controls the creativity and randomness of standard chat responses.
- `LLM_MAX_TOKENS` (Default: 1000) - Maximum token length for standard chat responses.

**Hardware & Parsing Limits:**

- `MAX_FILE_SIZE` (Default: 10MB) - Hard limit for Discord attachments and Jina URL scraping.
- `MAX_PDF_PAGES` (Default: 15) - Maximum pages read from a PDF to prevent context window overflow.
- `MAX_TEXT_EXTRACTION_LENGTH` (Default: 40,000) - Character limit for text extracted from URLs or PDFs.
- `MAX_IMAGE_DIMENSION` (Default: 1024) - Images are resized to this maximum width/height to save VRAM.
- `IMAGE_COMPRESSION_QUALITY` (Default: 85) - Pillow JPEG compression quality.
- `WEB_SEARCH_MAX_RESULTS` (Default: 3) - Number of search result snippets pulled from DuckDuckGo.
