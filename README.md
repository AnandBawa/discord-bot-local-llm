# Discord AI Agent

A highly scalable, multimodal, and autonomous Discord AI bot. Designed to interface seamlessly with local LLMs (via LM Studio or similar OpenAI-compatible APIs), this agent features long-term memory management, autonomous web searching, document parsing, and image analysis capabilities.

## Key Features

- **Intelligent Memory Management:** Uses SQLite (`aiosqlite`) to maintain a rolling chat history and automatically summarizes older conversations into permanent "core memories" for each user, preventing LLM context window overflow.
- **Multimodal Capabilities:** Safely processes user-uploaded images and Discord stickers using PIL (Pillow), downscaling them to conserve VRAM. Unsupported file formats are elegantly intercepted, prompting the AI to politely request supported formats (Images/PDFs).
- **Autonomous Web Search:** Integrates the DuckDuckGo search engine (`ddgs`) as an automated tool. The AI can independently query the web to answer questions about current events or missing facts.
- **URL and Document Parsing:** Extracts text from uploaded PDF files using PyMuPDF (fitz) and seamlessly converts shared URLs into readable Markdown using the Jina Reader API (r.jina.ai). It features strict context-locking to prevent AI hallucinations when processing documents, and safely intercepts unsupported binary files (like audio or zip archives) to conserve bandwidth.
- **Markdown-Aware Message Chunking:** Safely bypasses Discord's 2,000-character limit by intelligently splitting long responses without breaking Markdown code blocks.
- **Custom Personas:** Administrators can define custom system prompts on a per-server basis.

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
   Create a `requirements.txt` file or run the following pip command directly. Note that `ddgs` is used for web searching:

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

```

4. **Run the Bot:**

```bash
python bot.py

```

## Bot Commands

The bot responds to natural language when @mentioned, but also supports the following specific text commands:

### General & Persona

- **`@BotName [message]`**: Chat or ask questions. The bot will automatically analyze any attached files or links.
- **`@BotName role`**: View the active AI personality for the server.
- **`@BotName role [prompt]`**: Assign a new personality to the bot and start a fresh conversation.
- **`@BotName role clear`**: Restore the bot to its default neutral personality.

### Memory Management

- **`@BotName clear`**: Clear the current conversation history (core facts are safely retained).
- **`@BotName memory`**: List all users who have saved core memories in the server.
- **`@BotName memory [name]`**: Read the permanent facts the AI has learned about a specific user.
- **`@BotName memory clear`**: Delete your own personal memories and chat history from the server.

### System & Administration

- **`@BotName status`**: Check bot latency, the active LLM model, peak token usage, and active memory capacity.
- **`@BotName force-forget [name]`**: _(Bot Owner Only)_ Purge all stored data for a specific user across the server.
- **`@BotName wipe-server-memories`**: _(Bot Owner Only)_ Complete factory reset of all core memories and chat history for the server.

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
