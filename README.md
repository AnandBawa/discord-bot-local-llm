# Discord Bot with Local LLM

A Discord bot powered by a locally hosted large language model through [LM Studio](https://lmstudio.ai/). It supports multi-user conversations, persistent per-user memory, autonomous web search, image analysis, and customizable personas -- all running entirely on your own hardware.

## Features

- **Local LLM Integration** -- Connects to any model served by LM Studio via its OpenAI-compatible API (`localhost:1234`).
- **Conversation History** -- Maintains the last 50 messages across the channel, providing context for ongoing discussions.
- **Core Memories** -- Automatically extracts and persists long-term facts about users (preferences, projects, tech stacks) before conversation history is cleared or trimmed.
- **Autonomous Web Search** -- Uses DuckDuckGo to search the web when current information is needed, via OpenAI-style tool calling.
- **Image and Sticker Analysis** -- Accepts image attachments and Discord stickers for multimodal analysis (requires a vision-capable model).
- **Custom Personas** -- Set a global system prompt to change the bot's personality on the fly.
- **Concurrency-Safe** -- Handles simultaneous messages with async locks and atomic history insertion to prevent data corruption.

## Prerequisites

- Python 3.9+
- [LM Studio](https://lmstudio.ai/) running locally with a model loaded and the local server started on port `1234`
- A Discord bot token (from the [Discord Developer Portal](https://discord.com/developers/applications))

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AnandBawa/discord-bot-local-llm.git
   cd discord-bot-local-llm
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv_bot
   source venv_bot/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install discord.py openai python-dotenv duckduckgo-search
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:

   ```
   DISCORD_BOT_TOKEN=your_discord_bot_token_here
   ```

5. **Start LM Studio** and load a model, then enable the local server (defaults to `http://localhost:1234`).

## Usage

```bash
python bot.py
```

Once running, mention the bot in any Discord channel it has access to:

| Command                  | Description                                              |
|--------------------------|----------------------------------------------------------|
| `@Bot [message]`         | Chat, ask questions, or attach images for analysis.      |
| `@Bot role`              | Show the current global personality.                     |
| `@Bot role [prompt]`     | Set a custom personality (clears history, saves memory). |
| `@Bot role clear`        | Reset to default neutral behavior.                       |
| `@Bot clear`             | Clear conversation history (saves memory first).         |
| `@Bot memory`            | List users with stored core memories.                    |
| `@Bot memory [name]`     | View stored facts about a specific user.                 |
| `@Bot status`            | Show latency, loaded model, and token usage.             |
| `@Bot help`              | Display the help menu.                                   |

You can also **reply** to any message and tag the bot to continue a specific thread of conversation.

## Project Structure

```
bot.py               # Main bot application (single-file)
.env                 # Discord bot token (not committed)
global_data.json     # Conversation history and active persona (auto-generated)
core_memories.json   # Persistent per-user memory store (auto-generated)
```

## How It Works

1. When a user mentions the bot, the message (with optional images) is sent to the local LLM along with conversation history and relevant user memories.
2. If the LLM determines it needs current information, it autonomously triggers a web search via DuckDuckGo and incorporates the results into its response.
3. When conversation history exceeds 50 messages, the oldest messages are trimmed and a background task extracts permanent facts about each user into `core_memories.json`.
4. These core memories are injected into the system prompt on every request, giving the bot long-term recall without unbounded context growth.

## License

[MIT](LICENSE)
