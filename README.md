# Discord AI Bot

Chat and ComfyUI image generation in Discord channels, threads, and private DMs. Each conversation has its own persona and recent history, saved across restarts.

## Setup

Requires Python 3.12 and a running OpenAI-compatible chat endpoint. Image analysis needs a vision-capable model; web search needs tool calling.

1. Create a Discord application and bot in the [Developer Portal](https://discord.com/developers/applications). Enable **Message Content Intent**. Invite it with the `bot` and `applications.commands` scopes.
2. Grant **View Channel**, **Send Messages**, and **Send Messages in Threads** where needed. Image generation also needs **Attach Files**. **Read Message History** is optional and enables fetching referenced messages. Check channel overrides as well as server permissions.
3. From the repository directory, create the environment and install dependencies:

```bash
python3.12 -m venv venv_bot
source venv_bot/bin/activate
python -m pip install -r requirements.txt
```

These commands work in Linux/WSL. On Windows, activate with `venv_bot\Scripts\activate`.

4. Create `.env` beside `bot.py`:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=local-model
VISION_ENABLED=True
COMFYUI_BASE_URL=
IMAGEGEN_TIMEOUT=600
```

Set `LLM_MODEL_NAME` to your model's identifier. Leave `COMFYUI_BASE_URL` blank for chat only. Then start the bot:

```bash
python bot.py
```

Restart after changing configuration or code. Startup syncs the slash commands.

## Using the bot

Mention the bot or reply to it in a server. For a private conversation, open its profile, choose **Message**, and send a normal message. Bot DM slash commands require a mutual server.

| Command | Action |
| --- | --- |
| `/help` | Show usage instructions. |
| `/status` | Show models, supported inputs, limits, and this conversation's history count. |
| `/role` | View the current persona. Supply persona text to change it, or `clear` to reset it. Changes clear this conversation's history and require a public confirmation in servers. |
| `/clear` | Clear this conversation's history, keeping its persona. Server access defaults to members with Manage Messages. |
| `/imagegen` | Enter width, height, and prompt in one form. |

Chat supports text/code, UTF-8 text files (including `message.txt`), text-based PDFs, public links, and images/supported stickers when vision is enabled. Attachments are limited to **10 MiB** each; PDFs to **15 pages**; extracted text to **40,000 characters** per document. Long text files produce a truncation notice. Source citations are requested only when you ask for them.

People in one channel share its conversation; each thread and user's DM is separate. At 100 saved messages, the oldest 50 are discarded. `/clear` and persona changes remove saved context, not Discord messages. Replying to an old message can supply its text again. Original attachment contents are not retained for later turns; reattach a file when needed.

## Image generation

1. Start ComfyUI and confirm your complete workflow generates an image. Install all models and custom nodes it requires. The bot needs a recent ComfyUI server with client-supplied prompt IDs, the jobs cancellation API, and `PreviewAny`.
2. Choose **File → Export (API)** in ComfyUI, then save the export as **`workflow.json` beside `bot.py`**. This is a local, ignored file and is not supplied in a clone. The [API export](https://github.com/Comfy-Org/ComfyUI/blob/master/script_examples/basic_api_example.py) contains node IDs with `class_type` and `inputs`; a normal editor export is not interchangeable.
3. Check the node mappings below. Set `COMFYUI_BASE_URL` to the reachable server root, such as `http://localhost:8188`, without `/v1`.
4. For model switching, keep LM Studio 0.4+ running with its native model-management API and **Just-in-Time loading** enabled. Use the native model `key` for `LLM_MODEL_NAME`. Disable idle unloading if models should stay loaded until a switch.
5. Restart the bot, check `/status`, and use `/imagegen`. Try `1024` × `1024` with a short prompt.

For ComfyUI on Windows and the bot in WSL, `localhost` must be reachable from WSL. Otherwise use the Windows host address and let ComfyUI listen on a reachable interface; see [WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking).

The current adapter expects these mappings inside a complete, connected workflow:

| Node ID | Class | Bot input/output |
| --- | --- | --- |
| `48` | `PrimitiveStringMultiline` | Writes the prompt to `inputs.value`, connected to positive text encoding. |
| `232` | `EmptyLatentImage` | Sets `width`, `height`, and `batch_size` (1). |
| `213` | `SaveImage` | Reads exactly one final image. |
| `316` | `UNETLoader` | Reads `inputs.unet_name` for `/status`. |

A different node layout needs adapter changes in `bot.py`; the bot does not automatically discover arbitrary prompt, size, or output nodes. Model, sampler, steps, LoRAs, and seed remain controlled by the workflow. Workflows requiring extra inputs, such as a reference image or mask, need additional bot support.

The prompt is passed unchanged, up to **4,000 characters**. Dimensions are multiples of 16, with at least **1,048,576 total pixels** and at most **2048 per side**, preserving aspect ratio as closely as possible. For example, `512 × 512` becomes `1024 × 1024`, `1920 × 1080` becomes `1920 × 1088`, and `3840 × 2160` becomes `2048 × 1152`. Use a workflow/model compatible with these bounds. Its final output must match the selected dimensions; if its VAE doubles the size, keep a final 0.5 scaling node.

The result includes the original prompt and image as spoilers, with dimensions and generation time visible. The bot shows a queue message while waiting. The timeout is per submitted workflow; it excludes the bot's queue wait.

## Queues and local data

All servers and DMs share **three chat processing slots** and an image queue of **three total requests**, including the running image. Images run one at a time; a fourth is declined. Additional chat turns wait, with each conversation processed in order. Chat activity declines new image requests, and image activity declines new chat requests. An idle model switches only when the other request type is accepted. Keep both model servers dedicated to the bot so it can coordinate GPU use.

Conversations are stored in `bot_database.db`; activity is logged to `bot.log`. Credentials, workflows, images, logs, and databases are ignored by Git. Keep other personal material in `private/` or `local/`; Git cannot detect personal content inside an otherwise tracked file.

## Checks

```bash
python scripts/check_dependencies.py
python -m unittest discover -s scripts -p 'check_*.py'
```

Checks use synthetic inputs, temporary storage, and mocked services. Technical findings and validation details are in [the audit](docs/AUDIT.md) and [application review](docs/AUDIT-2026-10-05.md).
