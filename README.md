# Locally Hosted Discord AI Bot

Host the bot and its AI models on your own hardware. Enable chat through a local model server such as LM Studio, image generation through ComfyUI, or both. It works in Discord channels, threads, and private DMs; chat conversations have separate personas and saved histories.

## Setup

Requires Python 3.12 and the local server for each enabled feature. Chat uses an OpenAI-compatible API; image analysis needs a vision-capable model, and web search needs tool calling. Discord and web features require an internet connection.

1. Create a Discord application and bot in the [Developer Portal](https://discord.com/developers/applications). Enable **Message Content Intent**. Invite it with the `bot` and `applications.commands` scopes.
2. Grant **View Channel**, **Send Messages**, and **Send Messages in Threads** where needed. Image generation also needs **Attach Files**. **Read Message History** is optional and enables fetching referenced messages. Check channel overrides as well as server permissions. Members need **Use Application Commands**; check **Server Settings → Integrations** if slash commands are missing.
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

Choose your mode in `.env`:

| Mode | `LLM_BASE_URL` | `COMFYUI_BASE_URL` |
| --- | --- | --- |
| Chat only | Chat server URL, including `/v1` | Blank |
| Images only | Blank | ComfyUI server URL |
| Both | Chat server URL, including `/v1` | ComfyUI server URL |

Set `LLM_MODEL_NAME` when using chat. To disable chat, explicitly set `LLM_BASE_URL=`; omitting it uses the localhost default. Chat-only needs no workflow or ComfyUI. Image-only needs no chat server. Start the bot:

```bash
python bot.py
```

Restart after changing configuration or code. Startup syncs the slash commands.

## Using the bot

For chat, mention or reply to the bot in a server, or open its profile and **Message** it privately. Bot DM slash commands require a mutual server. `/help` and `/status` reflect enabled features; commands for a disabled feature explain that it is unavailable.

| Command | Action |
| --- | --- |
| `/help` | Show usage instructions. |
| `/status` | Show models, supported inputs, limits, and this conversation's history count. |
| `/role` | View the current persona. Supply persona text to change it, or `clear` to reset it. Changes clear this conversation's history and post a public bot announcement in servers. |
| `/clear` | Clear this conversation's history, keeping its persona. Server access defaults to members with Manage Messages. |
| `/imagegen` | Enter width, height, and prompt in one form. |

Chat supports text/code, UTF-8 text files (including `message.txt`), text-based PDFs, public links, and images/supported stickers when vision is enabled. Attachments are limited to **10 MiB** each; PDFs to **15 pages**; extracted text to **40,000 characters** per document. Long text files produce a truncation notice. Source citations are requested only when you ask for them.

People in one channel share its conversation; each thread and user's DM is separate. At 100 saved messages, the oldest 50 are discarded. `/clear` and persona changes remove saved context, not Discord messages. Replying to an old message can supply its text again. Original attachment contents are not retained for later turns; reattach a file when needed.

## Image generation

1. Start ComfyUI and confirm your complete workflow generates an image. Install all models and custom nodes it requires. The bot needs a recent ComfyUI server with client-supplied prompt IDs, the jobs cancellation API, and `PreviewAny`.
2. Choose **File → Export (API)** in ComfyUI, then save the export as **`workflow.json` beside `bot.py`**. This is a local, ignored file and is not supplied in a clone. The [API export](https://github.com/Comfy-Org/ComfyUI/blob/master/script_examples/basic_api_example.py) contains node IDs with `class_type` and `inputs`; a normal editor export is not interchangeable.
3. Set `COMFYUI_BASE_URL` to the reachable server root, such as `http://localhost:8188`, without `/v1`. The bot discovers the workflow's node IDs automatically.
4. **When both features are enabled**, keep LM Studio 0.4+ running with its native model-management API and **Just-in-Time loading** enabled. Use the native model `key` for `LLM_MODEL_NAME`. Disable idle unloading if models should stay loaded until a switch. Skip this step for image-only use.
5. Restart the bot, check `/status`, and use `/imagegen`. Try `1024` × `1024` with a short prompt.

For ComfyUI on Windows and the bot in WSL, `localhost` must be reachable from WSL. Otherwise use the Windows host address and let ComfyUI listen on a reachable interface; see [WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking).

For a minimal template, see [examples/workflow.example.json](examples/workflow.example.json). It uses built-in nodes and a standard checkpoint containing the model, CLIP, and VAE. Copy it to `workflow.json`, replace `YOUR_CHECKPOINT.safetensors` with an installed compatible checkpoint, and choose sampler settings suitable for that model. The example uses `1024 × 1024`; the bot replaces its positive prompt and dimensions for each request.

GGUF workflows are supported with [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) installed in ComfyUI. Export a working workflow using `UnetLoaderGGUF` or `UnetLoaderGGUFAdvanced` and its compatible text encoder/VAE nodes. The bot preserves model filenames and loader settings; `/status` hides the `.gguf` or `.safetensors` extension.

Detection follows the connections leading to the saved image. **No node IDs need configuring.** The supported workflow has:

- One standard `SaveImage` output producing one image.
- One positive `CLIPTextEncode` encoder, including its SDXL, SDXLRefiner, SD3, or Flux variants. Text may be inline or supplied by `PrimitiveString`/`PrimitiveStringMultiline`.
- One `EmptyLatentImage`, `EmptySD3LatentImage`, or `EmptyFlux2LatentImage` for dimensions.
- A model loader on the sampler's model path: `UNETLoader`, `CheckpointLoaderSimple`, or an equivalent exposing `unet_name`/`ckpt_name`, used for `/status`.

Negative prompt text is preserved; zeroed negative conditioning is supported. SDXL conditioning and Flux2 scheduler size hints follow the requested dimensions. Multiple sampling stages may share one positive encoder and latent. BasicGuider and CFGGuider are supported; multi-prompt guiders are rejected. Missing, ambiguous, or unsupported inputs produce an error naming the expected node types before model switching. Custom nodes need recognizable connections; arbitrary workflows are not guaranteed. Detection runs on image requests and status, never during chat-only startup. The original file, sampler, steps, LoRAs, and scaling settings stay unchanged. The form accepts a prompt and dimensions, not additional reference images or masks.

All three form fields are required. Width and height accept positive whole numbers using digits `0–9`; invalid submissions are rejected before generation. The prompt is passed unchanged, up to **4,000 characters**. Dimensions are multiples of 16, with at least **1,048,576 total pixels** and at most **2048 per side**, preserving aspect ratio as closely as possible. For example, `512 × 512` becomes `1024 × 1024`, `1920 × 1080` becomes `1920 × 1088`, and `3840 × 2160` becomes `2048 × 1152`. Use a workflow/model compatible with these bounds. Its final output must match the selected dimensions; if its VAE doubles the size, keep a final 0.5 scaling node.

The result includes the original prompt and image as spoilers, with dimensions and generation time visible. The bot shows a queue message while waiting. The timeout is per submitted workflow; it excludes the bot's queue wait.

## Queues and local data

All servers and DMs share **three chat processing slots** and an image queue of **three total requests**, including the running image. Each user can have **one pending or running image request** across all servers and DMs. Images run one at a time; a fourth is declined. Additional chat turns wait, with each conversation processed in order. Chat activity declines new image requests, and image activity declines new chat requests. An idle model switches only when the other request type is accepted. Keep both model servers dedicated to the bot so it can coordinate GPU use.

Conversations are stored in `bot_database.db`; activity is logged to `bot.log`. Credentials, workflows, images, logs, and databases are ignored by Git. Keep other personal material in `private/` or `local/`; Git cannot detect personal content inside an otherwise tracked file.
