# Discord AI Bot

A Discord server bot for chat and ComfyUI image generation. Recent conversations and server personas persist across restarts in SQLite. Chat uses a local OpenAI-compatible endpoint, with optional cloud fallback, web search, PDF extraction, and image analysis.

## Key Features

- **Saved Conversations:** SQLite keeps recent conversation text and the persona for each server across restarts. Channels within a server share that context; different servers stay separate. Attachment notes and assistant image descriptions persist; image bytes are sent only with the current request. No user profiles, fact extraction, semantic recall, or embedding model is used.
- **Ordered Conversations:** Chat turns run in arrival order within each server, including across channels. Different servers can progress concurrently, subject to the existing global limit of three chat LLM tasks.
- **Cloud Fallback:** A chat turn stays on its selected fallback throughout tool calls. Both chat SDK clients use a 2-second connection timeout, a 120-second read timeout, and no automatic SDK retries.
- **Image Analysis:** Passes images and supported Discord stickers to a vision-capable chat model. Images use Pillow resizing; `VISION_ENABLED` controls whether visual input is sent to the model.
- **Image Generation:** `/imagegen` collects dimensions and a prompt in one private form, then posts one image from the bundled Krea 2 ComfyUI workflow. The bot adjusts the size to the 1K–2K range described below and passes the prompt unchanged. While chat has work, image requests are declined; while images have work, chat is declined. Models unload only when switching between LM Studio and ComfyUI after the active work finishes.
- **Autonomous Web Search:** Uses `ddgs` to find missing information, including when an uploaded document is insufficient. Search terms and dates are preserved. Results include source URLs; the bot requests citations and appends up to three search source links if omitted from the answer.
- **URL and Document Parsing:** Extracts text from uploaded PDF files using PyMuPDF (`pymupdf`) and converts public URLs into readable Markdown using the Jina Reader API (`r.jina.ai`). URL downloads reject internal addresses, including redirect destinations.
- **Logging:** Writes logs to `bot.log` and truncates long console messages. Logging limitations and retention concerns are recorded in the audit.
- **Slash Commands:** `/help`, `/status`, `/role`, `/clear`, and `/imagegen`. Members can change the shared server persona; `/clear` defaults to members with Manage Messages, subject to the server's command settings.
- **Permission-Aware Replies:** Uses native Discord replies when permitted and ordinary messages mentioning the requester otherwise. Reply context uses content already delivered or cached; fetching older messages requires Read Message History.

## Prerequisites

- Python 3.12. The dependency snapshot is verified on Python 3.12 on Linux.
- A Discord Bot Token (with the **Message Content Intent** enabled in the Discord Developer Portal). When creating the OAuth2 URL for bot invite, select **bot** and **application.commands** under **Scopes**, and **View Channels** and **Send Messages** under **Bot Permissions**. For threads/posts, also grant **Send Messages in Threads**. **Read Message History** enables native replies and fetching referenced messages; the bot falls back to ordinary messages when it is missing. Effective permissions include channel overrides. See [Discord's message permissions](https://docs.discord.com/developers/resources/message#create-message).
- An active LLM API endpoint (defaults to a local instance running on `http://localhost:1234/v1`).
- A chat model available in your local inference server. Enable JIT loading in LM Studio when using image generation; no text embedding model is needed.

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

`requirements.txt` retains the application versions checked on October 3, 2026. The October 4 simplification removes ChromaDB and the direct `requests` dependency, leaving eight application packages and 31 resolved pins in `constraints.txt`. Installing this smaller manifest into an existing environment does not uninstall its unused packages; they are no longer imported by the bot.

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and populate it with your credentials:

```env
# Core Discord Setup

DISCORD_BOT_TOKEN=your_discord_bot_token_here

# Primary Local LLM Setup

LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=local-model
VISION_ENABLED=True

# Secondary Cloud Fallback Setup (Optional)

FALLBACK_BASE_URL=
FALLBACK_API_KEY=
FALLBACK_MODEL_NAME=

# ComfyUI Image Generation (Optional; blank disables it)
# Use the Windows server address reachable from WSL, without /v1.

COMFYUI_BASE_URL=
IMAGEGEN_TIMEOUT=600
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
python scripts/check_imagegen.py
python scripts/check_imagegen_ui.py
```

The checks use synthetic messages, mocked API responses, temporary storage, and blocked external sockets (with a controlled loopback HTTP fixture for URL tests). They do not load `.env`, log in to Discord, or use real conversations or model endpoints.

Coverage includes the five-command schema, SDK timeouts/tool calls, persisted history and personas across restart, legacy storage left unused, per-server history pruning and clearing, transaction rollback/cancellation, stale-turn invalidation, permission-aware delivery, search sources, media/PDF processing, and provider fallback. Image checks cover the form, size bounds, unchanged prompts, channel/thread permissions, queue limits, attachment delivery, model handoffs, busy refusals, cancellation, and model aliases. They do not establish real GPU release, live permissions, or model quality.

## Conversation Storage and Updating

The bot stores recent messages and personas in `bot_database.db`. A message can still refer to something a member said in the saved conversation; the bot no longer extracts separate facts about that member. On reaching 100 saved user/assistant messages in a server, it deletes the oldest 50. `/clear` deletes that server's conversation; changing `/role` does the same while saving the new persona. Deleted context is not archived or summarized.

Existing conversations and personas survive the update. Old fact tables and `chroma_storage` are left on disk but are no longer read, written, or used for replies. This update removes the feature, not that historical data. Existing logs are also retained. Old memory-related `.env` entries are ignored and can be removed: `EMB_MODEL_NAME`, `FALLBACK_EMB_API_KEY`, `MEMORY_DISTANCE_THRESHOLD`, and `BOT_OWNER_ID`.

Restart the bot to load the new code and sync the reduced command list; no re-invite is needed where slash commands already work. If an unused embedding model is still loaded in LM Studio, unload it manually once: the bot now manages only its configured chat model and refuses to unload unrelated models before image generation.

The [audit and improvement plan](docs/AUDIT.md) records findings, accepted server-wide behavior, implemented fixes, and deferred improvements.

## Bot Commands

The bot features two distinct ways to interact: standard conversational tagging, and native Slash Commands (`/`).

### General Chat

- **`@BotName [message]`**: Chat or ask questions in the channel. The bot can analyze supported attachments and public links.
- **Reply to the Bot**: A reply whose referenced message is delivered or cached is recognised without an extra tag. Tag the bot if the reference is unavailable. Without Read Message History, it can use its saved conversation but cannot fetch missing Discord messages or their attachments.

Unsupported file attachments and oversized images/PDFs are skipped, with a note passed to the chat model; remaining text and supported attachments are processed normally. The explanation to the user depends on the model, so there is no guaranteed rejection message. Unsupported animated stickers are silently skipped; a mention with only such a sticker receives the generic `/help` greeting.

### Slash Commands (`/`)

- **`/help`**: Display the command guide (Ephemeral - only visible to you).
- **`/status`**: Show ping, server history, chat and image models, supported inputs, and key limits. The chat line reflects the most recent successful provider across the bot and shows fallback details only after use. Before first use it shows the configured primary model. The image line reads the configured name from **#316 Load Diffusion Model** in `krea2.json`, omitting the `.safetensors` extension, or shows **Off** when image generation is disabled. If the workflow/model cannot be read, it shows **Configured (model unavailable)**. Status makes no provider requests or model loads; image analysis and web search need a compatible chat model.
- **`/imagegen`**: Enter width, height, and a prompt of up to 4000 characters in one form, then submit. The confirmation shows the chosen size and any adjustment; a queue message in the channel is replaced with the generated image and original prompt marked as Discord spoilers, with dimensions and generation time left visible. Available to all members when ComfyUI and channel permissions are configured; see setup below.
- **`/role`**: View the shared persona, or change it and start a fresh server conversation. Type `clear` to restore the neutral default.
- **`/clear`**: Delete this server's saved conversation, retaining its persona.

The per-server conversation queue covers context loading, generation, saving, and reply delivery. Slow requests delay later turns in that server. Clear and persona changes invalidate older running/queued turns before they save their answers. Already dispatched Discord messages are not retracted. The queue is in-process; waiting chat turns are not resumed after a restart.

## Image Generation Setup

1. Keep your working ComfyUI installation on Windows, with the models and custom nodes used by [krea2.json](krea2.json). Use a current ComfyUI server that supports client-supplied prompt IDs, `/api/jobs/{job_id}/cancel`, and the built-in `PreviewAny` node. The bot submits the API workflow directly; you do not need to queue it manually in the ComfyUI UI. See [ComfyUI's server API](https://docs.comfy.org/development/comfyui-server/comms_routes) and [job cancellation implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/server.py).
2. Set `COMFYUI_BASE_URL` in the bot's `.env` to the Windows server's root URL, for example `http://localhost:8188` **if that is its actual port and localhost is reachable from WSL**. Mirrored WSL networking supports Windows localhost; default NAT networking normally needs the Windows host IP and ComfyUI listening on a reachable interface. Use the same host arrangement that works for LM Studio, with ComfyUI's port. See [Microsoft's WSL networking guide](https://learn.microsoft.com/en-us/windows/wsl/networking).
3. Run LM Studio's API server as well. Image generation requires its native model-management API (LM Studio 0.4 or newer), in addition to the existing `/v1` endpoint. Enable **Just-in-Time model loading** so chat requests can reload the configured model after images. Disable **Idle TTL / automatic unloading** if you want it to stay loaded until a switch. The bot unloads its configured chat model before images and frees ComfyUI models before returning to LM Studio. See [LM Studio model unloading](https://lmstudio.ai/docs/developer/rest/unload) and [JIT loading and TTL](https://lmstudio.ai/docs/developer/core/ttl-and-auto-evict).
4. Grant the bot **Attach Files**, **View Channel**, and **Send Messages** in the destination channel, or **Send Messages in Threads** inside a thread/post. **Read Message History is not required.** Effective channel overrides apply. A server administrator must grant missing permissions. Restart the bot with `python bot.py` to register `/imagegen`.

For reliable reloads across bot restarts, use the chat model's native `key` from [LM Studio's `GET /api/v1/models`](https://lmstudio.ai/docs/developer/rest/list) for `LLM_MODEL_NAME`. A custom alias that is already loaded can be resolved while the bot runs, but that mapping is not saved across restarts. Keep your desired load settings saved in LM Studio for JIT loading.

Enter positive whole numbers for width and height; larger requests such as 3840 × 2160 are accepted. The bot uses the agreed **1K–2K** sizing rules: a minimum area of **1024 × 1024 = 1,048,576 pixels**, a maximum of **2048 pixels per side**, and dimensions in **multiples of 16**. Small requests scale up and large requests scale down, keeping size and aspect ratio as close as these bounds allow. The minimum is total area: a portrait or landscape image may have one side below 1024. Very wide or tall requests may need a different aspect ratio to satisfy both limits. The chosen dimensions, megapixel count, and any adjustment appear in the private confirmation after you submit the form.

| Requested size | Generation size | Total pixels |
| --- | --- | --- |
| 512 × 512 | 1024 × 1024 | 1,048,576 |
| 1536 × 1024 | 1536 × 1024 | 1,572,864 |
| 1920 × 1080 | 1920 × 1088 | 2,088,960 |
| 1920 × 1088 | 1920 × 1088 | 2,088,960 |
| 3840 × 2160 | 2048 × 1152 | 2,359,296 |
| 2048 × 2048 | 2048 × 2048 | 4,194,304 |
| 4096 × 4096 | 2048 × 2048 | 4,194,304 |

The former 2-megapixel cap has been removed: **2048 × 2048** is allowed and contains about **4.19 MP**. This square size appears in [Krea's official Turbo example](https://github.com/krea-ai/krea-2#usage). These are the bot's agreed sizing rules; no model or sampler settings are changed to enforce them.

The bot writes the unmodified prompt to **#48 Positive** and the calculated dimensions to **#232**, then reads the image from **#213 SaveImage**. The supplied 2× VAE decode is followed by **#324 ImageScaleBy at 0.5**, returning the decoded image to the selected size; the bot leaves these nodes unchanged. The workflow's model, LoRAs, sampler, and other generation settings remain in `krea2.json`. No chat-model request is used to rewrite the prompt or calculate the size. `VISION_ENABLED` affects image analysis, not generation.

For example, a 1920 × 1088 latent target is decoded to 3840 × 2176, then halved to a saved 1920 × 1088 image. The size limits apply to the selected generation/final size; the intermediate VAE image is larger. This is the intended behavior of the workflow's [Wan2.1 upscaling VAE](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x).

Chat and images share one activity rule across **all channels and servers**. Running or queued chat turns decline `/imagegen` with **“Chat is active right now. Image generation is unavailable. Please try again later.”** Running or queued images decline new chat with **“Image generation is active right now. Chat is unavailable. Please try again later.”** Declined requests are not queued or saved to chat history, and busy status does not trigger chat's cloud fallback. **Image generation uses ComfyUI only, with no cloud fallback.** Activity covers the complete accepted turn, including preparation, queue waits, model switching, and delivery. Opening an image form alone does not reserve the GPU; availability is checked again at submission. A loaded but idle model does not block the other request type.

Requests for the active type continue to queue. Images run one at a time, with at most **three accepted image requests total**, including the running image, across all servers and one per member. A fourth image request receives a queue-full message. The bot holds the waiting images and submits each to ComfyUI in turn. Chat retains **three chat processing slots**; additional chat requests wait, with no separate waiting-queue cap. Turns within a server run one at a time, including across channels; different servers can use the shared processing slots concurrently.

A local chat call already in progress blocks image admission, and cancellation retains that protection until the call finishes. When no work remains, the next accepted request can switch backends; repeated image or chat requests keep using their existing models. Keep these servers dedicated to the bot while it manages the shared GPU; independent manual requests cannot participate in its lock. The bot refuses to unload unrelated LM Studio models or interrupt unrelated ComfyUI jobs.

The form is private; the finished message posts the original prompt as spoiler text and the image as a spoiler attachment. Dimensions and generation time remain visible. All image uploads are marked as spoilers, including JPEG fallback, so viewers can reveal them in Discord. The bot strips workflow metadata from the uploaded image, sends PNG when it fits the server's attachment limit, and otherwise tries JPEG without reducing the selected dimensions. ComfyUI retains its own normal saved output. Image prompts/results are not added to the bot's conversation history.

Long prompts continue in additional spoilered messages to fit [Discord's 2,000-character message limit](https://docs.discord.com/developers/resources/message#create-message), without truncation. Markdown in the prompt is displayed literally so it cannot break the surrounding spoiler; prompt mentions do not ping anyone, and link previews are suppressed. If a continuation cannot be sent, the image and first prompt part remain posted, and the bot attempts to add a notice.

The finished message includes the workflow duration, for example `@member · 1920 × 1088 · Generated in 83.2s`. Timing starts when the bot submits the workflow to ComfyUI and ends when it detects completion. It includes any image-model loading performed by the workflow and completion-polling overhead, but excludes the bot's queue wait, the preceding model handoff, image download/conversion, and Discord upload.

`IMAGEGEN_TIMEOUT` defaults to 600 seconds per submitted workflow; waiting behind other image requests is separate. Timed-out or interrupted jobs are cancelled by their own ID. If ComfyUI cannot confirm a submission/cancellation, the bot blocks another GPU handoff rather than risking overlapping models. Check the ComfyUI queue and connectivity before retrying. Waiting image requests are not persisted across bot restarts. Leave `COMFYUI_BASE_URL` blank to disable this feature.

## Advanced Configuration

You can adjust constants directly in the `GLOBAL STATE & CONFIGURATION` section of `bot.py`. Key settings include:

**Model & Context Limits:**

- `MAX_HISTORY_LENGTH` (Default: 100) - At this many saved messages in a server, the oldest half are deleted.
- `MAX_TOOL_ITERATIONS` (Default: 3) - Maximum tool-call rounds in one turn; each round may contain multiple searches.
- `LLM_TEMPERATURE` (Default: 1.0) - Controls the creativity and randomness of standard chat responses.
- `LLM_MAX_TOKENS` (Default: 4096) - Maximum token length for standard chat responses.

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
- `DEFAULT_PERSONA` - Fallback system prompt if no custom role is set for a server.
- `CIRCUIT_BREAKER_COOLDOWN` (Default: 60) - Seconds to bypass a failed local chat service when its fallback is configured.
