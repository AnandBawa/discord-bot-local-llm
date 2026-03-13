import discord
import asyncio 
import json
import os
import base64
import contextlib
import aiosqlite
import io
import re
import fitz
import aiohttp
from PIL import Image
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')
LLM_API_KEY = os.getenv('LLM_API_KEY', 'lm-studio')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'local-model')

lm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# --- GLOBAL STATE & CONFIG ---
DB_FILE = "bot_database.db"
MAX_HISTORY_LENGTH = 50 
MAX_FILE_SIZE = 10 * 1024 * 1024 # 10 Megabytes
highest_token_count = 0
memory_lock = asyncio.Lock()
background_tasks = set()
llm_queue = asyncio.Semaphore(3)
pending_deletions = {}

# ==========================================
# 1. CORE DATABASE & UTILITY FUNCTIONS
# ==========================================

async def init_db():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('PRAGMA journal_mode=WAL;')
        await db.execute('''CREATE TABLE IF NOT EXISTS server_config (server_id TEXT PRIMARY KEY, prompt TEXT)''')
        await db.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, server_id TEXT, role TEXT, content TEXT, user_id TEXT, user_name TEXT)''')
        await db.execute('''CREATE TABLE IF NOT EXISTS core_memories (server_id TEXT, user_id TEXT, user_name TEXT, facts TEXT, PRIMARY KEY (server_id, user_id))''')
        await db.commit()

def register_deletion(key):
    """Registers a wipe request and prunes old requests to prevent memory leaks."""
    current_time = datetime.now().timestamp()
    pending_deletions[key] = current_time
    stale_keys = [k for k, v in pending_deletions.items() if current_time - v > 3600]
    for k in stale_keys:
        del pending_deletions[k]

@contextlib.asynccontextmanager
async def safe_typing(channel):
    """Safely handles the typing indicator even if the bot lacks permissions or Discord is lagging."""
    typing_ctx = channel.typing()
    success = False
    try:
        await typing_ctx.__aenter__()
        success = True
    except (discord.Forbidden, discord.HTTPException): pass 
    try: yield
    finally:
        if success:
            try: await typing_ctx.__aexit__(None, None, None)
            except Exception: pass

class URLImageAttachment:
    """Wrapper to seamlessly pass raw URL image bytes into existing Discord attachment logic"""
    def __init__(self, data): self.data = data
    async def read(self): return self.data

# ==========================================
# 2. MEDIA PROCESSING FUNCTIONS
# ==========================================

def extract_pdf_text(pdf_bytes):
    """Safely extracts text from PDF bytes in a background thread."""
    text = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                if i > 14:
                    text += "\n...[Additional pages skipped to save memory]"
                    break
                text += page.get_text() + "\n"
        return text.strip()
    except Exception as e: return f"Error reading PDF: {str(e)}"
    
def process_image_bytes(img_bytes):
    """Synchronous, CPU-bound image resizing and base64 encoding."""
    with Image.open(io.BytesIO(img_bytes)) as pil_img:
        if pil_img.mode in ("RGBA", "P"): pil_img = pil_img.convert("RGB")
        pil_img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
def process_sticker_bytes(sticker_bytes):
    """Synchronous base64 encoding for stickers."""
    return base64.b64encode(sticker_bytes).decode('utf-8')

# ==========================================
# 3. AUTONOMOUS TOOLS & SCRAPING
# ==========================================

async def resolve_tool_error(error_message):
    return error_message

async def perform_web_search(query):
    if not query or not isinstance(query, str): return "Search error: Invalid or missing search query."
    now = datetime.now()
    current_date = now.strftime('%B %d, %Y')
    date_variations = [now.strftime('%B %d, %Y'), now.strftime('%B %d %Y'), f"{now.strftime('%B')} {now.day}, {now.strftime('%Y')}", f"{now.strftime('%B')} {now.day} {now.strftime('%Y')}", now.strftime('%B %d'), f"{now.strftime('%B')} {now.day}", now.strftime('%B %Y'), now.strftime('%Y')]
    
    clean_query = query
    for variation in date_variations: clean_query = clean_query.replace(variation, "")
    optimized_query = f"{' '.join(clean_query.split())} {current_date}"

    print(f"🔍 AI initiated web search for: '{optimized_query}'")
    try:
        results = await asyncio.to_thread(lambda: list(DDGS().text(optimized_query, max_results=3)))
        if not results: return "No results."
        search_text = f"Date: {current_date}\n"
        for res in results: search_text += f"[{res.get('title', 'No Title')}] {res.get('body', 'No Body')}\n"
        return search_text
    except Exception as e: return f"Search error: {e}"
    
async def fetch_url_content(url):
    """Fetches a URL via Jina Reader with full logging and safety limits preserved."""
    print(f"📡 Jina attempting to fetch: {url}")
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with aiohttp.ClientSession() as session:
            jina_headers = {"X-Return-Format": "markdown", "X-No-Cache": "true"}
            async with session.get(jina_url, timeout=15, headers=jina_headers) as response:
                if response.status != 200:
                    print(f"❌ Jina Fetch Failed (HTTP {response.status})")
                    return {"type": "error", "data": f"Failed to access URL via Jina (HTTP {response.status})"}
                
                content_length = response.headers.get('Content-Length')
                if content_length and content_length.isdigit() and int(content_length) > MAX_FILE_SIZE:
                    response.close()
                    print(f"🚫 File skipped (Header Size): {url}")
                    return {"type": "error", "data": "File skipped: Exceeds the 10MB size limit."}
                
                file_bytes = await response.content.read(MAX_FILE_SIZE + 1)
                if len(file_bytes) > MAX_FILE_SIZE:
                    response.close()
                    print(f"🚫 File skipped (Actual Size): {url}")
                    return {"type": "error", "data": "File skipped: Exceeds the 10MB size limit."}
                
                content_type = response.headers.get('Content-Type', '').lower()
                if 'image' in content_type:
                    print(f"🖼️ Image detected via URL: {url}")
                    return {"type": "image", "data": file_bytes}
                elif 'application/pdf' in content_type:
                    extracted_text = await asyncio.to_thread(extract_pdf_text, file_bytes)
                    if len(extracted_text) > 40000: extracted_text = extracted_text[:40000] + "\n...[Content Truncated]"
                    print(f"📄 PDF successfully extracted: {url}")
                    return {"type": "text", "data": f"[Extracted PDF Document]:\n{extracted_text}"}
                else:
                    try:
                        text = file_bytes.decode('utf-8', errors='replace') 
                        if len(text) > 40000: text = text[:40000] + "\n...[Content Truncated]"
                        print(f"🌐 Webpage successfully scraped: {url}")
                        return {"type": "text", "data": text}
                    except Exception as decode_err:
                        print(f"❌ Decoding failed for {url}: {decode_err}")
                        return {"type": "error", "data": "The webpage content could not be read properly (Decoding Error)."}

    except asyncio.TimeoutError:
        print(f"⏱️ Jina Timeout (15s exceeded): {url}")
        return {"type": "error", "data": "The website took too long to respond."}
    except Exception as e:
        print(f"🚨 Critical Jina Error: {e}")
        return {"type": "error", "data": f"Unexpected scraper error: {str(e)}"}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform a web search to find current information, news, facts, OR to find more details for a follow-up question. NEVER guess or make up facts; always search if you need deeper context.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query to look up on the internet."}}, "required": ["query"]}
        }
    }
]

AVAILABLE_TOOLS = {"web_search": perform_web_search}

# ==========================================
# 4. MEMORY MANAGEMENT
# ==========================================

async def update_user_memory(server_id, user_id, user_name, forgotten_messages):
    task_start_time = datetime.now().timestamp()
    async with memory_lock:
        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT facts FROM core_memories WHERE server_id = ? AND user_id = ?", (server_id, str(user_id)))
            row = await cursor.fetchone()
            existing_memory = row[0] if row else "No core memories yet."
            
        chat_log = ""
        for msg in forgotten_messages:
            raw_content = msg['content']
            try:
                parsed_content = json.loads(raw_content)
                if isinstance(parsed_content, list) and len(parsed_content) > 0 and isinstance(parsed_content[0], dict):
                    text_only = next((item.get("text", "[Text Missing]") for item in parsed_content if item.get("type") == "text"), "[Image data]")
                    raw_content = text_only
                else: raw_content = str(parsed_content)
            except (json.JSONDecodeError, TypeError): pass
            chat_log += f"{msg['role'].capitalize()}: {raw_content}\n"
            
        memory_prompt = (f"You are an AI memory manager. Extract long-term, permanent facts about the user '{user_name}' from the chat log below. Update the existing memory with any new facts (preferences, tech stack, ongoing projects). Keep it strictly concise and bulleted. If there are no new facts about this user, just return the EXISTING MEMORY exactly as is. Do not include temporary conversational details.\n\nEXISTING MEMORY for {user_name}:\n{existing_memory}\n\nRECENT CHAT LOG:\n{chat_log}")
        
        try:
            async with llm_queue:
                response = await lm_client.chat.completions.create(model=LLM_MODEL_NAME, messages=[{"role": "user", "content": memory_prompt}], temperature=0.3, max_tokens=300)
            new_memory = response.choices[0].message.content.strip()
            
            if new_memory != existing_memory and new_memory:
                if pending_deletions.get(str(user_id), 0) > task_start_time or pending_deletions.get(f"wipe_{server_id}", 0) > task_start_time:
                    print(f"[Memory] Aborted save for {user_name} due to mid-generation wipe request.")
                    return
                async with aiosqlite.connect(DB_FILE) as db:
                    await db.execute('''INSERT INTO core_memories (server_id, user_id, user_name, facts) VALUES (?, ?, ?, ?) ON CONFLICT(server_id, user_id) DO UPDATE SET facts=excluded.facts, user_name=excluded.user_name''', (server_id, str(user_id), user_name, new_memory))
                    await db.commit()
                print(f"[Memory] Core memory updated for {user_name} in server {server_id}.")
        except Exception as e: print(f"Failed to update core memory for {user_name}: {e}")

async def process_memories_sequentially(server_id, users_dict, forgotten_messages):
    for uid, uname in users_dict.items():
        await update_user_memory(server_id, uid, uname, forgotten_messages)
        await asyncio.sleep(1)

# ==========================================
# 5. PIPELINE MODULES (Refactored Logic)
# ==========================================

async def handle_text_commands(message, clean_message, server_id):
    """Handles all admin and user slash/text commands. Returns True if a command was intercepted."""
    global highest_token_count
    cmd = clean_message.lower()

    if cmd == 'help':
        help_text = (f"**Here is how you can interact with me:**\n\n• **`@{client.user.name} [message]`** - Tag me to ask a question, chat, or analyze an attached image/PDF.\n• **`@{client.user.name} role`** - Show my currently active server persona.\n• **`@{client.user.name} role [prompt]`** - Set a custom persona and wipe history (saves core memories first).\n• **`@{client.user.name} role clear`** - Reset me to my default neutral persona.\n• **`@{client.user.name} clear`** - Wipe the server conversation history (saves core memories first).\n• **`@{client.user.name} memory`** - View the list of users I have core memories for in this server.\n• **`@{client.user.name} memory [name]`** - View specific permanent facts learned about a user.\n• **`@{client.user.name} memory clear`** - Erase your own core memories and chat history from this server.\n• **`@{client.user.name} status`** - Show my latency, AI model, history capacity, and peak context usage.\n• **`@{client.user.name} help`** - Show this menu.\n\n")
        app_info = await client.application_info()
        if message.author.id == app_info.owner.id:
            help_text += ("**Bot Owner Commands:**\n" f"• **`@{client.user.name} force-forget [@user]`** - Permanently erase a specific user's data from this server's database.\n" f"• **`@{client.user.name} wipe-server-memories`** - Erase EVERYONE'S core memories and chat history for this specific server.\n\n")
        help_text += ("**Good to know:**\n> **Long-term Memory:** I remember the last 50 interactions in the server. Core facts are automatically extracted and saved before history is cleared.\n> **Contextual Replies:** You can reply to an old message and tag me, or reply to my messages to continue that specific thought.\n> **Live Web Search:** If you ask about current events or facts I don't know, I will search the web to find the answer.\n> **Media Support:** I can analyze images and PDFs you attach, or that are attached in messages you reply to.\n> **URL Analysis:** I can scrape a URL that you share. Support direct links to a PDF or an image.\n> **Size Limits:** Limit of 10MB for images/PDFs. For large PDFs, the first 15 pages only, or maximum of 40k characters are analyzed.")
        await message.reply(help_text)
        return True

    if cmd == 'role':
        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT prompt FROM server_config WHERE server_id = ?", (server_id,))
            row = await cursor.fetchone()
            current_role = row[0] if row and row[0] else None
        await message.reply(f"**Current Server Persona:**\n> *{current_role}*" if current_role else "**Current Server Persona:**\n> *(Default) You are a neutral, conversational AI.*")
        return True

    if cmd.startswith('role '):
        new_prompt = clean_message[5:].strip() 
        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT role, content, user_id, user_name FROM chat_history WHERE server_id = ? ORDER BY id ASC", (server_id,))
            rows = await cursor.fetchall()
            if rows:
                history_to_save = [{"role": r[0], "content": r[1], "user_id": r[2], "user_name": r[3]} for r in rows]
                users_in_history = {msg["user_id"]: msg["user_name"] for msg in history_to_save if msg.get("user_id")}
                task = asyncio.create_task(process_memories_sequentially(server_id, users_in_history, history_to_save))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
            await db.execute("DELETE FROM chat_history WHERE server_id = ?", (server_id,))
            
            if new_prompt.lower() == 'clear':
                await db.execute("INSERT INTO server_config (server_id, prompt) VALUES (?, ?) ON CONFLICT(server_id) DO UPDATE SET prompt=excluded.prompt", (server_id, ""))
                await db.commit()
                await message.reply(f"✅ Server persona removed and conversation history cleared! *(Recent memories were saved)*\n\n**Current Persona:**\n> You are a neutral, conversational AI.")
            else:
                if not new_prompt:
                    await message.reply(f"Please provide a prompt! Example: `@{client.user.name} role You are a pirate.`")
                    return True
                await db.execute("INSERT INTO server_config (server_id, prompt) VALUES (?, ?) ON CONFLICT(server_id) DO UPDATE SET prompt=excluded.prompt", (server_id, new_prompt))
                await db.commit()
                await message.reply(f"✅ Saved server persona and cleared conversation history for a fresh start! *(Recent memories were saved)*\n> *{new_prompt}*")
        return True

    if cmd == 'clear':
        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT role, content, user_id, user_name FROM chat_history WHERE server_id = ? ORDER BY id ASC", (server_id,))
            rows = await cursor.fetchall()
            if rows:
                history_to_save = [{"role": r[0], "content": r[1], "user_id": r[2], "user_name": r[3]} for r in rows]
                users_in_history = {msg["user_id"]: msg["user_name"] for msg in history_to_save if msg.get("user_id")}
                task = asyncio.create_task(process_memories_sequentially(server_id, users_in_history, history_to_save))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
            await db.execute("DELETE FROM chat_history WHERE server_id = ?", (server_id,))
            await db.commit()
        await message.reply("🗑️ Server conversation history cleared! *(Recent memories were saved)*")
        return True

    if cmd == 'status':
        ping_ms = round(client.latency * 1000)
        status_msg = await message.reply("Fetching diagnostics...")
        try:
            models_response = await lm_client.models.list()
            current_model = models_response.data[0].id if models_response.data else "None"
        except Exception: current_model = f"Offline / Unreachable"
        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM chat_history WHERE server_id = ?", (server_id,))
            history_length = (await cursor.fetchone())[0]
        diagnostics = (f"**Bot Diagnostics & Status**\n\n• **Discord Ping:** `{ping_ms}ms`\n• **Loaded AI Model:** `{current_model}`\n• **Peak Context Used:** `{highest_token_count} tokens`\n• **Current History Length:** `{history_length}/{MAX_HISTORY_LENGTH} messages`")
        await status_msg.edit(content=diagnostics)
        return True

    if cmd.startswith('force-forget'):
        app_info = await client.application_info()
        if message.author.id != app_info.owner.id:
            await message.reply("⛔ **Permission denied.** Only the bot owner can use this command.")
            return True
        target_user = next((u for u in message.mentions if u.id != client.user.id), None)
        if not target_user:
            await message.reply("Please mention the user you want to erase. Example: `@Bot force-forget @User`")
            return True
        target_id = str(target_user.id)
        register_deletion(target_id)
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute("DELETE FROM core_memories WHERE server_id = ? AND user_id = ?", (server_id, target_id))
            await db.execute("DELETE FROM chat_history WHERE server_id = ? AND user_id = ?", (server_id, target_id))
            await db.commit()
        await message.reply(f"🗑️ **Admin Override Executed.** All data for **{target_user.display_name}** has been permanently erased.")
        return True

    if cmd == 'wipe-server-memories':
        app_info = await client.application_info()
        if message.author.id != app_info.owner.id:
            await message.reply("⛔ **Permission denied.** Only the bot owner can use this command.")
            return True
        register_deletion(f"wipe_{server_id}")
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute("DELETE FROM core_memories WHERE server_id = ?", (server_id,))
            await db.execute("DELETE FROM chat_history WHERE server_id = ?", (server_id,))
            await db.commit()
        await message.reply("☢️ **SERVER WIPED.** All core memories and chat histories for **this specific server** have been erased.")
        return True

    if cmd.startswith('memory'):
        command_parts = cmd.split()
        if len(command_parts) == 2 and command_parts[1] == 'clear':
            register_deletion(str(message.author.id))
            async with aiosqlite.connect(DB_FILE) as db:
                await db.execute("DELETE FROM core_memories WHERE server_id = ? AND user_id = ?", (server_id, str(message.author.id)))
                await db.execute("DELETE FROM chat_history WHERE server_id = ? AND user_id = ?", (server_id, str(message.author.id)))
                await db.commit()
            await message.reply("🗑️ **Forget successful.** All your core memories and recent chat history have been erased.")
            return True

        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT DISTINCT user_id FROM chat_history WHERE server_id = ? AND user_id IS NOT NULL", (server_id,))
            active_users = [str(r[0]) for r in await cursor.fetchall()]
            active_users.append(str(message.author.id))
            active_users = list(set(active_users))
            
            if len(command_parts) > 1:
                target_name = " ".join(command_parts[1:]).lower()
                found_user = None
                if active_users:
                    placeholders = ','.join('?' * len(active_users))
                    params = [server_id] + active_users
                    cursor = await db.execute(f"SELECT user_name, facts FROM core_memories WHERE server_id = ? AND user_id IN ({placeholders})", params)
                    for row in await cursor.fetchall():
                        if target_name in row[0].lower():
                            found_user = {"name": row[0], "facts": row[1]}
                            break
                memory_text = f"**Facts known about {found_user['name']}:**\n{found_user['facts']}" if found_user else f"I couldn't find any memories for '{target_name}'."
            else:
                memory_text = f"**Tracked Active Users**\nType `@{client.user.name} memory [name]` to search.\nType `@{client.user.name} memory clear` to erase your own data.\n\n"
                found_any = False
                if active_users:
                    placeholders = ','.join('?' * len(active_users))
                    params = [server_id] + active_users
                    cursor = await db.execute(f"SELECT user_name, facts FROM core_memories WHERE server_id = ? AND user_id IN ({placeholders})", params)
                    for row in await cursor.fetchall():
                        if row[1] and row[1] != "No core memories yet.":
                            memory_text += f"• {row[0]}\n"
                            found_any = True
                if not found_any: memory_text += "*No core memories found for active users in this chat.*"
        
        # Simple chunker for memory output
        remaining_text = memory_text
        is_first = True
        while len(remaining_text) > 0:
            split_index = remaining_text.rfind('\n', 0, 1990) if len(remaining_text) > 1990 else len(remaining_text)
            if split_index == -1: split_index = 1990
            chunk = remaining_text[:split_index]; remaining_text = remaining_text[split_index:]
            try:
                if is_first:
                    try: await message.reply(chunk)
                    except discord.HTTPException as http_exc:
                        if http_exc.code == 50035: await message.channel.send(f"<@{message.author.id}> {chunk}")
                    is_first = False
                else:
                    async with safe_typing(message.channel): await asyncio.sleep(1.0)
                    await message.channel.send(chunk)
            except discord.Forbidden: break
        return True
    return False

async def extract_message_context(message, clean_message, user_name):
    """Filters attachments, extracts PDFs, fetches replies, and scrapes URLs."""
    ephemeral_context = ""
    image_attachments = []
    
    for att in message.attachments:
        if att.content_type and att.content_type.startswith('image/'):
            if att.size <= MAX_FILE_SIZE: image_attachments.append(att)
            else: clean_message += f"\n[System note: Attached image '{att.filename}' ignored because it exceeds the 10MB limit.]"

    valid_stickers = [s for s in message.stickers if s.format != discord.StickerFormatType.lottie]
    
    pdf_attachments = [att for att in message.attachments if att.filename.lower().endswith('.pdf') and att.size <= MAX_FILE_SIZE]
    if pdf_attachments:
        async with safe_typing(message.channel):
            for pdf in pdf_attachments:
                pdf_bytes = await pdf.read()
                pdf_text = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
                if len(pdf_text) > 40000: pdf_text = pdf_text[:40000] + "\n...[Content Truncated due to length limit]"
                print(f"📎 AI successfully extracted UPLOADED PDF: {pdf.filename}")
                ephemeral_context += f"\n\n[Extracted PDF Content from {pdf.filename}]:\n{pdf_text}"
                clean_message += f"\n[System note: User attached PDF '{pdf.filename}']"

    if message.reference and message.reference.message_id:
        try:
            replied_msg = await message.channel.fetch_message(message.reference.message_id)
            if replied_msg.content:
                replied_user_name = f"{replied_msg.author.display_name}_{str(replied_msg.author.id)[-4:]}"
                clean_message += f"\n\n[Context: {user_name} is replying to the following message by {replied_user_name}: \"{replied_msg.content}\"]"
                if replied_msg.author == client.user:
                    clean_message += "\n[System Directive: If you need more specific facts to answer this follow-up, you MUST output a web_search tool call. Do not guess.]"
            image_attachments.extend([att for att in replied_msg.attachments if att.content_type and att.content_type.startswith('image/')])
            valid_stickers.extend([s for s in replied_msg.stickers if s.format != discord.StickerFormatType.lottie])
        except Exception as e: print(f"⚠️ Could not fetch the replied message: {e}")

    url_pattern = r'(https?://[^\s<>]+)'
    found_urls = re.findall(url_pattern, clean_message)
    if found_urls:
        scraped_texts = []
        async with safe_typing(message.channel):
            for url in found_urls:
                url_result = await fetch_url_content(url)
                if url_result["type"] == "image": image_attachments.append(URLImageAttachment(url_result["data"]))
                elif url_result["type"] == "text": scraped_texts.append(f"\n\n[Extracted webpage content from {url}]:\n{url_result['data']}")
                elif url_result["type"] == "error": scraped_texts.append(f"\n\n[System note: Attempted to read {url} but failed: {url_result['data']}]")
        if scraped_texts: ephemeral_context += "".join(scraped_texts)

    return clean_message, image_attachments, valid_stickers, ephemeral_context

async def build_user_payloads(clean_message, ephemeral_context, image_attachments, valid_stickers, user_name):
    """Packages text and asynchronously encodes media into the DB and API payloads."""
    api_user_content = []
    db_user_content_obj = [] 
    
    ai_text_part = f"{user_name}: {clean_message}{ephemeral_context}" if (clean_message or ephemeral_context) else f"{user_name}: What is in this image?"
    db_text_part = f"{user_name}: {clean_message}" if clean_message else f"{user_name}: [Media attached]"

    if image_attachments or valid_stickers:
        api_user_content.append({"type": "text", "text": ai_text_part})
        db_user_content_obj.append({"type": "text", "text": db_text_part}) 
        
        for img in image_attachments:
            img_bytes = await img.read()
            img_b64 = await asyncio.to_thread(process_image_bytes, img_bytes)
            api_user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            db_user_content_obj.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            
        for sticker in valid_stickers:
            sticker_bytes = await sticker.read()
            sticker_b64 = await asyncio.to_thread(process_sticker_bytes, sticker_bytes)
            api_user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{sticker.format.name};base64,{sticker_b64}"}})
            db_user_content_obj.append({"type": "image_url", "image_url": {"url": f"data:image/{sticker.format.name};base64,{sticker_b64}"}})
    else: 
        api_user_content = ai_text_part
        db_user_content_obj = db_text_part 

    return api_user_content, db_user_content_obj

async def build_ai_context(server_id, author_id, api_user_content):
    """Compiles the system prompt, handles memory injection, and builds the strict VRAM-safe history array."""
    current_system_prompt = (f"Today's date is {datetime.now().strftime('%B %d, %Y')}.\nCRITICAL INSTRUCTIONS:\n1. EXTREME BREVITY: Answer in 1-3 sentences unless asked otherwise.\n2. DOCUMENT ANALYSIS: You will receive webpage and PDF data in Markdown format. Use headers (#), lists (*), and bold text within that data to identify key information accurately.\n3. SEARCH POLICY: Only use `web_search` if the provided URL content or attachments do not contain the answer. If a URL is provided, prioritize its content first.\n4. MULTI-USER CHAT: Address users by their names when appropriate.\n5. MEMORY USAGE (CRITICAL): Use user facts silently to guide your context. NEVER repeat or summarize these facts back to the user unless they explicitly ask what you remember about them.\n6. STRICT RULE: Do not use emojis unless your persona requires it.")
    
    user_context_str = ""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute("SELECT DISTINCT user_id FROM chat_history WHERE server_id = ? AND user_id IS NOT NULL", (server_id,))
        active_user_ids = list(set([str(r[0]) for r in await cursor.fetchall()] + [author_id]))
        
        if active_user_ids:
            placeholders = ','.join('?' * len(active_user_ids))
            params = [server_id] + active_user_ids 
            cursor = await db.execute(f"SELECT user_name, facts FROM core_memories WHERE server_id = ? AND user_id IN ({placeholders})", params)
            for row in await cursor.fetchall():
                if row[1] != "No core memories yet.": user_context_str += f"- {row[0]}: {row[1]}\n"
                    
        cursor = await db.execute("SELECT prompt FROM server_config WHERE server_id = ?", (server_id,))
        row = await cursor.fetchone()
        base_persona = row[0] if row and row[0] else "You are a neutral, conversational AI."

    if user_context_str: current_system_prompt += f"\n\nCRITICAL CONTEXT ABOUT ACTIVE USERS (READ ONLY - DO NOT REPEAT):\n{user_context_str}"
    current_system_prompt += f"\n\nYOUR ASSIGNED PERSONA AND ROLE:\n{base_persona}"
    system_message = {"role": "system", "content": current_system_prompt}

    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute("SELECT role, content FROM chat_history WHERE server_id = ? ORDER BY id ASC", (server_id,))
        api_history = []
        for r in await cursor.fetchall():
            try:
                parsed_content = json.loads(r[1])
                if isinstance(parsed_content, list):
                    if len(parsed_content) > 0 and not isinstance(parsed_content[0], dict): parsed_content = r[1]
                    else:
                        scrubbed_content = []
                        for item in parsed_content:
                            if item.get("type") == "image_url": scrubbed_content.append({"type": "text", "text": "[Image attached in previous message]"})
                            else: scrubbed_content.append(item)
                        parsed_content = scrubbed_content
                elif not isinstance(parsed_content, str): parsed_content = str(parsed_content)
            except (json.JSONDecodeError, TypeError): parsed_content = r[1]
            api_history.append({"role": r[0], "content": parsed_content})
        
    cleaned_history = []
    for msg in api_history:
        if not cleaned_history:
            if msg["role"] == "assistant": cleaned_history.append({"role": "user", "content": "[Conversation Started]"})
            cleaned_history.append(msg)
        else:
            if cleaned_history[-1]["role"] == msg["role"]:
                c1, c2 = cleaned_history[-1]["content"], msg["content"]
                if isinstance(c1, str) and isinstance(c2, str): cleaned_history[-1]["content"] = f"{c1}\n\n{c2}"
                else:
                    list1 = [{"type": "text", "text": c1}] if isinstance(c1, str) else c1.copy()
                    list2 = [{"type": "text", "text": f"\n\n{c2}"}] if isinstance(c2, str) else c2.copy()
                    cleaned_history[-1]["content"] = list1 + list2
            else: cleaned_history.append(msg)
                
    if cleaned_history and cleaned_history[-1]["role"] == "user":
        c1, c2 = cleaned_history[-1]["content"], api_user_content
        if isinstance(c1, str) and isinstance(c2, str): cleaned_history[-1]["content"] = f"{c1}\n\n{c2}"
        else:
            list1 = [{"type": "text", "text": c1}] if isinstance(c1, str) else c1.copy()
            list2 = [{"type": "text", "text": f"\n\n{c2}"}] if isinstance(c2, str) else c2.copy()
            cleaned_history[-1]["content"] = list1 + list2
    else: cleaned_history.append({"role": "user", "content": api_user_content})
        
    return [system_message] + cleaned_history

async def generate_ai_response(messages_to_send, message, has_scraped_data, has_media):
    """Executes the LLM request, handles Pydantic schemas, and manages the Tool Iteration Loop."""
    global highest_token_count
    
    api_kwargs = {"model": LLM_MODEL_NAME, "messages": messages_to_send, "temperature": 1, "max_tokens": 1000}
    if not has_scraped_data:
        api_kwargs["tools"] = tools_schema
        api_kwargs["tool_choice"] = "auto"

    async with safe_typing(message.channel):
        async with llm_queue:
            try:
                max_iterations, current_iteration, final_reply = 3, 0, ""
                
                response = await lm_client.chat.completions.create(**api_kwargs)
                if response.usage and response.usage.total_tokens > highest_token_count: highest_token_count = response.usage.total_tokens
                response_message = response.choices[0].message
                
                while current_iteration < max_iterations:
                    if response_message.tool_calls:
                        msg_dump = response_message.model_dump(exclude_none=True)
                        if "content" not in msg_dump: msg_dump["content"] = "" 
                        messages_to_send.append(msg_dump)
                        
                        search_tasks, tool_call_metadata = [], []
                        for tool_call in response_message.tool_calls:
                            func_name = tool_call.function.name
                            if func_name in AVAILABLE_TOOLS:
                                try:
                                    args = json.loads(tool_call.function.arguments)
                                    if isinstance(args, dict): search_tasks.append(AVAILABLE_TOOLS[func_name](**args))
                                    else: search_tasks.append(resolve_tool_error("System error: Arguments must be a JSON object. TOOL FAILED. DO NOT attempt to use the tool again. Answer directly."))
                                    tool_call_metadata.append(tool_call)
                                except json.JSONDecodeError:
                                    search_tasks.append(resolve_tool_error("System error: Invalid JSON. TOOL FAILED. DO NOT attempt to use the tool again. Answer directly."))
                                    tool_call_metadata.append(tool_call)
                            else:
                                search_tasks.append(resolve_tool_error(f"System error: Tool '{func_name}' does not exist. Do not use it."))
                                tool_call_metadata.append(tool_call)
                        
                        if search_tasks:
                            completed_results = await asyncio.gather(*search_tasks)
                            for tool_call, result_text in zip(tool_call_metadata, completed_results):
                                messages_to_send.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result_text})
                        
                        response = await lm_client.chat.completions.create(model=LLM_MODEL_NAME, messages=messages_to_send, tools=tools_schema, tool_choice="auto", temperature=1, max_tokens=1000)
                        if response.usage and response.usage.total_tokens > highest_token_count: highest_token_count = response.usage.total_tokens
                        response_message = response.choices[0].message
                        current_iteration += 1
                    else:
                        final_reply = response_message.content
                        break
                
                return final_reply or response_message.content or "⚠️ *System error: Reached max tool iterations or empty response.*"
            except Exception as e:
                error_str = str(e).lower()
                print(f"Generation Error: {e}")
                if has_media and ("400" in error_str or "vision" in error_str or "image" in error_str):
                    await message.reply("⚠️ **Compatibility Error:** The AI model currently loaded in my local server does not support image analysis. Please try again with text only, or load a multimodal/vision model!")
                else: await message.reply("Oops! I couldn't process that. Please check my terminal for more details.")
                return None

async def save_and_send_response(message, server_id, user_name, db_user_content_obj, final_reply):
    """Saves atomic pairs to the DB, handles chunked eviction, and sends State-Aware Markdown messages to Discord."""
    async with aiosqlite.connect(DB_FILE) as db:
        db_user_content = json.dumps(db_user_content_obj) if isinstance(db_user_content_obj, list) else str(db_user_content_obj)
        await db.execute("INSERT INTO chat_history (server_id, role, content, user_id, user_name) VALUES (?, ?, ?, ?, ?)", (server_id, "user", db_user_content, str(message.author.id), user_name))
        await db.execute("INSERT INTO chat_history (server_id, role, content, user_id, user_name) VALUES (?, ?, ?, ?, ?)", (server_id, "assistant", final_reply, str(message.author.id), user_name))
        
        cursor = await db.execute("SELECT COUNT(*) FROM chat_history WHERE server_id = ?", (server_id,))
        if (await cursor.fetchone())[0] >= 50:
            cursor = await db.execute('''SELECT id, role, content, user_id, user_name FROM chat_history WHERE server_id = ? ORDER BY id ASC LIMIT 20''', (server_id,))
            forgotten_rows = await cursor.fetchall()
            if forgotten_rows:
                forgotten_msgs = [{"role": r[1], "content": r[2], "user_id": r[3], "user_name": r[4]} for r in forgotten_rows]
                users_in_forgotten = {msg["user_id"]: msg["user_name"] for msg in forgotten_msgs if msg.get("user_id")}
                if users_in_forgotten:
                    task = asyncio.create_task(process_memories_sequentially(server_id, users_in_forgotten, forgotten_msgs))
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
                ids_to_delete = [r[0] for r in forgotten_rows]
                placeholders = ','.join('?' * len(ids_to_delete))
                await db.execute(f"DELETE FROM chat_history WHERE id IN ({placeholders})", ids_to_delete)
        await db.commit()
    
    remaining_text, is_first, in_code_block = final_reply, True, False
    while len(remaining_text) > 0:
        chunk_limit = 1980 
        if len(remaining_text) <= chunk_limit: chunk, remaining_text = remaining_text, ""
        else:
            split_index = remaining_text.rfind('\n', 0, chunk_limit)
            if split_index == -1: split_index = remaining_text.rfind(' ', 0, chunk_limit)
            if split_index == -1: split_index = chunk_limit
            else: split_index += 1 
            chunk, remaining_text = remaining_text[:split_index], remaining_text[split_index:]

        code_markers = chunk.count("```")
        if in_code_block: chunk = "```\n" + chunk
        if code_markers % 2 != 0: in_code_block = not in_code_block
        if in_code_block and len(remaining_text) > 0: chunk += "\n```"
        
        try:
            if is_first:
                try: await message.reply(chunk)
                except discord.HTTPException as http_exc:
                    if http_exc.code == 50035: await message.channel.send(f"<@{message.author.id}> {chunk}")
                    else: raise http_exc
                is_first = False
            else:
                async with safe_typing(message.channel): await asyncio.sleep(1.5) 
                await message.channel.send(chunk)
        except discord.Forbidden: break


# ==========================================
# 6. DISCORD EVENTS (The Clean Monolith)
# ==========================================

@client.event
async def on_ready():
    await init_db()
    print(f'✅ Logged in successfully as {client.user}')
    print('🌐 Database & Autonomous Web Search enabled!')

@client.event
async def on_message(message):
    if message.author == client.user or not client.user.mentioned_in(message) or not message.guild:
        return

    server_id = str(message.guild.id)
    bot_mention, bot_nickname_mention = f'<@{client.user.id}>', f'<@!{client.user.id}>' 
    clean_message = message.content.replace(bot_mention, '').replace(bot_nickname_mention, '').strip()
    user_name = f"{message.author.display_name}_{str(message.author.id)[-4:]}"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message.guild.name} | #{message.channel.name} | {message.author}: {(clean_message if clean_message else '[Media]')}...")

    for mentioned_user in message.mentions:
        if mentioned_user.id != client.user.id:
            memory_formatted_name = f"{mentioned_user.display_name}_{str(mentioned_user.id)[-4:]}"
            clean_message = clean_message.replace(f"<@{mentioned_user.id}>", f"@{memory_formatted_name}").replace(f"<@!{mentioned_user.id}>", f"@{memory_formatted_name}")

    # 1. Pipeline: Command Interception
    if await handle_text_commands(message, clean_message, server_id): return

    # 2. Pipeline: Data Extraction (PDFs, URLs, Files)
    clean_message, image_attachments, valid_stickers, ephemeral_context = await extract_message_context(message, clean_message, user_name)
    if not clean_message.strip() and not image_attachments and not valid_stickers:
        await handle_text_commands(message, "help", server_id) # Trigger help menu
        return

    # 3. Pipeline: Base64 Encoding & Payload Generation
    api_user_content, db_user_content_obj = await build_user_payloads(clean_message, ephemeral_context, image_attachments, valid_stickers, user_name)

    # 4. Pipeline: VRAM Scrubbing & Prompt Assembly
    messages_to_send = await build_ai_context(server_id, str(message.author.id), api_user_content)

    # 5. Pipeline: AI Generation & Tool Execution
    has_scraped_data = ephemeral_context and ("Extracted webpage content" in ephemeral_context or "Extracted PDF Content" in ephemeral_context)
    has_media = bool(image_attachments or valid_stickers)
    final_reply = await generate_ai_response(messages_to_send, message, has_scraped_data, has_media)

    # 6. Pipeline: Finalizing State
    if final_reply:
        await save_and_send_response(message, server_id, user_name, db_user_content_obj, final_reply)

if TOKEN: client.run(TOKEN)