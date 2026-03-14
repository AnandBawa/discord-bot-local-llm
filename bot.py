import os
import io
import re
import json
import base64
import logging
import asyncio
import contextlib
from datetime import datetime

import fitz  # PyMuPDF
import aiohttp
import discord
import aiosqlite
from PIL import Image
from ddgs import DDGS
from openai import AsyncOpenAI
from dotenv import load_dotenv

# ==========================================
# ENVIRONMENT & API SETUP
# ==========================================
load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler() # Also prints to the terminal
    ]
)

LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')
LLM_API_KEY = os.getenv('LLM_API_KEY', 'lm-studio')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'local-model')

lm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ==========================================
# GLOBAL STATE & CONFIGURATION
# ==========================================
DB_FILE = "bot_database.db"

# --- MODEL & CONTEXT LIMITS ---
MAX_HISTORY_LENGTH = 50                             # Max messages kept in active chat history
MAX_TOOL_ITERATIONS = 3                             # Max consecutive tool calls per AI turn
LLM_TEMPERATURE = 1.0                               # Chat creativity (0.0 - 2.0)
LLM_MAX_TOKENS = 1000                               # Max tokens for standard responses
MEMORY_TEMPERATURE = 0.3                            # Low temp for strict memory extraction
MEMORY_MAX_TOKENS = 300                             # Max tokens for memory summaries

# --- HARDWARE & PARSING LIMITS ---
MAX_FILE_SIZE = 10 * 1024 * 1024                    # Max file size
MAX_PDF_PAGES = 15                                  # Max pages read from a single PDF
MAX_TEXT_EXTRACTION_LENGTH = 40000                  # Character limit for scraped text/PDFs
MAX_IMAGE_DIMENSION = 1024                          # Images resized to this max dimension
IMAGE_COMPRESSION_QUALITY = 85                      # Pillow JPEG quality (1-100)
SCRAPER_TIMEOUT = 15                                # Max seconds to wait for Jina Reader
WEB_SEARCH_MAX_RESULTS = 3                          # Number of DuckDuckGo snippets to pull

# --- DISCORD & SYSTEM LIMITS ---
DISCORD_CHUNK_LIMIT = 1980                          # Safe limit for Discord's 2000 char cap
CHUNK_MESSAGE_DELAY = 1.5                           # Wait time (seconds) between chunks
WIPE_REQUEST_EXPIRY = 3600                          # Seconds to hold a data-deletion lock
DEFAULT_PERSONA = "You are a neutral, conversational AI." 

# --- INTERNAL STATE (Do not modify) ---
highest_token_count = 0                             
background_tasks = set()                            
pending_deletions = {}                              
server_personas_cache = {}                          # RAM cache for server personas

# FIX: Initialized as None to prevent Python 3.8/3.9 event loop crashes
memory_lock = None                                  
llm_queue = None                                  

# ==========================================
# 1. CORE DATABASE & UTILITY FUNCTIONS
# ==========================================

async def init_db():
    """Initializes SQLite tables for configuration, history, and long-term memory."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('PRAGMA journal_mode=WAL;')
        await db.execute('''CREATE TABLE IF NOT EXISTS server_config (server_id TEXT PRIMARY KEY, prompt TEXT)''')
        await db.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, server_id TEXT, role TEXT, content TEXT, user_id TEXT, user_name TEXT)''')
        await db.execute('''CREATE TABLE IF NOT EXISTS core_memories (server_id TEXT, user_id TEXT, user_name TEXT, facts TEXT, PRIMARY KEY (server_id, user_id))''')
        await db.commit()

def register_deletion(key):
    """Registers a wipe request and prunes stale locks to prevent memory leaks."""
    current_time = datetime.now().timestamp()
    pending_deletions[key] = current_time
    
    stale_keys = [k for k, v in pending_deletions.items() if current_time - v > WIPE_REQUEST_EXPIRY]
    for k in stale_keys:
        del pending_deletions[k]

@contextlib.asynccontextmanager
async def safe_typing(channel):
    """Safely handles the typing indicator without crashing on permissions/network lag."""
    typing_ctx = channel.typing()
    success = False
    try:
        await typing_ctx.__aenter__()
        success = True
    except (discord.Forbidden, discord.HTTPException): 
        pass 
        
    try: 
        yield
    finally:
        if success:
            try: 
                await typing_ctx.__aexit__(None, None, None)
            except Exception: 
                pass

class URLImageAttachment:
    """Wrapper to seamlessly pass raw URL image bytes into Discord's attachment pipeline."""
    def __init__(self, data): 
        self.data = data
        
    async def read(self): 
        return self.data

# ==========================================
# 2. MEDIA PROCESSING FUNCTIONS
# ==========================================

def extract_pdf_text(pdf_bytes):
    """Safely extracts text from PDF bytes via PyMuPDF."""
    text = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                if i >= MAX_PDF_PAGES:
                    text += "\n...[Additional pages skipped to save memory]"
                    break
                text += page.get_text() + "\n"
        return text.strip()
    except Exception as e: 
        return f"Error reading PDF: {str(e)}"
    
def process_image_bytes(img_bytes):
    """Synchronous CPU-bound task for downscaling and compressing images."""
    with Image.open(io.BytesIO(img_bytes)) as pil_img:
        if pil_img.mode in ("RGBA", "P"): 
            pil_img = pil_img.convert("RGB")
            
        pil_img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=IMAGE_COMPRESSION_QUALITY)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
def process_sticker_bytes(sticker_bytes):
    """Synchronous base64 encoding for standard Discord stickers."""
    return base64.b64encode(sticker_bytes).decode('utf-8')

# ==========================================
# 3. AUTONOMOUS TOOLS & SCRAPING
# ==========================================

async def resolve_tool_error(error_message):
    return error_message

async def perform_web_search(query):
    """Uses DuckDuckGo to pull current facts and context snippets."""
    if not query or not isinstance(query, str): 
        return "Search error: Invalid or missing search query."
        
    now = datetime.now()
    current_date = now.strftime('%B %d, %Y')
    
    # Strip dates from the AI's query to prevent search engine confusion, then append today's date
    date_variations = [now.strftime('%B %d, %Y'), now.strftime('%B %d %Y'), f"{now.strftime('%B')} {now.day}, {now.strftime('%Y')}", f"{now.strftime('%B')} {now.day} {now.strftime('%Y')}", now.strftime('%B %d'), f"{now.strftime('%B')} {now.day}", now.strftime('%B %Y'), now.strftime('%Y')]
    clean_query = query
    for variation in date_variations: 
        clean_query = clean_query.replace(variation, "")
    optimized_query = f"{' '.join(clean_query.split())} {current_date}"

    logging.info(f"🔍 AI initiated web search for: '{optimized_query}'")
    try:
        results = await asyncio.to_thread(lambda: list(DDGS().text(optimized_query, max_results=WEB_SEARCH_MAX_RESULTS)))
        if not results: 
            return "No results."
            
        search_text = f"Date: {current_date}\n"
        for res in results: 
            search_text += f"[{res.get('title', 'No Title')}] {res.get('body', 'No Body')}\n"
        return search_text
    except Exception as e: 
        return f"Search error: {e}"
    
async def fetch_url_content(url):
    """Fetches a URL via Jina Reader to convert webpages directly to Markdown."""
    logging.info(f"📡 Jina attempting to fetch: {url}")
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with aiohttp.ClientSession() as session:
            jina_headers = {"X-Return-Format": "markdown", "X-No-Cache": "true"}
            async with session.get(jina_url, timeout=SCRAPER_TIMEOUT, headers=jina_headers) as response:
                if response.status != 200:
                    logging.warning(f"❌ Jina Fetch Failed (HTTP {response.status})")
                    return {"type": "error", "data": f"Failed to access URL (HTTP {response.status})"}
                
                # Pre-flight check on declared file size
                content_length = response.headers.get('Content-Length')
                if content_length and content_length.isdigit() and int(content_length) > MAX_FILE_SIZE:
                    response.close()
                    return {"type": "error", "data": "File skipped: Exceeds the 10MB limit."}
                
                # Strict download cap
                file_bytes = await response.content.read(MAX_FILE_SIZE + 1)
                if len(file_bytes) > MAX_FILE_SIZE:
                    response.close()
                    return {"type": "error", "data": "File skipped: Exceeds the 10MB limit."}
                
                content_type = response.headers.get('Content-Type', '').lower()
                
                # Route based on content type
                if 'image' in content_type:
                    return {"type": "image", "data": file_bytes}
                elif 'application/pdf' in content_type:
                    extracted_text = await asyncio.to_thread(extract_pdf_text, file_bytes)
                    if len(extracted_text) > MAX_TEXT_EXTRACTION_LENGTH: 
                        extracted_text = extracted_text[:MAX_TEXT_EXTRACTION_LENGTH] + "\n...[Content Truncated]"
                    return {"type": "text", "data": f"[Extracted PDF Document]:\n{extracted_text}"}
                elif 'text' in content_type or 'json' in content_type or 'markdown' in content_type or 'xml' in content_type:
                    try:
                        text = file_bytes.decode('utf-8', errors='replace') 
                        if len(text) > MAX_TEXT_EXTRACTION_LENGTH: 
                            text = text[:MAX_TEXT_EXTRACTION_LENGTH] + "\n...[Content Truncated]"
                        return {"type": "text", "data": text}
                    except Exception as decode_err:
                        return {"type": "error", "data": "Webpage content could not be decoded."}
                else:
                    # Gracefully reject binary formats like audio, video, or archives
                    return {"type": "error", "data": f"Unsupported URL media type ({content_type}). Please provide a standard webpage, image, or PDF."}

    except asyncio.TimeoutError:
        return {"type": "error", "data": "The website took too long to respond."}
    except Exception as e:
        return {"type": "error", "data": f"Unexpected scraper error: {str(e)}"}

# Define the JSON schema for the AI Tool
tools_schema = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Perform a web search to find current information, news, facts, OR to find more details for a follow-up question. NEVER guess or make up facts.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"]}
    }
}]

AVAILABLE_TOOLS = {"web_search": perform_web_search}

# ==========================================
# 4. MEMORY MANAGEMENT
# ==========================================

async def update_user_memory(server_id, user_id, user_name, forgotten_messages):
    """Summarizes chat history into permanent core facts when history cycles out."""
    task_start_time = datetime.now().timestamp()
    
    async with memory_lock:
        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT facts FROM core_memories WHERE server_id = ? AND user_id = ?", (server_id, str(user_id)))
            row = await cursor.fetchone()
            existing_memory = row[0] if row else "No core memories yet."
            
        # Compile forgotten history into a single string
        chat_log = ""
        for msg in forgotten_messages:
            raw_content = msg['content']
            try:
                parsed_content = json.loads(raw_content)
                if isinstance(parsed_content, list) and len(parsed_content) > 0 and isinstance(parsed_content[0], dict):
                    text_only = next((item.get("text", "[Text Missing]") for item in parsed_content if item.get("type") == "text"), "[Image data]")
                    raw_content = text_only
                else: 
                    raw_content = str(parsed_content)
            except (json.JSONDecodeError, TypeError): 
                pass
            chat_log += f"{msg['role'].capitalize()}: {raw_content}\n"
            
        memory_prompt = (f"You are an AI memory manager. Extract long-term, permanent facts about the user '{user_name}' from the chat log below. Update the existing memory with any new facts (preferences, tech stack, ongoing projects). Keep it strictly concise and bulleted. If there are no new facts about this user, just return the EXISTING MEMORY exactly as is. Do not include temporary conversational details.\n\nEXISTING MEMORY for {user_name}:\n{existing_memory}\n\nRECENT CHAT LOG:\n{chat_log}")
        
        try:
            async with llm_queue:
                response = await lm_client.chat.completions.create(
                    model=LLM_MODEL_NAME, 
                    messages=[{"role": "user", "content": memory_prompt}], 
                    temperature=MEMORY_TEMPERATURE, 
                    max_tokens=MEMORY_MAX_TOKENS
                )
            new_memory = response.choices[0].message.content.strip()
            
            # Save memory unless an admin issued a wipe command during generation
            if new_memory != existing_memory and new_memory:
                if pending_deletions.get(str(user_id), 0) > task_start_time or pending_deletions.get(f"wipe_{server_id}", 0) > task_start_time:
                    logging.info(f"[Memory] Aborted save for {user_name} due to mid-generation wipe request.")
                    return
                async with aiosqlite.connect(DB_FILE) as db:
                    await db.execute('''INSERT INTO core_memories (server_id, user_id, user_name, facts) VALUES (?, ?, ?, ?) ON CONFLICT(server_id, user_id) DO UPDATE SET facts=excluded.facts, user_name=excluded.user_name''', (server_id, str(user_id), user_name, new_memory))
                    await db.commit()
                logging.info(f"[Memory] Core memory updated for {user_name} in server {server_id}.")
        except Exception as e: 
            logging.error(f"Failed to update core memory for {user_name}: {e}")

async def process_memories_sequentially(server_id, users_dict, forgotten_messages):
    """Processes background memories sequentially to avoid API throttling."""
    for uid, uname in users_dict.items():
        await update_user_memory(server_id, uid, uname, forgotten_messages)
        await asyncio.sleep(CHUNK_MESSAGE_DELAY)

# ==========================================
# 5. PIPELINE MODULES
# ==========================================

async def handle_text_commands(message, clean_message, server_id):
    """Handles all Discord textual slash/admin commands."""
    global highest_token_count
    cmd = clean_message.lower()

    if cmd == 'help':
        help_text_main = f"""**How to interact with me:**

**General Chat**
• **`@{client.user.name} [message]`** - Chat, ask questions, or analyze attached files and links.

**Persona Management**
• **`@{client.user.name} role`** - View the active AI personality for this server.
• **`@{client.user.name} role [prompt]`** - Assign a new personality and start a fresh conversation.
• **`@{client.user.name} role clear`** - Restore the default neutral personality.

**Memory & Context**
• **`@{client.user.name} clear`** - Clear the current conversation history (core facts retained).
• **`@{client.user.name} memory`** - List all users who have saved core memories in this server.
• **`@{client.user.name} memory [name]`** - Read the permanent facts I have learned about a specific user.
• **`@{client.user.name} memory clear`** - Delete your own personal memories and chat history.

**System**
• **`@{client.user.name} status`** - Check bot health, latency, active model, and memory usage.
• **`@{client.user.name} help`** - Display this command list.
"""

        app_info = await client.application_info()
        if message.author.id == app_info.owner.id:
            help_text_main += f"""
**Bot Owner Commands:**
• **`@{client.user.name} force-forget [name]`** - (Admin) Purge all stored data for a specific user.
• **`@{client.user.name} wipe-server-memories`** - (Admin) Complete factory reset of all data for this server.
"""

        good_to_know_text = f"""**Feature Overview:**
> **Smart Memory:** I retain the last {MAX_HISTORY_LENGTH} messages. Important user facts are extracted before old messages are forgotten.
> **Autonomous Web Search:** I will search the internet to answer questions about current events or missing facts.
> **Media & URL Parsing:** Attach PDFs/images, or drop a URL in the chat. I will automatically scrape and analyze it.
> **Contextual Replies:** Reply to an old message and tag me to seamlessly resume that exact topic.
> **System Limits:** Files capped at {MAX_FILE_SIZE // (1024*1024)}MB. PDFs limited to {MAX_PDF_PAGES} pages. Web scraping capped at {MAX_TEXT_EXTRACTION_LENGTH} characters.
"""
        try:
            await message.reply(help_text_main.strip())
        except discord.HTTPException as http_exc:
            if http_exc.code == 50035: 
                await message.channel.send(f"<@{message.author.id}>\n{help_text_main.strip()}")
            else: 
                raise http_exc

        try:
            await message.channel.send(good_to_know_text.strip())
        except discord.Forbidden: 
            pass

        return True

    if cmd == 'role':
        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT prompt FROM server_config WHERE server_id = ?", (server_id,))
            row = await cursor.fetchone()
            current_role = row[0] if row and row[0] else None
            
        if current_role:
            await message.reply(f"**Current Server Persona:**\n> *{current_role}*")
        else:
            await message.reply(f"**Current Server Persona:**\n> *(Default) {DEFAULT_PERSONA}*")
        return True

    if cmd.startswith('role '):
        new_prompt = clean_message[5:].strip() 
        async with aiosqlite.connect(DB_FILE) as db:
            # Backup current memory before assigning new role
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
                server_personas_cache[server_id] = DEFAULT_PERSONA # Instantly update RAM cache
                await message.reply(f"✅ Server persona removed and history cleared! *(Recent memories saved)*\n\n**Current Persona:**\n> {DEFAULT_PERSONA}")
            else:
                if not new_prompt:
                    await message.reply(f"Please provide a prompt! Example: `@{client.user.name} role You are a pirate.`")
                    return True
                    
                await db.execute("INSERT INTO server_config (server_id, prompt) VALUES (?, ?) ON CONFLICT(server_id) DO UPDATE SET prompt=excluded.prompt", (server_id, new_prompt))
                await db.commit()
                server_personas_cache[server_id] = new_prompt # Instantly update RAM cache
                await message.reply(f"✅ Saved server persona and cleared history for a fresh start! *(Recent memories saved)*\n> *{new_prompt}*")
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
            
        await message.reply("🗑️ Server conversation history cleared! *(Recent memories saved)*")
        return True

    if cmd == 'status':
        ping_ms = round(client.latency * 1000)
        status_msg = await message.reply("Fetching diagnostics...")
        try:
            models_response = await lm_client.models.list()
            current_model = models_response.data[0].id if models_response.data else "None"
        except Exception: 
            current_model = f"Offline / Unreachable"
            
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
        
        command_parts = cmd.split()
        if len(command_parts) < 2:
            await message.reply(f"Please provide the name of the user you want to erase. Example: `@{client.user.name} force-forget john`")
            return True
            
        target_name = " ".join(command_parts[1:]).lower()
        
        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT DISTINCT user_id, user_name FROM chat_history WHERE server_id = ? AND user_id IS NOT NULL", (server_id,))
            history_users = await cursor.fetchall()
            
            cursor = await db.execute("SELECT DISTINCT user_id, user_name FROM core_memories WHERE server_id = ?", (server_id,))
            memory_users = await cursor.fetchall()
            
            all_users = {row[0]: row[1] for row in history_users + memory_users if row[1]}
            found_users = []
            
            for uid, uname in all_users.items():
                if target_name in uname.lower():
                    found_users.append({"id": uid, "name": uname})
                    
            if not found_users:
                await message.reply(f"⚠️ I couldn't find any saved data for a user matching '{target_name}'.")
                return True
            
            if len(found_users) > 1:
                names_list = "\n".join([f"• {u['name']}" for u in found_users])
                await message.reply(f"⚠️ Found multiple users matching '{target_name}'. Please be more specific:\n{names_list}")
                return True
                
            target_id = str(found_users[0]['id'])
            target_display_name = found_users[0]['name']
            
            register_deletion(target_id)
            await db.execute("DELETE FROM core_memories WHERE server_id = ? AND user_id = ?", (server_id, target_id))
            await db.execute("DELETE FROM chat_history WHERE server_id = ? AND user_id = ?", (server_id, target_id))
            await db.commit()
            
        await message.reply(f"🗑️ **Admin Override Executed.** All data for **{target_display_name}** has been permanently erased.")
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
                found_users = []
                
                if active_users:
                    placeholders = ','.join('?' * len(active_users))
                    params = [server_id] + active_users
                    cursor = await db.execute(f"SELECT user_name, facts FROM core_memories WHERE server_id = ? AND user_id IN ({placeholders})", params)
                    for row in await cursor.fetchall():
                        if target_name in row[0].lower():
                            found_users.append({"name": row[0], "facts": row[1]})
                
                if not found_users:
                    memory_text = f"I couldn't find any memories for '{target_name}'."
                elif len(found_users) == 1:
                    memory_text = f"**Facts known about {found_users[0]['name']}:**\n{found_users[0]['facts']}"
                else:
                    names_list = "\n".join([f"• {u['name']}" for u in found_users])
                    memory_text = f"⚠️ Found multiple users matching '{target_name}'. Please be more specific:\n{names_list}"
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
                if not found_any: 
                    memory_text += "*No core memories found for active users in this chat.*"
        
        # Output chunker logic
        remaining_text = memory_text
        is_first = True
        while len(remaining_text) > 0:
            split_index = remaining_text.rfind('\n', 0, DISCORD_CHUNK_LIMIT) if len(remaining_text) > DISCORD_CHUNK_LIMIT else len(remaining_text)
            
            if split_index == -1: 
                split_index = DISCORD_CHUNK_LIMIT
            elif split_index < len(remaining_text):
                split_index += 1 # <--- Swallow the newline!
                
            chunk = remaining_text[:split_index]
            remaining_text = remaining_text[split_index:]
            try:
                if is_first:
                    try: 
                        await message.reply(chunk)
                    except discord.HTTPException as http_exc:
                        if http_exc.code == 50035: 
                            await message.channel.send(f"<@{message.author.id}> {chunk}")
                    is_first = False
                else:
                    async with safe_typing(message.channel): 
                        await asyncio.sleep(CHUNK_MESSAGE_DELAY)
                    await message.channel.send(chunk)
            except discord.Forbidden: 
                break
        return True
    return False

async def extract_message_context(message, clean_message, user_name):
    """Filters attachments, extracts PDFs, parses replied messages, and scrapes URLs."""
    ephemeral_context = ""
    image_attachments = []
    
    # 1. Native Image Filtering & Unsupported File Detection
    for att in message.attachments:
        if att.content_type and att.content_type.startswith('image/'):
            if att.size <= MAX_FILE_SIZE: 
                image_attachments.append(att)
            else: 
                clean_message += f"\n[System note: Attached image '{att.filename}' ignored because it exceeds the limit.]"
        elif not att.filename.lower().endswith('.pdf'):
            # Catch everything that isn't an image or a PDF
            clean_message += f"\n[System note: The user attached an unsupported file type '{att.filename}'. Politely inform them that you can only read Images, PDFs, and Web Links.]"

    # 2. Native Sticker Filtering
    valid_stickers = [s for s in message.stickers if s.format != discord.StickerFormatType.lottie]
    
    # 3. PDF Extraction
    pdf_attachments = [att for att in message.attachments if att.filename.lower().endswith('.pdf') and att.size <= MAX_FILE_SIZE]
    if pdf_attachments:
        async with safe_typing(message.channel):
            for pdf in pdf_attachments:
                pdf_bytes = await pdf.read()
                pdf_text = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
                
                if len(pdf_text) > MAX_TEXT_EXTRACTION_LENGTH: 
                    pdf_text = pdf_text[:MAX_TEXT_EXTRACTION_LENGTH] + "\n...[Content Truncated due to length limit]"
                    
                logging.info(f"📎 AI successfully extracted UPLOADED PDF: {pdf.filename}")
                ephemeral_context += f"\n\n[Extracted PDF Content from {pdf.filename}]:\n{pdf_text}"
                clean_message += f"\n[System note: User attached PDF '{pdf.filename}']"

    # 4. Replied-to Context Retrieval
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
        except Exception as e: 
            logging.warning(f"⚠️ Could not fetch the replied message: {e}")

    # 5. URL Scraping
    url_pattern = r'(https?://[^\s<>]+)'
    found_urls = re.findall(url_pattern, clean_message)
    if found_urls:
        scraped_texts = []
        async with safe_typing(message.channel):
            for url in found_urls:
                url_result = await fetch_url_content(url)
                if url_result["type"] == "image": 
                    image_attachments.append(URLImageAttachment(url_result["data"]))
                elif url_result["type"] == "text": 
                    scraped_texts.append(f"\n\n[Extracted webpage content from {url}]:\n{url_result['data']}")
                elif url_result["type"] == "error": 
                    scraped_texts.append(f"\n\n[System note: Attempted to read {url} but failed: {url_result['data']}]")
                    
        if scraped_texts: 
            ephemeral_context += "".join(scraped_texts)

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
    
    # Inject user memories
    user_context_str = ""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute("SELECT DISTINCT user_id FROM chat_history WHERE server_id = ? AND user_id IS NOT NULL", (server_id,))
        active_user_ids = list(set([str(r[0]) for r in await cursor.fetchall()] + [author_id]))
        
        if active_user_ids:
            placeholders = ','.join('?' * len(active_user_ids))
            params = [server_id] + active_user_ids 
            cursor = await db.execute(f"SELECT user_name, facts FROM core_memories WHERE server_id = ? AND user_id IN ({placeholders})", params)
            for row in await cursor.fetchall():
                if row[1] != "No core memories yet.": 
                    user_context_str += f"- {row[0]}: {row[1]}\n"
                    
        # Check RAM cache first to avoid unnecessary disk I/O
        if server_id in server_personas_cache:
            base_persona = server_personas_cache[server_id]
        else:
            cursor = await db.execute("SELECT prompt FROM server_config WHERE server_id = ?", (server_id,))
            row = await cursor.fetchone()
            base_persona = row[0] if row and row[0] else DEFAULT_PERSONA
            server_personas_cache[server_id] = base_persona # Save to cache for next time

    if user_context_str: 
        current_system_prompt += f"\n\nCRITICAL CONTEXT ABOUT ACTIVE USERS (READ ONLY - DO NOT REPEAT):\n{user_context_str}"
        
    current_system_prompt += f"\n\nYOUR ASSIGNED PERSONA AND ROLE:\n{base_persona}"
    system_message = {"role": "system", "content": current_system_prompt}

    # Fetch and format chat history
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute("SELECT role, content FROM chat_history WHERE server_id = ? ORDER BY id ASC", (server_id,))
        api_history = []
        for r in await cursor.fetchall():
            try:
                parsed_content = json.loads(r[1])
                # CRITICAL: Memory Protection Filter
                # This ensures we don't pass massive base64 image strings from older messages
                # back into the context window, which would crash the AI due to VRAM limits.
                if isinstance(parsed_content, list):
                    if len(parsed_content) > 0 and not isinstance(parsed_content[0], dict): 
                        parsed_content = r[1]
                    else:
                        scrubbed_content = []
                        for item in parsed_content:
                            if item.get("type") == "image_url": 
                                scrubbed_content.append({"type": "text", "text": "[Image attached in previous message]"})
                            else: 
                                scrubbed_content.append(item)
                        parsed_content = scrubbed_content
                elif not isinstance(parsed_content, str): 
                    parsed_content = str(parsed_content)
            except (json.JSONDecodeError, TypeError): 
                parsed_content = r[1]
                
            api_history.append({"role": r[0], "content": parsed_content})
        
    # CRITICAL: API Compatibility Fix
    # Many LLMs crash if they receive multiple messages from the same role in a row.
    # This loop safely merges adjacent same-role messages into a single text block.
    cleaned_history = []
    for msg in api_history:
        if not cleaned_history:
            if msg["role"] == "assistant": 
                cleaned_history.append({"role": "user", "content": "[Conversation Started]"})
            cleaned_history.append(msg)
        else:
            if cleaned_history[-1]["role"] == msg["role"]:
                c1, c2 = cleaned_history[-1]["content"], msg["content"]
                if isinstance(c1, str) and isinstance(c2, str): 
                    cleaned_history[-1]["content"] = f"{c1}\n\n{c2}"
                else:
                    list1 = [{"type": "text", "text": c1}] if isinstance(c1, str) else c1.copy()
                    list2 = [{"type": "text", "text": f"\n\n{c2}"}] if isinstance(c2, str) else c2.copy()
                    cleaned_history[-1]["content"] = list1 + list2
            else: 
                cleaned_history.append(msg)
                
    # Merge the current user prompt into the end of the history array
    if cleaned_history and cleaned_history[-1]["role"] == "user":
        c1, c2 = cleaned_history[-1]["content"], api_user_content
        if isinstance(c1, str) and isinstance(c2, str): 
            cleaned_history[-1]["content"] = f"{c1}\n\n{c2}"
        else:
            list1 = [{"type": "text", "text": c1}] if isinstance(c1, str) else c1.copy()
            list2 = [{"type": "text", "text": f"\n\n{c2}"}] if isinstance(c2, str) else c2.copy()
            cleaned_history[-1]["content"] = list1 + list2
    else: 
        cleaned_history.append({"role": "user", "content": api_user_content})
        
    return [system_message] + cleaned_history

async def generate_ai_response(messages_to_send, message, disable_search, has_media):
    """Executes the LLM request, parses JSON schemas, and manages the iterative Web Search tool loop."""
    global highest_token_count
    
    api_kwargs = {
        "model": LLM_MODEL_NAME, 
        "messages": messages_to_send, 
        "temperature": LLM_TEMPERATURE, 
        "max_tokens": LLM_MAX_TOKENS
    }
    
    # 100% Guarantee: Hide the tool if explicitly disabled by a URL or PDF
    if not disable_search:
        api_kwargs["tools"] = tools_schema
        api_kwargs["tool_choice"] = "auto"

    async with safe_typing(message.channel):
        async with llm_queue:
            try:
                max_iterations, current_iteration, final_reply = MAX_TOOL_ITERATIONS, 0, ""
                
                response = await lm_client.chat.completions.create(**api_kwargs)
                
                if response.usage and response.usage.total_tokens > highest_token_count: 
                    highest_token_count = response.usage.total_tokens
                    
                response_message = response.choices[0].message
                
                # The Tool Iteration Loop
                while current_iteration < max_iterations:
                    if response_message.tool_calls:
                        msg_dump = response_message.model_dump(exclude_none=True)
                        if "content" not in msg_dump: 
                            msg_dump["content"] = "" 
                        messages_to_send.append(msg_dump)
                        
                        search_tasks, tool_call_metadata = [], []
                        for tool_call in response_message.tool_calls:
                            func_name = tool_call.function.name
                            if func_name in AVAILABLE_TOOLS:
                                try:
                                    args = json.loads(tool_call.function.arguments)
                                    if isinstance(args, dict): 
                                        search_tasks.append(AVAILABLE_TOOLS[func_name](**args))
                                    else: 
                                        search_tasks.append(resolve_tool_error("System error: Arguments must be JSON."))
                                    tool_call_metadata.append(tool_call)
                                except json.JSONDecodeError:
                                    search_tasks.append(resolve_tool_error("System error: Invalid JSON. DO NOT use tool."))
                                    tool_call_metadata.append(tool_call)
                            else:
                                search_tasks.append(resolve_tool_error(f"System error: Tool '{func_name}' does not exist."))
                                tool_call_metadata.append(tool_call)
                        
                        if search_tasks:
                            completed_results = await asyncio.gather(*search_tasks)
                            for tool_call, result_text in zip(tool_call_metadata, completed_results):
                                messages_to_send.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result_text})
                        
                        response = await lm_client.chat.completions.create(
                            model=LLM_MODEL_NAME, 
                            messages=messages_to_send, 
                            tools=tools_schema, 
                            tool_choice="auto", 
                            temperature=LLM_TEMPERATURE, 
                            max_tokens=LLM_MAX_TOKENS
                        )
                        
                        if response.usage and response.usage.total_tokens > highest_token_count: 
                            highest_token_count = response.usage.total_tokens
                            
                        response_message = response.choices[0].message
                        current_iteration += 1
                    else:
                        final_reply = response_message.content
                        break
                
                return final_reply or response_message.content or "⚠️ *System error: Empty response.*"
            except Exception as e:
                error_str = str(e).lower()
                logging.error(f"Generation Error: {e}")
                if has_media and ("400" in error_str or "vision" in error_str or "image" in error_str):
                    await message.reply("⚠️ **Compatibility Error:** Your local AI model does not support image analysis. Please use text only, or load a Multimodal Vision model.")
                else: 
                    await message.reply("Oops! I couldn't process that. Please check my terminal for details.")
                return None

async def save_and_send_response(message, server_id, user_name, db_user_content_obj, final_reply):
    """Saves atomic pairs to the DB, handles memory chunk eviction, and sends State-Aware Markdown messages to Discord."""
    async with aiosqlite.connect(DB_FILE) as db:
        db_user_content = json.dumps(db_user_content_obj) if isinstance(db_user_content_obj, list) else str(db_user_content_obj)
        await db.execute("INSERT INTO chat_history (server_id, role, content, user_id, user_name) VALUES (?, ?, ?, ?, ?)", (server_id, "user", db_user_content, str(message.author.id), user_name))
        await db.execute("INSERT INTO chat_history (server_id, role, content, user_id, user_name) VALUES (?, ?, ?, ?, ?)", (server_id, "assistant", final_reply, str(message.author.id), user_name))
        
        cursor = await db.execute("SELECT COUNT(*) FROM chat_history WHERE server_id = ?", (server_id,))
        
        # Trigger background memory summarization if capacity is reached
        if (await cursor.fetchone())[0] >= MAX_HISTORY_LENGTH:
            eviction_count = MAX_HISTORY_LENGTH // 2  
            cursor = await db.execute('''SELECT id, role, content, user_id, user_name FROM chat_history WHERE server_id = ? ORDER BY id ASC LIMIT ?''', (server_id, eviction_count))
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
    
    # ==========================================
    # MARKDOWN CHUNKER LOGIC
    # ==========================================
    remaining_text = final_reply
    is_first = True
    in_code_block = False
    
    while len(remaining_text) > 0:
        chunk_limit = DISCORD_CHUNK_LIMIT 
        
        if len(remaining_text) <= chunk_limit: 
            chunk = remaining_text
            remaining_text = ""
        else:
            # Try to safely split the message at the last new line or space
            split_index = remaining_text.rfind('\n', 0, chunk_limit)
            if split_index == -1: 
                split_index = remaining_text.rfind(' ', 0, chunk_limit)
            if split_index == -1: 
                split_index = chunk_limit
            else: 
                split_index += 1 
                
            chunk = remaining_text[:split_index]
            remaining_text = remaining_text[split_index:]

        # Tracks whether the current split happened inside a ` ``` ` markdown block,
        # ensuring the bot re-opens the block on the next split message so Discord formatting doesn't break.
        code_markers = chunk.count("```")
        if in_code_block: 
            chunk = "```\n" + chunk
        if code_markers % 2 != 0: 
            in_code_block = not in_code_block
        if in_code_block and len(remaining_text) > 0: 
            chunk += "\n```"
        
        try:
            if is_first:
                try: 
                    await message.reply(chunk)
                except discord.HTTPException as http_exc:
                    if http_exc.code == 50035: 
                        await message.channel.send(f"<@{message.author.id}> {chunk}")
                    else: 
                        raise http_exc
                is_first = False
            else:
                async with safe_typing(message.channel): 
                    await asyncio.sleep(CHUNK_MESSAGE_DELAY) 
                await message.channel.send(chunk)
        except discord.Forbidden: 
            break

# ==========================================
# 6. DISCORD EVENTS (The Clean Monolith)
# ==========================================

@client.event
async def on_ready():
    # FIX: Instantiate asyncio primitives inside the running event loop for Python 3.8/3.9 compatibility
    global memory_lock, llm_queue
    if memory_lock is None: 
        memory_lock = asyncio.Lock()
    if llm_queue is None: 
        llm_queue = asyncio.Semaphore(3)

    await init_db()
    logging.info(f'✅ Logged in successfully as {client.user}')
    logging.info('🌐 Database & Autonomous Web Search enabled!')

@client.event
async def on_message(message):
    # GUARANTEE: Ensure primitives exist even if a message arrives before on_ready fires
    global memory_lock, llm_queue
    if memory_lock is None: memory_lock = asyncio.Lock()
    if llm_queue is None: llm_queue = asyncio.Semaphore(3)
    
    if message.author == client.user or not client.user.mentioned_in(message) or not message.guild:
        return

    server_id = str(message.guild.id)
    bot_mention = f'<@{client.user.id}>'
    bot_nickname_mention = f'<@!{client.user.id}>' 
    clean_message = message.content.replace(bot_mention, '').replace(bot_nickname_mention, '').strip()
    user_name = f"{message.author.display_name}_{str(message.author.id)[-4:]}"

    # Smart logger: Checks if there is text, media, or just an empty ping
    log_content = clean_message if clean_message else ("[Media attached]" if message.attachments or message.stickers else "[Empty Ping / Help Trigger]")
    logging.info(f"{message.guild.name} | #{message.channel.name} | {message.author}: {log_content}")

    # Substitute tagged users into the text for the AI's contextual awareness
    for mentioned_user in message.mentions:
        if mentioned_user.id != client.user.id:
            memory_formatted_name = f"{mentioned_user.display_name}_{str(mentioned_user.id)[-4:]}"
            clean_message = clean_message.replace(f"<@{mentioned_user.id}>", f"@{memory_formatted_name}").replace(f"<@!{mentioned_user.id}>", f"@{memory_formatted_name}")

    # 1. Pipeline: Command Interception
    if await handle_text_commands(message, clean_message, server_id): 
        return

    # 2. Pipeline: Data Extraction (PDFs, URLs, Files)
    clean_message, image_attachments, valid_stickers, ephemeral_context = await extract_message_context(message, clean_message, user_name)
    if not clean_message.strip() and not image_attachments and not valid_stickers:
        await handle_text_commands(message, "help", server_id) 
        return

    # 3. Pipeline: Base64 Encoding & Payload Generation
    api_user_content, db_user_content_obj = await build_user_payloads(clean_message, ephemeral_context, image_attachments, valid_stickers, user_name)

    # 4. Pipeline: VRAM Scrubbing & Prompt Assembly
    messages_to_send = await build_ai_context(server_id, str(message.author.id), api_user_content)

    # 5. Pipeline: AI Generation & Tool Execution
    # Scans the raw message for any HTTP/HTTPS links to strictly disable web search
    disable_search = bool(re.search(r'(https?://[^\s<>]+)', clean_message)) or (ephemeral_context and "Extracted PDF Content" in ephemeral_context)
    has_media = bool(image_attachments or valid_stickers)
    final_reply = await generate_ai_response(messages_to_send, message, disable_search, has_media)

    # 6. Pipeline: Finalizing State
    if final_reply:
        await save_and_send_response(message, server_id, user_name, db_user_content_obj, final_reply)

# Intercept the default close method
original_close = client.close

async def graceful_shutdown():
    logging.info("Shutdown signal received. Waiting for background memory tasks to finish...")
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)
        logging.info("All background tasks completed. Memories safely saved.")
    logging.info("Disconnecting from Discord. Goodbye!")
    await original_close()

# Assign our custom shutdown sequence to the client
client.close = graceful_shutdown

if TOKEN: 
    client.run(TOKEN)