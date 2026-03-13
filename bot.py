import discord
import asyncio 
import json
import os
import base64
import contextlib
import aiosqlite
import io
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

# --- DATA SETUP ---
DB_FILE = "bot_database.db"
MAX_HISTORY_LENGTH = 50 
highest_token_count = 0
memory_lock = asyncio.Lock()
background_tasks = set()
llm_queue = asyncio.Semaphore(3)
pending_deletions = {}

async def init_db():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('PRAGMA journal_mode=WAL;')
        
        await db.execute('''CREATE TABLE IF NOT EXISTS server_config
                            (server_id TEXT PRIMARY KEY, prompt TEXT)''')
        await db.execute('''CREATE TABLE IF NOT EXISTS chat_history
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                             server_id TEXT, role TEXT, content TEXT, 
                             user_id TEXT, user_name TEXT)''')
                             
        # FIX: Make core memories isolated to specific servers
        await db.execute('''CREATE TABLE IF NOT EXISTS core_memories
                            (server_id TEXT, user_id TEXT, user_name TEXT, facts TEXT,
                             PRIMARY KEY (server_id, user_id))''')
        await db.commit()

@contextlib.asynccontextmanager
async def safe_typing(channel):
    """Safely handles the typing indicator even if the bot lacks permissions or Discord is lagging."""
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

async def resolve_tool_error(error_message):
    return error_message

async def send_help_menu(message, bot_name):
    # 1. Base help text visible to EVERYONE
    help_text = (
        "**Here is how you can interact with me:**\n\n"
        f"• **`@{bot_name} [message]`** - Tag me to ask a question, chat, or analyze an attached image/sticker.\n"
        f"  Example: @{bot_name} what is the AQI in Delhi?\n\n"
        f"• **`@{bot_name} role`** - Show my currently active server personality.\n"
        f"• **`@{bot_name} role [prompt]`** - Set a custom personality and wipe history (saves core memories first).\n"
        f"  Example: @{bot_name} role you are a pirate.\n\n"
        f"• **`@{bot_name} role clear`** - Reset me to my default neutral behavior.\n"
        f"• **`@{bot_name} clear`** - Wipe the server conversation history (saves core memories first).\n"
        f"• **`@{bot_name} memory`** - View the list of users I have core memories for in this server.\n"
        f"• **`@{bot_name} memory [name]`** - View specific permanent facts learned about a person.\n"
        f"  Example: @{bot_name} memory Icy\n"
        f"• **`@{bot_name} memory clear`** - Permanently erase your own core memories and chat history from this server.\n\n"
        f"• **`@{bot_name} status`** - Show my latency, loaded AI model, history capacity, and token usage.\n"
        f"• **`@{bot_name} help`** - Show this menu.\n\n"
    )

    # 2. DYNAMIC CHECK: Inject secret commands ONLY if the user is the bot owner
    app_info = await client.application_info()
    if message.author.id == app_info.owner.id:
        help_text += (
            "**Bot Owner Commands:**\n"
            f"• **`@{bot_name} force-forget [@user]`** - Permanently erase a specific user's data from this server's database.\n"
            f"• **`@{bot_name} wipe-server-memories`** - Erase EVERYONE'S core memories and chat history for this specific server.\n\n"
        )

    # 3. Append the tips at the very end for everyone
    help_text += (
        "**Good to know:**\n"
        "> **Long-term Memory:** I remember the last 50 interactions in the channel. Core facts are automatically extracted and saved before history is cleared or naturally cycles out.\n"
        "> **Contextual Replies:** You can **reply** to an old message or image and tag me to continue that specific thought.\n"
        "> **Live Web Search:** If you ask about current events or facts I don't know, I will autonomously search the web to find the answer!"
    )
    
    await message.reply(help_text)

async def perform_web_search(query):
    if not query or not isinstance(query, str):
        return "Search error: Invalid or missing search query."
    
    now = datetime.now()
    current_date = now.strftime('%B %d, %Y')
    
    date_variations = [
        now.strftime('%B %d, %Y'), now.strftime('%B %d %Y'),
        f"{now.strftime('%B')} {now.day}, {now.strftime('%Y')}", f"{now.strftime('%B')} {now.day} {now.strftime('%Y')}",
        now.strftime('%B %d'), f"{now.strftime('%B')} {now.day}", now.strftime('%B %Y'), now.strftime('%Y')
    ]
    
    clean_query = query
    for variation in date_variations:
        clean_query = clean_query.replace(variation, "")
    optimized_query = f"{' '.join(clean_query.split())} {current_date}"

    print(f"🔍 AI initiated web search for: '{optimized_query}'")
    try:
        # FIX: Wrap the generator in list() so all network I/O happens in the background thread!
        results = await asyncio.to_thread(
            lambda: list(DDGS().text(optimized_query, max_results=3))
        )
            
        if not results:
            return "No results."
        
        search_text = f"Date: {current_date}\n"
        for res in results:
            search_text += f"[{res.get('title', 'No Title')}] {res.get('body', 'No Body')}\n"
        return search_text
    except Exception as e:
        return f"Search error: {e}"
    
# FIX: Add server_id parameter to memory functions
async def update_user_memory(server_id, user_id, user_name, forgotten_messages):
    task_start_time = datetime.now().timestamp()
    
    async with memory_lock:
        async with aiosqlite.connect(DB_FILE) as db:
            # Check for existing memory in THIS specific server
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
                else:
                    raw_content = str(parsed_content)
            except (json.JSONDecodeError, TypeError):
                pass
                
            chat_log += f"{msg['role'].capitalize()}: {raw_content}\n"
            
        memory_prompt = (
            f"You are an AI memory manager. Extract long-term, permanent facts about the user '{user_name}' from the chat log below. "
            "Update the existing memory with any new facts (preferences, tech stack, ongoing projects). "
            "Keep it strictly concise and bulleted. If there are no new facts about this user, just return the EXISTING MEMORY exactly as is. "
            "Do not include temporary conversational details.\n\n"
            f"EXISTING MEMORY for {user_name}:\n{existing_memory}\n\n"
            f"RECENT CHAT LOG:\n{chat_log}"
        )
        
        try:
            async with llm_queue:
                response = await lm_client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=[{"role": "user", "content": memory_prompt}],
                    temperature=0.3, 
                    max_tokens=300
                )
            
            new_memory = response.choices[0].message.content.strip()
            
            if new_memory != existing_memory and new_memory:
                # FIX: Check if either the specific user OR the entire server requested a wipe
                if pending_deletions.get(str(user_id), 0) > task_start_time or pending_deletions.get(f"wipe_{server_id}", 0) > task_start_time:
                    print(f"[Memory] Aborted save for {user_name} due to mid-generation wipe request.")
                    return
                
                async with aiosqlite.connect(DB_FILE) as db:
                    # FIX: Save the new memory bound to THIS server
                    await db.execute('''INSERT INTO core_memories (server_id, user_id, user_name, facts) 
                                        VALUES (?, ?, ?, ?) 
                                        ON CONFLICT(server_id, user_id) DO UPDATE SET facts=excluded.facts, user_name=excluded.user_name''', 
                                     (server_id, str(user_id), user_name, new_memory))
                    await db.commit()
                print(f"[Memory] Core memory updated for {user_name} in server {server_id}.")
        except Exception as e:
            print(f"Failed to update core memory for {user_name}: {e}")

async def process_memories_sequentially(server_id, users_dict, forgotten_messages):
    for uid, uname in users_dict.items():
        await update_user_memory(server_id, uid, uname, forgotten_messages)
        await asyncio.sleep(1)

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform a web search to find current information, news, facts, OR to find more details for a follow-up question. NEVER guess or make up facts; always search if you need deeper context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the internet."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

@client.event
async def on_ready():
    await init_db()
    print(f'✅ Logged in successfully as {client.user}')
    print('🌐 Database & Autonomous Web Search enabled!')

@client.event
async def on_message(message):
    global highest_token_count
    
    if message.author == client.user or not client.user.mentioned_in(message):
        return

    # IGNORE DMs: The bot must be used inside a Server (Guild) to track histories properly
    if not message.guild:
        return
        
    server_id = str(message.guild.id)

    # Stripping logic that handles both @Name and @Nickname formats
    bot_mention = f'<@{client.user.id}>'
    bot_nickname_mention = f'<@!{client.user.id}>' 
    
    clean_message = message.content.replace(bot_mention, '').replace(bot_nickname_mention, '').strip()
    user_name = f"{message.author.display_name}_{str(message.author.id)[-4:]}"

    # --- TERMINAL LOGGING ---
    log_content = clean_message if clean_message else ("[Image]" if message.attachments else "[Sticker]" if message.stickers else "")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message.guild.name} | #{message.channel.name} | {message.author}: {log_content[:50]}...")

    for mentioned_user in message.mentions:
        if mentioned_user.id != client.user.id:
            memory_formatted_name = f"{mentioned_user.display_name}_{str(mentioned_user.id)[-4:]}"
            clean_message = clean_message.replace(f"<@{mentioned_user.id}>", f"@{memory_formatted_name}")
            clean_message = clean_message.replace(f"<@!{mentioned_user.id}>", f"@{memory_formatted_name}")

    if clean_message.lower() == 'role':
        async with aiosqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT prompt FROM server_config WHERE server_id = ?", (server_id,))
            row = await cursor.fetchone()
            current_role = row[0] if row and row[0] else None

        if current_role:
            await message.reply(f"**Current Server Personality:**\n> *{current_role}*")
        else:
            await message.reply("**Current Server Personality:**\n> *(Default) You are a neutral, conversational AI.*")
        return

    if clean_message.lower().startswith('role '):
        new_prompt = clean_message[5:].strip() 
        if new_prompt.lower() == 'clear':
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
                await db.execute("INSERT INTO server_config (server_id, prompt) VALUES (?, ?) ON CONFLICT(server_id) DO UPDATE SET prompt=excluded.prompt", (server_id, ""))
                await db.commit()
            
            default_persona = "You are a neutral, conversational AI."
            await message.reply(
                f"✅ Server personality removed and conversation history cleared! *(Recent memories were saved)*\n\n"
                f"**Current Personality:**\n> {default_persona}"
            )
            return
            
        if not new_prompt:
            await message.reply(f"Please provide a prompt! Example: `@{client.user.name} role You are a pirate.`")
            return
            
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
            await db.execute("INSERT INTO server_config (server_id, prompt) VALUES (?, ?) ON CONFLICT(server_id) DO UPDATE SET prompt=excluded.prompt", (server_id, new_prompt))
            await db.commit()
            
        await message.reply(f"✅ Saved server personality and cleared conversation history for a fresh start! *(Recent memories were saved)*\n> *{new_prompt}*")
        return

    if clean_message.lower() == 'clear':
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
        return
    
    if clean_message.lower() == 'help':
        await send_help_menu(message, client.user.name)
        return
    
    if clean_message.lower() == 'status':
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
            
        diagnostics = (
            "**Bot Diagnostics & Status**\n\n"
            f"• **Discord Ping:** `{ping_ms}ms`\n"
            f"• **Loaded AI Model:** `{current_model}`\n"
            f"• **Peak Context Used:** `{highest_token_count} tokens`\n"
            f"• **Current History Length:** `{history_length}/{MAX_HISTORY_LENGTH} messages`"
        )
        await status_msg.edit(content=diagnostics)
        return
    
    if clean_message.lower().startswith('force-forget'):
        # 1. Verify the user is the official Bot Owner
        app_info = await client.application_info()
        if message.author.id != app_info.owner.id:
            await message.reply("⛔ **Permission denied.** Only the bot owner can use this command.")
            return
            
        # 2. Extract the mentioned user (ignoring the bot itself)
        target_user = next((u for u in message.mentions if u.id != client.user.id), None)
        if not target_user:
            await message.reply("Please mention the user you want to erase. Example: `@Bot force-forget @User`")
            return
            
        target_id = str(target_user.id)
        
        # 3. Apply the timestamp shield to prevent background tasks from reviving them
        pending_deletions[target_id] = datetime.now().timestamp()
        
        # 4. Shred their data
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute("DELETE FROM core_memories WHERE server_id = ? AND user_id = ?", (server_id, target_id))
            await db.execute("DELETE FROM chat_history WHERE server_id = ? AND user_id = ?", (server_id, target_id))
            await db.commit()
            
        await message.reply(f"🗑️ **Admin Override Executed.** All data for **{target_user.display_name}** has been permanently erased from the database.")
        return
    
    # --- ADMIN: WIPE ENTIRE SERVER DATABASE ---
    if clean_message.lower() == 'wipe-server-memories':
        # 1. Verify the user is the official Bot Owner
        app_info = await client.application_info()
        if message.author.id != app_info.owner.id:
            await message.reply("⛔ **Permission denied.** Only the bot owner can use this command.")
            return
            
        # 2. Shield THIS server from any currently running background tasks reviving data
        pending_deletions[f"wipe_{server_id}"] = datetime.now().timestamp()
        
        # 3. Shred everything strictly for THIS server (using server_id)
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute("DELETE FROM core_memories WHERE server_id = ?", (server_id,))
            await db.execute("DELETE FROM chat_history WHERE server_id = ?", (server_id,))
            await db.commit()
            
        await message.reply("☢️ **SERVER WIPED.** All core memories and chat histories for **this specific server** have been permanently erased. Data in other servers remains safe.")
        return
    
    if clean_message.lower().startswith('memory'):
        command_parts = clean_message.split()
        
        # NEW FEATURE: User-controlled memory deletion
        if len(command_parts) == 2 and command_parts[1].lower() == 'clear':
            
            # FIX: Record exactly when the user requested the wipe
            pending_deletions[str(message.author.id)] = datetime.now().timestamp()
            
            async with aiosqlite.connect(DB_FILE) as db:
                await db.execute("DELETE FROM core_memories WHERE server_id = ? AND user_id = ?", (server_id, str(message.author.id)))
                await db.execute("DELETE FROM chat_history WHERE server_id = ? AND user_id = ?", (server_id, str(message.author.id)))
                await db.commit()
            await message.reply("🗑️ **Forget successful.** All your core memories and recent chat history have been permanently erased from my database.")
            return

        async with aiosqlite.connect(DB_FILE) as db:
            # Build the active users list for the current server
            cursor = await db.execute("SELECT DISTINCT user_id FROM chat_history WHERE server_id = ? AND user_id IS NOT NULL", (server_id,))
            active_users = [str(r[0]) for r in await cursor.fetchall()]
            active_users.append(str(message.author.id))
            active_users = list(set(active_users))
            
            if len(command_parts) > 1:
                target_name = " ".join(command_parts[1:]).lower()
                found_user = None
                
                if active_users:
                    placeholders = ','.join('?' * len(active_users))
                    
                    # FIX: Make the specific user memory search strictly isolated to this server
                    params = [server_id] + active_users
                    cursor = await db.execute(f"SELECT user_name, facts FROM core_memories WHERE server_id = ? AND user_id IN ({placeholders})", params)
                    
                    rows = await cursor.fetchall()
                    for row in rows:
                        if target_name in row[0].lower():
                            found_user = {"name": row[0], "facts": row[1]}
                            break
                        
                if found_user:
                    memory_text = f"**Facts known about {found_user['name']}:**\n{found_user['facts']}"
                else:
                    memory_text = f"I couldn't find any memories for '{target_name}' in this server's recent chat history."
            else:
                memory_text = f"**Tracked Active Users**\nType `@{client.user.name} memory [name]` to search all known users.\nType `@{client.user.name} memory clear` to erase your own data.\n\n"
                found_any = False
                
                if active_users:
                    placeholders = ','.join('?' * len(active_users))
                    
                    # FIX: Make the global memory list strictly isolated to this server
                    params = [server_id] + active_users
                    cursor = await db.execute(f"SELECT user_name, facts FROM core_memories WHERE server_id = ? AND user_id IN ({placeholders})", params)
                    
                    rows = await cursor.fetchall()
                    for row in rows:
                        if row[1] and row[1] != "No core memories yet.":
                            memory_text += f"• {row[0]}\n"
                            found_any = True
                            
                if not found_any:
                    memory_text += "*No core memories found for active users in this chat.*"
        
        chunk_size = 1990
        remaining_text = memory_text
        is_first = True
        while len(remaining_text) > 0:
            if len(remaining_text) <= chunk_size:
                chunk = remaining_text; remaining_text = ""
            else:
                split_index = remaining_text.rfind('\n', 0, chunk_size)
                if split_index == -1: split_index = remaining_text.rfind(' ', 0, chunk_size)
                if split_index == -1: split_index = chunk_size
                else: split_index += 1
                chunk = remaining_text[:split_index]; remaining_text = remaining_text[split_index:]
            try:
                if is_first:
                    try: await message.reply(chunk)
                    except discord.HTTPException as http_exc:
                        if http_exc.code == 50035: await message.channel.send(f"<@{message.author.id}> {chunk}")
                        else: raise http_exc
                    is_first = False
                else:
                    async with safe_typing(message.channel): await asyncio.sleep(1.0)
                    await message.channel.send(chunk)
            except discord.Forbidden: break
        return

    image_attachments = [att for att in message.attachments if att.content_type and att.content_type.startswith('image/')]
    valid_stickers = [s for s in message.stickers if s.format != discord.StickerFormatType.lottie]
    
    if message.reference and message.reference.message_id:
        try:
            replied_msg = await message.channel.fetch_message(message.reference.message_id)
            if replied_msg.content:
                replied_user_name = f"{replied_msg.author.display_name}_{str(replied_msg.author.id)[-4:]}"
                reply_context = f"\n\n[Context: {user_name} is replying to the following message by {replied_user_name}: \"{replied_msg.content}\"]"
                clean_message += reply_context
                if replied_msg.author == client.user:
                    clean_message += "\n[System Directive: If you need more specific facts to answer this follow-up, you MUST output a web_search tool call. Do not guess.]"
            replied_images = [att for att in replied_msg.attachments if att.content_type and att.content_type.startswith('image/')]
            image_attachments.extend(replied_images)
            replied_stickers = [s for s in replied_msg.stickers if s.format != discord.StickerFormatType.lottie]
            valid_stickers.extend(replied_stickers)
        except Exception as e: print(f"⚠️ Could not fetch the replied message: {e}")

    if not clean_message.strip() and not image_attachments and not valid_stickers:
        await send_help_menu(message, client.user.name)
        return

    api_user_content = []
    text_part = f"{user_name}: {clean_message}" if clean_message else f"{user_name}: What is in this image?"
    if image_attachments or valid_stickers:
        api_user_content.append({"type": "text", "text": text_part})
        for img in image_attachments:
            img_bytes = await img.read()
            
            # FIX: VRAM Optimization - Resize massive images before sending to local LLM
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                # Convert RGBA (transparent PNGs) to RGB to prevent JPEG conversion crashes
                if pil_img.mode in ("RGBA", "P"):
                    pil_img = pil_img.convert("RGB")
                    
                # thumbnail() mathematically shrinks the image to fit within the box while keeping aspect ratio
                pil_img.thumbnail((1024, 1024))
                
                # Save it into a new buffer as a standardized, compressed JPEG
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=85)
                
                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
            api_user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
        for sticker in valid_stickers:
            sticker_bytes = await sticker.read(); sticker_b64 = base64.b64encode(sticker_bytes).decode('utf-8')
            api_user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{sticker.format.name};base64,{sticker_b64}"}})
    else: api_user_content = text_part

    current_system_prompt = (
        f"Today's date is {datetime.now().strftime('%B %d, %Y')}.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. EXTREME BREVITY: Answer in 1-3 sentences unless asked otherwise.\n"
        "2. MULTI-USER CHAT: Address users by their names when appropriate.\n"
        "3. SEARCH DIRECTIVE: Use the web_search tool if current info is needed. NEVER guess.\n"
        "4. STRICT RULE: Do not use emojis unless your personality requires it.\n"
        "5. MEMORY USAGE (CRITICAL): Use user facts silently to guide your context. NEVER repeat or summarize these facts back to the user unless they explicitly ask what you remember about them."
    )
    
    user_context_str = ""
    async with aiosqlite.connect(DB_FILE) as db:
        # Fetch Active Users
        cursor = await db.execute("SELECT DISTINCT user_id FROM chat_history WHERE server_id = ? AND user_id IS NOT NULL", (server_id,))
        active_user_ids = [str(r[0]) for r in await cursor.fetchall()]
        active_user_ids.append(str(message.author.id))
        active_user_ids = list(set(active_user_ids))
        
        # Inject Memories
        if active_user_ids:
            placeholders = ','.join('?' * len(active_user_ids))
            # Include server_id in the parameters list
            params = [server_id] + active_user_ids 
            cursor = await db.execute(f"SELECT user_name, facts FROM core_memories WHERE server_id = ? AND user_id IN ({placeholders})", params)
            rows = await cursor.fetchall()
            for row in rows:
                if row[1] != "No core memories yet.":
                    user_context_str += f"- {row[0]}: {row[1]}\n"
                    
        # Fetch Role
        cursor = await db.execute("SELECT prompt FROM server_config WHERE server_id = ?", (server_id,))
        row = await cursor.fetchone()
        base_personality = row[0] if row and row[0] else "You are a neutral, conversational AI."

    if user_context_str: 
        current_system_prompt += f"\n\nCRITICAL CONTEXT ABOUT ACTIVE USERS (READ ONLY - DO NOT REPEAT):\n{user_context_str}"

    current_system_prompt += f"\n\nYOUR ASSIGNED PERSONA AND ROLE:\n{base_personality}"
    system_message = {"role": "system", "content": current_system_prompt}

    # Grab a snapshot of current history
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute("SELECT role, content FROM chat_history WHERE server_id = ? ORDER BY id ASC", (server_id,))
        
        api_history = []
        for r in await cursor.fetchall():
            try:
                parsed_content = json.loads(r[1])
                
                # FIX: Prevent json.loads from hijacking pure numbers or raw text arrays
                if isinstance(parsed_content, list):
                    # Ensure it is a Vision array (contains dicts), otherwise revert to string
                    if len(parsed_content) > 0 and not isinstance(parsed_content[0], dict):
                        parsed_content = r[1]
                elif not isinstance(parsed_content, str):
                    # If it parsed into an int, float, or bool, revert it to string
                    parsed_content = str(parsed_content)
                    
            except (json.JSONDecodeError, TypeError):
                parsed_content = r[1] # Fallback for older messages
                
            api_history.append({"role": r[0], "content": parsed_content})
        
    cleaned_history = []
    for msg in api_history:
        if not cleaned_history:
            if msg["role"] == "assistant": cleaned_history.append({"role": "user", "content": "[Conversation Started]"})
            cleaned_history.append(msg)
        else:
            if cleaned_history[-1]["role"] == msg["role"]:
                c1 = cleaned_history[-1]["content"]
                c2 = msg["content"]
                
                # Scenario A: Both are simple text strings
                if isinstance(c1, str) and isinstance(c2, str):
                    cleaned_history[-1]["content"] = f"{c1}\n\n{c2}"
                # Scenario B: One or both are vision arrays (lists)
                else:
                    # Convert strings to OpenAI text dicts if necessary
                    list1 = [{"type": "text", "text": c1}] if isinstance(c1, str) else c1.copy()
                    list2 = [{"type": "text", "text": f"\n\n{c2}"}] if isinstance(c2, str) else c2.copy()
                    # Safely merge the two lists
                    cleaned_history[-1]["content"] = list1 + list2
            else: 
                cleaned_history.append(msg)
                
    if cleaned_history and cleaned_history[-1]["role"] == "user":
        c1 = cleaned_history[-1]["content"]
        c2 = api_user_content
        
        # Scenario A: Both are simple text strings
        if isinstance(c1, str) and isinstance(c2, str):
            cleaned_history[-1]["content"] = f"{c1}\n\n{c2}"
        # Scenario B: One or both are vision arrays (lists)
        else:
            # Convert strings to OpenAI text dicts if necessary
            list1 = [{"type": "text", "text": c1}] if isinstance(c1, str) else c1.copy()
            list2 = [{"type": "text", "text": f"\n\n{c2}"}] if isinstance(c2, str) else c2.copy()
            # Safely merge the two lists
            cleaned_history[-1]["content"] = list1 + list2
    else: 
        cleaned_history.append({"role": "user", "content": api_user_content})
        
    messages_to_send = [system_message] + cleaned_history

    async with safe_typing(message.channel):
        async with llm_queue:
            try:
                max_iterations = 3
                current_iteration = 0
                final_reply = ""
                
                response = await lm_client.chat.completions.create(model=LLM_MODEL_NAME, messages=messages_to_send, tools=tools_schema, tool_choice="auto", temperature=1, max_tokens=1000)
                if response.usage and response.usage.total_tokens > highest_token_count: highest_token_count = response.usage.total_tokens
                response_message = response.choices[0].message
                
                while current_iteration < max_iterations:
                    if response_message.tool_calls:
                        # FIX: Safely dump the model but force the 'content' key to exist to prevent schema crashes
                        msg_dump = response_message.model_dump(exclude_none=True)
                        if "content" not in msg_dump:
                            msg_dump["content"] = "" # Satisfies the strict OpenAI schema requirement
                        messages_to_send.append(msg_dump)
                        
                        search_tasks = []; tool_call_metadata = []
                        
                        for tool_call in response_message.tool_calls:
                            if tool_call.function.name == "web_search":
                                try:
                                    args = json.loads(tool_call.function.arguments)
                                    
                                    # FIX: Ensure the LLM didn't hallucinate a string or list before using .get()
                                    if isinstance(args, dict):
                                        search_tasks.append(perform_web_search(args.get("query")))
                                    else:
                                        search_tasks.append(resolve_tool_error("System error: Tool arguments must be a valid JSON object."))
                                        
                                    tool_call_metadata.append(tool_call)
                                except json.JSONDecodeError:
                                    search_tasks.append(resolve_tool_error("System error: Invalid JSON."))
                                    tool_call_metadata.append(tool_call)
                            else:
                                error_str = f"System error: Tool '{tool_call.function.name}' does not exist. Do not use it."
                                search_tasks.append(resolve_tool_error(error_str))
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
                            temperature=1, 
                            max_tokens=1000
                        )
                        
                        if response.usage and response.usage.total_tokens > highest_token_count: 
                            highest_token_count = response.usage.total_tokens
                            
                        response_message = response.choices[0].message
                        current_iteration += 1
                    else:
                        final_reply = response_message.content
                        break
                
                if not final_reply:
                    final_reply = response_message.content or "⚠️ *System error: Reached max tool iterations or empty response.*"
                    
                # 3. ATOMIC PAIR INSERTION & CHUNKED PRUNING VIA SQLITE
                async with aiosqlite.connect(DB_FILE) as db:
                    # Convert the full multimodal arrays into JSON strings for storage (Vision Bug Fix)
                    db_user_content = json.dumps(api_user_content)
                    
                    # FIX: Save ONLY the final synthesized reply! (No JSON encoding needed for strings)
                    db_assistant_content = final_reply
                    
                    await db.execute("INSERT INTO chat_history (server_id, role, content, user_id, user_name) VALUES (?, ?, ?, ?, ?)", 
                                    (server_id, "user", db_user_content, str(message.author.id), user_name))
                    # FIX: Tag the assistant's reply with the user's ID so 'memory clear' deletes the full interaction!
                    await db.execute("INSERT INTO chat_history (server_id, role, content, user_id, user_name) VALUES (?, ?, ?, ?, ?)", 
                                    (server_id, "assistant", db_assistant_content, str(message.author.id), user_name))
                    
                    # FIX: "Chunked Eviction" - Check current history length
                    cursor = await db.execute("SELECT COUNT(*) FROM chat_history WHERE server_id = ?", (server_id,))
                    current_count = (await cursor.fetchone())[0]
                    
                    # If history hits 50, slice the oldest 20 messages for bulk summarization
                    if current_count >= 50:
                        cursor = await db.execute('''SELECT id, role, content, user_id, user_name FROM chat_history 
                                                    WHERE server_id = ? ORDER BY id ASC LIMIT 20''', (server_id,))
                        forgotten_rows = await cursor.fetchall()
                        
                        if forgotten_rows:
                            forgotten_msgs = [{"role": r[1], "content": r[2], "user_id": r[3], "user_name": r[4]} for r in forgotten_rows]
                            users_in_forgotten = {msg["user_id"]: msg["user_name"] for msg in forgotten_msgs if msg.get("user_id")}
                            
                            if users_in_forgotten:
                                task = asyncio.create_task(process_memories_sequentially(server_id, users_in_forgotten, forgotten_msgs))
                                background_tasks.add(task)
                                task.add_done_callback(background_tasks.discard)
                                
                            # Safely delete only the exact 20 messages we just extracted
                            ids_to_delete = [r[0] for r in forgotten_rows]
                            placeholders = ','.join('?' * len(ids_to_delete))
                            await db.execute(f"DELETE FROM chat_history WHERE id IN ({placeholders})", ids_to_delete)
                    
                    await db.commit()
                
                remaining_text = final_reply; is_first = True
                while len(remaining_text) > 0:
                    if len(remaining_text) <= 1990: chunk = remaining_text; remaining_text = ""
                    else:
                        split_index = remaining_text.rfind('\n', 0, 1990)
                        if split_index == -1: split_index = remaining_text.rfind(' ', 0, 1990)
                        if split_index == -1: split_index = 1990
                        else: split_index += 1 
                        chunk = remaining_text[:split_index]; remaining_text = remaining_text[split_index:]
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
            except Exception as e:
                print(f"Error: {e}")
                try: await message.reply("Oops! I couldn't process that.")
                except: pass
                return

if TOKEN: client.run(TOKEN)