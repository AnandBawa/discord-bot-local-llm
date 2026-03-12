import discord
import asyncio 
import json
import os
import base64
import contextlib
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

lm_client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# --- DATA SETUP ---
GLOBAL_DATA_FILE = "global_data.json"
MEMORY_FILE = "core_memories.json"
MAX_HISTORY_LENGTH = 50 
highest_token_count = 0
global_lock = asyncio.Lock()
memory_lock = asyncio.Lock()

def load_json(filepath, default_val):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return default_val

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

# Initialize as an empty dictionary since we now store data per server_id
global_data = load_json(GLOBAL_DATA_FILE, {})

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
    help_text = (
        "**Here is how you can interact with me:**\n\n"
        f"• **`@{bot_name} [message]`** - Tag me to ask a question, chat, or analyze an attached image/sticker.\n"
        f"  Example: @{bot_name} what is the weather in Tokyo?\n\n"
        f"• **`@{bot_name} role`** - Show my currently active global personality.\n"
        f"• **`@{bot_name} role [prompt]`** - Set a custom personality and wipe history (saves core memories first).\n"
        f"  Example: @{bot_name} role You are a grumpy history professor who hates technology.\n\n"
        f"• **`@{bot_name} role clear`** - Reset me to my default neutral behavior.\n"
        f"• **`@{bot_name} clear`** - Wipe the global conversation history (saves core memories first).\n"
        f"• **`@{bot_name} memory`** - View the list of users I have core memories for.\n"
        f"• **`@{bot_name} memory [name]`** - View specific permanent facts learned about a person.\n"
        f"  Example: @{bot_name} memory Charlie\n\n"
        f"• **`@{bot_name} status`** - Show my latency, loaded AI model, history capacity, and token usage.\n"
        f"• **`@{bot_name} help`** - Show this menu.\n\n"
        "**Good to know:**\n"
        "> **Long-term Memory:** I remember the last 50 interactions in the channel. Core facts are automatically extracted and saved before history is cleared or naturally cycles out.\n"
        "> **Contextual Replies:** You can **reply** to an old message or image and tag me to continue that specific thought.\n"
        "> **Live Web Search:** If you ask about current events or facts I don't know, I will autonomously search the web to find the answer!"
    )
    await message.reply(help_text)

async def perform_web_search(query):
    # GUARD: Prevent crashes if the LLM hallucinates an empty search
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
        results = await asyncio.to_thread(
            lambda: DDGS().text(optimized_query, max_results=3)
        )
            
        if not results:
            return "No results."
        
        search_text = f"Date: {current_date}\n"
        for res in results:
            search_text += f"[{res.get('title', 'No Title')}] {res.get('body', 'No Body')}\n"
        return search_text
    except Exception as e:
        return f"Search error: {e}"
    
async def update_user_memory(user_id, user_name, forgotten_messages):
    # FIX: Lock the ENTIRE process so simultaneous memory queues don't overwrite each other
    async with memory_lock:
        current_memories = load_json(MEMORY_FILE, {})
        user_data = current_memories.get(str(user_id), {"name": user_name, "facts": "No core memories yet."})
        existing_memory = user_data["facts"]
        
        chat_log = ""
        for msg in forgotten_messages:
            chat_log += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
        memory_prompt = (
            f"You are an AI memory manager. Extract long-term, permanent facts about the user '{user_name}' from the chat log below. "
            "Update the existing memory with any new facts (preferences, tech stack, ongoing projects). "
            "Keep it strictly concise and bulleted. If there are no new facts about this user, just return the EXISTING MEMORY exactly as is. "
            "Do not include temporary conversational details.\n\n"
            f"EXISTING MEMORY for {user_name}:\n{existing_memory}\n\n"
            f"RECENT CHAT LOG:\n{chat_log}"
        )
        
        try:
            response = await lm_client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": memory_prompt}],
                temperature=0.3, 
                max_tokens=300
            )
            
            new_memory = response.choices[0].message.content.strip()
            
            if new_memory != existing_memory and new_memory:
                # File is already safely locked from the top!
                user_data["facts"] = new_memory
                user_data["name"] = user_name 
                current_memories[str(user_id)] = user_data
                save_json(MEMORY_FILE, current_memories)
                print(f"[Memory] Core memory updated for {user_name}.")
            else:
                print(f"[Memory] Scanned old messages, no new permanent facts found for {user_name}.")
                
        except Exception as e:
            print(f"Failed to update core memory for {user_name}: {e}")

async def process_memories_sequentially(users_dict, forgotten_messages):
    for uid, uname in users_dict.items():
        await update_user_memory(uid, uname, forgotten_messages)
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
    print(f'✅ Logged in successfully as {client.user}')
    print('🌐 Autonomous Web Search capability enabled!')

@client.event
async def on_message(message):
    global highest_token_count
    
    if message.author == client.user or not client.user.mentioned_in(message):
        return

    # IGNORE DMs: The bot must be used inside a Server (Guild) to track histories properly
    if not message.guild:
        return
        
    server_id = str(message.guild.id)
    
    # Initialize this specific server in the JSON if it's their first time talking
    async with global_lock:
        if server_id not in global_data:
            global_data[server_id] = {"history": [], "prompt": ""}
            save_json(GLOBAL_DATA_FILE, global_data)

    # Stripping logic that handles both @Name and @Nickname formats
    bot_mention = f'<@{client.user.id}>'
    bot_nickname_mention = f'<@!{client.user.id}>' 
    
    clean_message = message.content.replace(bot_mention, '').replace(bot_nickname_mention, '').strip()
    user_name = f"{message.author.display_name}_{str(message.author.id)[-4:]}"

    for mentioned_user in message.mentions:
        if mentioned_user.id != client.user.id:
            memory_formatted_name = f"{mentioned_user.display_name}_{str(mentioned_user.id)[-4:]}"
            clean_message = clean_message.replace(f"<@{mentioned_user.id}>", f"@{memory_formatted_name}")
            clean_message = clean_message.replace(f"<@!{mentioned_user.id}>", f"@{memory_formatted_name}")

    if clean_message.lower() == 'role':
        current_role = global_data[server_id].get("prompt")
        if current_role:
            await message.reply(f"**Current Server Personality:**\n> *{current_role}*")
        else:
            await message.reply("**Current Server Personality:**\n> *(Default) You are a neutral, conversational AI.*")
        return

    if clean_message.lower().startswith('role '):
        new_prompt = clean_message[5:].strip() 
        if new_prompt.lower() == 'clear':
            async with global_lock:
                history_to_save = global_data[server_id].get("history", [])
                if history_to_save:
                    users_in_history = {msg["user_id"]: msg["user_name"] for msg in history_to_save if msg.get("user_id")}
                    asyncio.create_task(process_memories_sequentially(users_in_history, history_to_save))
                
                global_data[server_id]["prompt"] = ""
                global_data[server_id]["history"] = [] 
                save_json(GLOBAL_DATA_FILE, global_data)
            
            default_persona = "You are a neutral, conversational AI."
            await message.reply(
                f"✅ Server personality removed and conversation history cleared! *(Recent memories were saved)*\n\n"
                f"**Current Personality:**\n> {default_persona}"
            )
            return
        if not new_prompt:
            await message.reply(f"Please provide a prompt! Example: `@{client.user.name} role You are a pirate.`")
            return
        async with global_lock:
            history_to_save = global_data[server_id].get("history", [])
            if history_to_save:
                users_in_history = {msg["user_id"]: msg["user_name"] for msg in history_to_save if msg.get("user_id")}
                asyncio.create_task(process_memories_sequentially(users_in_history, history_to_save))
            global_data[server_id]["prompt"] = new_prompt
            global_data[server_id]["history"] = [] 
            save_json(GLOBAL_DATA_FILE, global_data)
        await message.reply(f"✅ Saved server personality and cleared conversation history for a fresh start! *(Recent memories were saved)*\n> *{new_prompt}*")
        return

    if clean_message.lower() == 'clear':
        async with global_lock:
            history_to_save = global_data[server_id].get("history", [])
            if history_to_save:
                users_in_history = {msg["user_id"]: msg["user_name"] for msg in history_to_save if msg.get("user_id")}
                asyncio.create_task(process_memories_sequentially(users_in_history, history_to_save))
            global_data[server_id]["history"] = []
            save_json(GLOBAL_DATA_FILE, global_data)
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
        diagnostics = (
            "**Bot Diagnostics & Status**\n\n"
            f"• **Discord Ping:** `{ping_ms}ms`\n"
            f"• **Loaded AI Model:** `{current_model}`\n"
            f"• **Peak Context Used:** `{highest_token_count} tokens`\n"
            f"• **Current History Length:** `{len(global_data[server_id]['history'])}/{MAX_HISTORY_LENGTH} messages`"
        )
        await status_msg.edit(content=diagnostics)
        return
    
    if clean_message.lower().startswith('memory'):
        current_memories = load_json(MEMORY_FILE, {})
        if not current_memories:
            await message.reply("I don't have any core memories saved yet.")
            return

        command_parts = clean_message.split()
        if len(command_parts) > 1:
            target_name = " ".join(command_parts[1:]).lower()
            
            # FIX: Build the active users list for the targeted search
            active_users = {str(message.author.id)}
            for msg in global_data[server_id].get("history", []):
                if "user_id" in msg: active_users.add(str(msg["user_id"]))
                
            found_user = None
            # FIX: Only search for the name if they are in the active_users set
            for uid in active_users:
                if uid in current_memories and target_name in current_memories[uid]['name'].lower():
                    found_user = current_memories[uid]
                    break
                    
            if found_user:
                memory_text = f"**Facts known about {found_user['name']}:**\n{found_user['facts']}"
            else:
                memory_text = f"I couldn't find any memories for '{target_name}' in this server's recent chat history."
        else:
            # FIX: Only list users active in the current server's history
            active_users = {str(message.author.id)}
            for msg in global_data[server_id].get("history", []):
                if "user_id" in msg: active_users.add(str(msg["user_id"]))
                
            memory_text = f"**Tracked Active Users**\nType `@{client.user.name} memory [name]` to search all known users.\n\n"
            found_any = False
            for uid in active_users:
                if uid in current_memories and current_memories[uid]['facts'] and current_memories[uid]['facts'] != "No core memories yet.":
                    memory_text += f"• {current_memories[uid]['name']}\n"
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
            img_bytes = await img.read(); img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            api_user_content.append({"type": "image_url", "image_url": {"url": f"data:{img.content_type};base64,{img_b64}"}})
        for sticker in valid_stickers:
            sticker_bytes = await sticker.read(); sticker_b64 = base64.b64encode(sticker_bytes).decode('utf-8')
            api_user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{sticker.format.name};base64,{sticker_b64}"}})
    else: api_user_content = text_part

    current_system_prompt = (
        f"Today's date is {datetime.now().strftime('%B %d, %Y')}.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. EXTREME BREVITY: Answer in 1-5 sentences unless asked otherwise.\n"
        "2. MULTI-USER CHAT: Address users by their names when appropriate.\n"
        "3. SEARCH DIRECTIVE: Use the web_search tool if current info is needed. NEVER guess.\n"
        "4. STRICT RULE: Do not use emojis unless your personality requires it.\n"
        "5. MEMORY USAGE (CRITICAL): Use user facts below silently. NEVER repeat facts back unless asked."
    )
    
    current_memories = load_json(MEMORY_FILE, {})
    
    # FIX: Only inject memories of users actively participating in THIS server's conversation
    active_users = {str(message.author.id)}
    for msg in global_data[server_id].get("history", []):
        if "user_id" in msg:
            active_users.add(str(msg["user_id"]))
            
    user_context_str = ""
    for uid in active_users:
        if uid in current_memories and current_memories[uid]['facts'] != "No core memories yet.":
            user_context_str += f"- {current_memories[uid]['name']}: {current_memories[uid]['facts']}\n"
            
    if user_context_str: 
        current_system_prompt += f"\n\nCRITICAL CONTEXT ABOUT ACTIVE USERS (READ ONLY - DO NOT REPEAT):\n{user_context_str}"

    base_personality = global_data[server_id].get("prompt") or "You are a neutral, conversational AI."
    current_system_prompt += f"\n\nYOUR ASSIGNED PERSONA AND ROLE:\n{base_personality}"
    system_message = {"role": "system", "content": current_system_prompt}

    history_text_save = text_part
    total_visuals = len(image_attachments) + len(valid_stickers)
    if total_visuals > 0: history_text_save += f" [User attached {total_visuals} visual(s)]"

    # --- START OF CONCURRENCY FIX ---
    # 1. Grab a snapshot of current history WITHOUT locking or appending yet
    async with global_lock:
        raw_history = list(global_data[server_id].get("history", []))
        
    api_history = [{"role": m["role"], "content": m["content"]} for m in raw_history]
        
    cleaned_history = []
    for msg in api_history:
        if not cleaned_history:
            if msg["role"] == "assistant": cleaned_history.append({"role": "user", "content": "[Conversation Started]"})
            cleaned_history.append(msg)
        else:
            if cleaned_history[-1]["role"] == msg["role"]: cleaned_history[-1]["content"] += f"\n\n{msg['content']}"
            else: cleaned_history.append(msg)
                
    if cleaned_history and cleaned_history[-1]["role"] == "user":
        if isinstance(api_user_content, str): cleaned_history[-1]["content"] += f"\n\n{api_user_content}"
        else:
            prev_text = cleaned_history[-1]["content"]
            merged_content = [{"type": "text", "text": f"{prev_text}\n\n{api_user_content[0]['text']}"}] + api_user_content[1:]
            cleaned_history[-1]["content"] = merged_content
    else: cleaned_history.append({"role": "user", "content": api_user_content})
        
    messages_to_send = [system_message] + cleaned_history

    async with safe_typing(message.channel):
        try:
            # 2. Make API Call with a Loop for chained tool calls
            max_iterations = 3
            current_iteration = 0
            final_reply = ""
            search_context_for_history = ""
            
            response = await lm_client.chat.completions.create(model="local-model", messages=messages_to_send, tools=tools_schema, tool_choice="auto", temperature=1, max_tokens=1000)
            if response.usage and response.usage.total_tokens > highest_token_count: highest_token_count = response.usage.total_tokens
            response_message = response.choices[0].message
            
            while current_iteration < max_iterations:
                if response_message.tool_calls:
                    messages_to_send.append(response_message.model_dump(exclude_none=True))
                    search_tasks = []; tool_call_metadata = []
                    
                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "web_search":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                search_tasks.append(perform_web_search(args.get("query")))
                                tool_call_metadata.append(tool_call)
                            except json.JSONDecodeError:
                                # FIX: Evaluated immediately, no closure traps
                                search_tasks.append(resolve_tool_error("System error: Invalid JSON."))
                                tool_call_metadata.append(tool_call)
                        else:
                            # FIX: Evaluated immediately
                            error_str = f"System error: Tool '{tool_call.function.name}' does not exist. Do not use it."
                            search_tasks.append(resolve_tool_error(error_str))
                            tool_call_metadata.append(tool_call)
                    
                    if search_tasks:
                        completed_results = await asyncio.gather(*search_tasks)
                        search_context_for_history += "\n".join(completed_results) + "\n" # Accumulate history
                        
                        for tool_call, result_text in zip(tool_call_metadata, completed_results):
                            messages_to_send.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result_text})
                    
                    # Request the next step from the model (NOW WITH TOOLS AND TRACKING!)
                    response = await lm_client.chat.completions.create(
                        model="local-model", 
                        messages=messages_to_send, 
                        tools=tools_schema, # FIX: Re-enable tools for chaining!
                        tool_choice="auto", 
                        temperature=1, 
                        max_tokens=1000
                    )
                    
                    # FIX: Accurately track the heavier chained requests
                    if response.usage and response.usage.total_tokens > highest_token_count: 
                        highest_token_count = response.usage.total_tokens
                        
                    response_message = response.choices[0].message
                    current_iteration += 1
                else:
                    # The model returned a text response! Break the loop.
                    final_reply = response_message.content
                    break
            
            # Fallback if the model chained tools too many times or returned empty content
            if not final_reply:
                final_reply = response_message.content or "⚠️ *System error: Reached max tool iterations or empty response.*"
                
            history_save_text = f"[Internal Web Search Context:\n{search_context_for_history.strip()}]\n\n{final_reply}" if search_context_for_history else final_reply
            
            # 3. ATOMIC PAIR INSERTION: Append User AND Assistant simultaneously to prevent scrambling
            async with global_lock:
                global_data[server_id].setdefault("history", []).append({"role": "user", "content": history_text_save, "user_id": str(message.author.id), "user_name": user_name})
                global_data[server_id]["history"].append({"role": "assistant", "content": history_save_text})
                
                if len(global_data[server_id]["history"]) > MAX_HISTORY_LENGTH:
                    # Dynamic slice prevents Parity Overlap bugs
                    forgotten = global_data[server_id]["history"][:-MAX_HISTORY_LENGTH]
                    global_data[server_id]["history"] = global_data[server_id]["history"][-MAX_HISTORY_LENGTH:]
                    
                    # Extract ALL unique users from the forgotten messages (Fixes Memory Dropout)
                    users_in_forgotten = {msg["user_id"]: msg["user_name"] for msg in forgotten if msg.get("user_id")}
                    if users_in_forgotten:
                        asyncio.create_task(process_memories_sequentially(users_in_forgotten, forgotten))
                save_json(GLOBAL_DATA_FILE, global_data)
            
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