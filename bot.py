import discord
import asyncio 
import json
import os
import base64
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

def load_json(filepath, default_val):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return default_val

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

# Load our separated data files
global_data = load_json(GLOBAL_DATA_FILE, {"history": [], "prompt": ""})

async def send_help_menu(message, bot_name):
    help_text = (
        "Here is how you can interact with me:\n\n"
        f"• **`@{bot_name} [your message]`** - Tag me to ask a question, say hello, or attach an image!\n"
        f"• **`@{bot_name} role [prompt]`** - Give me a custom personality. Note: This changes my behavior globally for everyone, but keeps the ongoing conversation history intact.\n"
        f"• **`@{bot_name} role clear`** - Remove the custom personality and reset me to my default behavior. Conversation history is kept intact.\n"
        f"• **`@{bot_name} clear`** - Wipe the current global conversation history (up to 50 messages) to start a completely fresh topic.\n"
        f"• **`@{bot_name} memory`** - View the permanent facts I have learned about everyone! This displays a user-by-user breakdown of long-term memories (like names, preferences, and ongoing projects) that I secretly track and save for each individual person in the server.\n"
        f"• **`@{bot_name} status`** - Show the bot's latency, currently loaded AI model, peak token usage, and how full the current history is.\n"
        f"• **`@{bot_name} help`** - Show this list of commands.\n"
        "**Good to know:**\n"
        "• **Memory:** I remember our recent chat context of last 50 messages and continuously summarize older messages to build a permanent core memory of our interactions!\n"
        "• **Replies:** You can **reply** to any message (text or image) and tag me to summarize it, roast it, or explain it!\n"
        "• **Web Search:** If you ask me about current events, news, or anything I might not know, I'll automatically perform a web search to find the latest information for you!\n"
    )
    await message.reply(help_text)

async def perform_web_search(query):
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
        # Run the synchronous DDGS search in a background thread so it doesn't block Discord
        results = await asyncio.to_thread(
            lambda: DDGS().text(optimized_query, max_results=3)
        )
            
        if not results:
            return "No results."
        
        # Token-optimized formatting
        search_text = f"Date: {current_date}\n"
        for res in results:
            search_text += f"[{res.get('title', 'No Title')}] {res.get('body', 'No Body')}\n"
        return search_text
    except Exception as e:
        return f"Search error: {e}"
    
async def update_user_memory(user_id, user_name, forgotten_messages):
    """Summarizes old messages specifically for the user who sent them."""
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
            user_data["facts"] = new_memory
            user_data["name"] = user_name 
            current_memories[str(user_id)] = user_data
            save_json(MEMORY_FILE, current_memories)
            print(f"[Memory] Core memory updated for {user_name}.")
        else:
            print(f"[Memory] Scanned old messages, no new permanent facts found for {user_name}.")
            
    except Exception as e:
        print(f"Failed to update core memory for {user_name}: {e}")

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

    bot_mention = f'<@{client.user.id}>'
    clean_message = message.content.replace(bot_mention, '').strip()
    user_name = f"{message.author.display_name}_{str(message.author.id)[-4:]}"

    if clean_message.lower().startswith('role '):
        new_prompt = clean_message[5:].strip() 
        
        if new_prompt.lower() == 'clear':
            # Add lock here
            async with global_lock:
                global_data["prompt"] = ""
                save_json(GLOBAL_DATA_FILE, global_data)
            await message.reply("✅ Global personality removed! The ongoing conversation history was kept intact.")
            return

        if not new_prompt:
            await message.reply(f"Please provide a prompt! Example: `@{client.user.name} role You are a pirate.`")
            return
            
        # Add lock here
        async with global_lock:
            global_data["prompt"] = new_prompt
            save_json(GLOBAL_DATA_FILE, global_data)
        await message.reply(f"✅ Saved global personality (ongoing conversation history was kept intact):\n> *{new_prompt}*")
        return

    if clean_message.lower() == 'clear':
        # Add lock here
        async with global_lock:
            global_data["history"] = []
            save_json(GLOBAL_DATA_FILE, global_data)
        await message.reply("🗑️ Global conversation history cleared!")
        return
    
    if clean_message.lower() == 'help':
        await send_help_menu(message, client.user.name)
        return
    
    if clean_message.lower() == 'status':
        ping_ms = round(client.latency * 1000)
        status_msg = await message.reply("Fetching diagnostics...")
        try:
            models_response = await lm_client.models.list()
            current_model = models_response.data[0].id if models_response.data else "None (No model loaded in LM Studio)"
        except Exception as e:
            current_model = f"Offline / Unreachable"
            
        diagnostics = (
            "**Bot Diagnostics & Status**\n\n"
            f"• **Discord Ping:** `{ping_ms}ms`\n"
            f"• **Loaded AI Model:** `{current_model}`\n"
            f"• **Peak Context Used:** `{highest_token_count} tokens`\n"
            f"• **Current History Length:** `{len(global_data['history'])}/{MAX_HISTORY_LENGTH} messages`"
        )
        await status_msg.edit(content=diagnostics)
        return
    
    if clean_message.lower() == 'memory':
        current_memories = load_json(MEMORY_FILE, {})
        if not current_memories:
            await message.reply("I don't have any core memories saved yet.")
            return
            
        memory_text = "**• Channel Memory (Tracked by User)**\n*Here is what I currently know about everyone:*\n\n"
        has_facts = False
        for uid, m_data in current_memories.items():
            if m_data['facts'] and m_data['facts'] != "No core memories yet.":
                memory_text += f"**{m_data['name']}**:\n{m_data['facts']}\n\n"
                has_facts = True
                
        if not has_facts:
            await message.reply("I don't have any core memories saved yet.")
        else:
            await message.reply(memory_text[:1990])
        return

    image_attachments = [att for att in message.attachments if att.content_type and att.content_type.startswith('image/')]
    
    if message.reference and message.reference.message_id:
        try:
            replied_msg = await message.channel.fetch_message(message.reference.message_id)
            if replied_msg.content:
                reply_context = f"\n\n[Context: {user_name} is replying to the following message by {replied_msg.author.display_name}: \"{replied_msg.content}\"]"
                clean_message += reply_context
                if replied_msg.author == client.user:
                    clean_message += "\n[System Directive: If you need more specific facts to answer this follow-up, you MUST output a web_search tool call. Do not guess.]"
            replied_images = [att for att in replied_msg.attachments if att.content_type and att.content_type.startswith('image/')]
            image_attachments.extend(replied_images)
        except Exception as e:
            print(f"⚠️ Could not fetch the replied message: {e}")

    if not clean_message.strip() and not image_attachments:
        await send_help_menu(message, client.user.name)
        return

    api_user_content = []
    text_part = f"{user_name}: {clean_message}" if clean_message else f"{user_name}: What is in this image?"
    
    if image_attachments:
        api_user_content.append({"type": "text", "text": text_part})
        for img in image_attachments:
            img_bytes = await img.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            api_user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img.content_type};base64,{img_b64}"}
            })
    else:
        api_user_content = text_part

    base_personality = global_data.get("prompt") or "You are a neutral, conversational AI."
        
    current_system_prompt = (
        f"Today's date is {datetime.now().strftime('%B %d, %Y')}. {base_personality}\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. EXTREME BREVITY: You must keep your responses short, concise, and straight to the point. Answer in 1-3 sentences unless the user explicitly asks for a detailed explanation, an essay, or a script.\n"
        "2. MULTI-USER CHAT: You are in a group chat with multiple users. Usernames are prepended to their messages. Address users by their names when appropriate.\n"
        "3. SEARCH DIRECTIVE: If asked about current events, real-time info, OR if the user asks a follow-up question related to it, you MUST use the web_search tool. NEVER guess. \n"
        "4. STRICT RULE: Do not use emojis unless your personality requires it."
    )
    
    # Inject all known user memories into the system prompt
    current_memories = load_json(MEMORY_FILE, {})
    user_context_str = ""
    for uid, m_data in current_memories.items():
         if m_data['facts'] and m_data['facts'] != "No core memories yet.":
             user_context_str += f"- {m_data['name']}: {m_data['facts']}\n"
             
    if user_context_str:
         current_system_prompt += f"\n\nCRITICAL CONTEXT ABOUT CHANNEL USERS:\n{user_context_str}"
        
    system_message = {"role": "system", "content": current_system_prompt}

    history_text_save = text_part
    if image_attachments:
        history_text_save += f" [User attached {len(image_attachments)} image(s)]"

    # Grab the lock to safely update the shared global history
    async with global_lock:
        global_data.setdefault("history", []).append({
            "role": "user", 
            "content": history_text_save,
            "user_id": str(message.author.id),
            "user_name": user_name
        })
        # Save immediately so the next concurrent user sees this message
        save_json(GLOBAL_DATA_FILE, global_data)

        # IMPORTANT: Strip out our custom metadata before sending history to the API
        api_history = [{"role": m["role"], "content": m["content"]} for m in global_data.get("history", [])]
        
    # Build the payload using the safely updated history
    messages_to_send = [system_message] + api_history

    messages_to_send[-1]["content"] = api_user_content

    async with message.channel.typing():
        try:
            response = await lm_client.chat.completions.create(
                model="local-model",
                messages=messages_to_send, 
                tools=tools_schema,       
                tool_choice="auto",       
                temperature=1, 
                max_tokens=1000 
            )
            
            if response.usage and response.usage.total_tokens > highest_token_count:
                highest_token_count = response.usage.total_tokens
            
            response_message = response.choices[0].message
            
            if response_message.tool_calls:
                messages_to_send.append(response_message.model_dump(exclude_none=True))
                
                # 1. Prepare lists to hold our tasks and their corresponding metadata
                search_tasks = []
                tool_call_metadata = []

                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "web_search":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            search_query = args.get("query")
                            # Queue the asynchronous function without awaiting it yet
                            search_tasks.append(perform_web_search(search_query))
                            tool_call_metadata.append(tool_call)
                        except json.JSONDecodeError:
                            # Create a quick dummy async function to return the error so the lists stay aligned
                            async def return_error(): return "System error: Invalid JSON."
                            search_tasks.append(return_error())
                            tool_call_metadata.append(tool_call)

                # 2. Fire all queued searches simultaneously and wait for them to finish
                if search_tasks:
                    completed_results = await asyncio.gather(*search_tasks)
                    
                    # 3. Match the gathered results back to their specific tool call IDs
                    for tool_call, result_text in zip(tool_call_metadata, completed_results):
                        messages_to_send.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": result_text
                        })
                
                second_response = await lm_client.chat.completions.create(
                    model="local-model",
                    messages=messages_to_send,
                    temperature=1,
                    max_tokens=1000
                )
                
                if second_response.usage and second_response.usage.total_tokens > highest_token_count:
                    highest_token_count = second_response.usage.total_tokens
                    
                final_reply = second_response.choices[0].message.content
            else:
                final_reply = response_message.content
            
            # Grab the lock again to safely save the AI's reply
            async with global_lock:
                global_data["history"].append({"role": "assistant", "content": final_reply})
                
                if len(global_data["history"]) > MAX_HISTORY_LENGTH:
                    forgotten_messages = global_data["history"][:2]
                    global_data["history"] = global_data["history"][-MAX_HISTORY_LENGTH:]
                    
                    target_user_id = forgotten_messages[0].get("user_id")
                    target_user_name = forgotten_messages[0].get("user_name")
                    
                    if target_user_id:
                        asyncio.create_task(update_user_memory(target_user_id, target_user_name, forgotten_messages))
                    
                save_json(GLOBAL_DATA_FILE, global_data)
            
            chunk_size = 1990
            chunks = []
            remaining_text = final_reply

            while len(remaining_text) > 0:
                if len(remaining_text) <= chunk_size:
                    chunks.append(remaining_text)
                    break
                
                split_index = remaining_text.rfind('\n', 0, chunk_size)
                if split_index == -1:
                    split_index = remaining_text.rfind(' ', 0, chunk_size)
                    
                if split_index == -1:
                    split_index = chunk_size
                else:
                    split_index += 1 
                    
                chunks.append(remaining_text[:split_index])
                remaining_text = remaining_text[split_index:]

            for index, chunk in enumerate(chunks):
                if index == 0:
                    await message.reply(chunk)
                else:
                    async with message.channel.typing():
                        await asyncio.sleep(1.5) 
                        await message.channel.send(chunk)
            
        except Exception as e:
            print(f"Error connecting to LM Studio: {e}")
            await message.reply("Oops! I couldn't process that. Make sure LM Studio is running and your model supports tool-calling!")
            return

if TOKEN is None:
    print("❌ ERROR: Could not find DISCORD_BOT_TOKEN in the .env file!")
else:
    client.run(TOKEN)