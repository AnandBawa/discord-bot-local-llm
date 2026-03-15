import os
import io
import re
import json
import base64
import logging
import asyncio
import contextlib
import requests
import httpx
import time
from datetime import datetime

import fitz  # PyMuPDF
import aiohttp
import discord
from discord import app_commands
import aiosqlite
import chromadb 
from chromadb import Documents, EmbeddingFunction, Embeddings
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from ddgs import DDGS
from openai import AsyncOpenAI
from dotenv import load_dotenv

# ==========================================
# ENVIRONMENT & API SETUP
# ==========================================
load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
try:
    BOT_OWNER_ID = int(os.getenv('BOT_OWNER_ID', '0'))
except ValueError:
    BOT_OWNER_ID = 0

# --- LOGGING SETUP ---
class TerminalTruncatedFormatter(logging.Formatter):
    """Custom formatter to truncate long terminal outputs."""
    def format(self, record):
        formatted_message = super().format(record)
        max_len = 100  # <-- Adjust this number to make terminal logs wider or narrower
        if len(formatted_message) > max_len:
            return formatted_message[:max_len] + "... [truncated]"
        return formatted_message

log_format = '%(asctime)s | %(levelname)s | %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# 1. File Handler (Keeps the FULL log intact for debugging)
file_handler = logging.FileHandler("bot.log", encoding='utf-8')
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 2. Terminal Handler (Truncates long messages to keep your screen clean)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(TerminalTruncatedFormatter(log_format, datefmt=date_format))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)

# Silence noisy external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("ddgs").setLevel(logging.WARNING)

LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')
LLM_API_KEY = os.getenv('LLM_API_KEY', 'lm-studio')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'local-model')
EMB_MODEL_NAME = os.getenv('EMB_MODEL_NAME', 'local-model')
VISION_ENABLED = os.getenv('VISION_ENABLED', 'True').lower() in ('true', '1', 'yes')
FALLBACK_BASE_URL = os.getenv('FALLBACK_BASE_URL', '')
FALLBACK_API_KEY = os.getenv('FALLBACK_API_KEY', '')
FALLBACK_MODEL_NAME = os.getenv('FALLBACK_MODEL_NAME', '')
CIRCUIT_BREAKER_COOLDOWN = 60.0  # Seconds to bypass the local node after a failure
local_node_dead_until = 0.0      # Timestamp tracker

# 2 seconds to detect an offline server, 120 seconds to wait for a slow local AI reply
llm_timeout = httpx.Timeout(120.0, connect=2.0)

lm_client = AsyncOpenAI(
    base_url=LLM_BASE_URL, 
    api_key=LLM_API_KEY,
    timeout=llm_timeout,
    max_retries=0  # <-- Critical: Prevents the silent 2x retry loop!
)

fallback_client = None
if FALLBACK_BASE_URL and FALLBACK_API_KEY:
    fallback_client = AsyncOpenAI(base_url=FALLBACK_BASE_URL, api_key=FALLBACK_API_KEY)

intents = discord.Intents.default()
intents.message_content = True

class JinaAPIEmbeddingFunction:
    """Custom explicit Jina API handler that accepts dynamic tasks."""
    def __init__(self, api_key: str, model_name: str = "jina-embeddings-v5-text-small"):
        self.api_key = api_key
        self.model_name = model_name

    def embed(self, input_texts: list[str], task: str) -> list[list[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model_name,
            "input": input_texts,
            "task": task,
            "normalized": True
        }
        response = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=data, timeout=15.0)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    
class LocalAPIEmbeddingFunction:
    """Custom explicit Local API handler with strict fail-fast timeouts."""
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name

    def embed(self, input_texts: list[str], task: str = None) -> list[list[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model_name,
            "input": input_texts
        }
        # Tuple: (2 seconds to connect, 15 seconds to read)
        response = requests.post(f"{self.base_url}/embeddings", headers=headers, json=data, timeout=(2.0, 15.0))
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

class ResilientEmbeddingFunction:
    """Thread-safe fallback wrapper for manual embedding."""
    def __init__(self, primary_ef, fallback_ef=None):
        self.primary_ef = primary_ef  
        self.fallback_ef = fallback_ef 

    def embed(self, input_texts: list[str], task: str) -> list[list[float]]:
        global local_node_dead_until
        
        # 1. If the breaker is tripped, go straight to the cloud
        if time.time() < local_node_dead_until and self.fallback_ef:
            return self.fallback_ef.embed(input_texts, task)
            
        # 2. Otherwise, try local
        try:
            result = self.primary_ef.embed(input_texts, task)
            local_node_dead_until = 0.0 # Reset breaker on success!
            return result
        except Exception as e:
            logging.warning(f"⚠️ Local Embedding failed: {e}. Tripping circuit breaker and routing to cloud...")
            local_node_dead_until = time.time() + CIRCUIT_BREAKER_COOLDOWN
            if self.fallback_ef:
                return self.fallback_ef.embed(input_texts, task)
            raise e

class MyAIClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_lock = None
        self.db_lock = None
        self.llm_queue = None
        self.db_conn = None
        self.vector_db = None
        self.memory_collection = None
        
        # --- NEW: Internal State moved from global scope ---
        self.highest_token_count = 0                             
        self.background_tasks = set()                            
        self.pending_deletions = {}                              
        self.server_personas_cache = {}

    async def setup_hook(self):
        self.memory_lock = asyncio.Lock()
        self.db_lock = asyncio.Lock()
        self.llm_queue = asyncio.Semaphore(3)
        
        self.db_conn = await aiosqlite.connect(DB_FILE)
        await self.db_conn.execute('PRAGMA journal_mode=WAL;')
        await self.db_conn.commit()  # <--- NEW: Close the pending transaction
        
        logging.info("Vacuuming SQLite database...")
        await self.db_conn.execute('VACUUM;') 
        await init_db(self.db_conn)
        
        # 1A. Primary Local Embedding Function (LM Studio) - Fail Fast
        primary_ef = LocalAPIEmbeddingFunction(
            api_key=LLM_API_KEY, 
            base_url=LLM_BASE_URL, 
            model_name=EMB_MODEL_NAME
        )
        
        # 1B. Fallback Cloud Embedding Function (Jina API)
        fallback_emb_key = os.getenv('FALLBACK_EMB_API_KEY', '')
        fallback_ef = None
        if fallback_emb_key:
            fallback_ef = JinaAPIEmbeddingFunction(
                api_key=fallback_emb_key,
                model_name="jina-embeddings-v5-text-small"
            )
            
        # 1C. Wrap them together
        self.custom_ef = ResilientEmbeddingFunction(primary_ef, fallback_ef)
        
        self.vector_db = await asyncio.to_thread(chromadb.PersistentClient, path="./chroma_storage")
        
        # 2. IMPORTANT: Remove embedding_function=self.custom_ef
        # We will now manually embed to ensure thread-safe task swapping
        self.memory_collection = await asyncio.to_thread(
            self.vector_db.get_or_create_collection, 
            name="user_memories",
            embedding_function=None, # <-- THE PROPER FIX
            metadata={"hnsw:space": "cosine"}
        )
        
        await tree.sync()
        logging.info('🔄 Databases loaded and Slash Commands synced globally!')

    async def close(self):
        logging.info("Shutdown signal received. Waiting for background memory tasks to finish...")
        if self.background_tasks:
            logging.info(f"Waiting for {len(self.background_tasks)} memory tasks to finish...")
            try:
                # Give background tasks 10 seconds to finish before forcing a kill
                await asyncio.wait_for(asyncio.gather(*self.background_tasks, return_exceptions=True), timeout=10.0)
            except asyncio.TimeoutError:
                logging.warning("Shutdown timeout reached. Killing pending memory tasks.")
            
        if self.db_conn:
            await self.db_conn.close()
            
        logging.info("Disconnecting from Discord. Goodbye!")
        await super().close()

    def register_deletion(self, key):
        current_time = datetime.now().timestamp()
        self.pending_deletions[key] = current_time
        stale_keys = [k for k, v in self.pending_deletions.items() if current_time - v > WIPE_REQUEST_EXPIRY]
        for k in stale_keys:
            del self.pending_deletions[k]

client = MyAIClient(intents=intents)
tree = app_commands.CommandTree(client)

# ==========================================
# GLOBAL STATE & CONFIGURATION
# ==========================================
DB_FILE = "bot_database.db"

# --- MODEL & CONTEXT LIMITS ---
MAX_HISTORY_LENGTH = 100            # Max messages kept in SQLite short-term history before summarizing to vector memory
MAX_TOOL_ITERATIONS = 3             # Max consecutive tool calls (like web searches) the AI can make in a single turn
LLM_TEMPERATURE = 1.0               # Creativity/randomness of the AI's standard chat responses (higher = more creative)
LLM_MAX_TOKENS = 4096               # Maximum output token length for standard chat responses
MEMORY_TEMPERATURE = 0.1            # Creativity for fact extraction (kept very low to ensure strict, factual JSON output)
MEMORY_MAX_TOKENS = 500             # Maximum output token length when the AI is generating the memory JSON array
MEMORY_DISTANCE_THRESHOLD = float(os.getenv('MEMORY_DISTANCE_THRESHOLD', '0.4')) # Cosine distance threshold for RAG
MEMORY_DEDUPLICATION_THRESHOLD = 0.15 # Strict threshold to prevent saving nearly identical facts
MEMORY_MAX_MSG_CHARS = 2000         # Max characters per message fed into the background memory extractor

# --- HARDWARE & PARSING LIMITS ---
MAX_FILE_SIZE = 10 * 1024 * 1024    # 10MB hard limit for Discord attachments and web scraper downloads
MAX_PDF_PAGES = 15                  # Maximum number of pages to read from an uploaded PDF
MAX_TEXT_EXTRACTION_LENGTH = 40000  # Character limit for text extracted from PDFs or scraped web pages
MAX_IMAGE_DIMENSION = 1024          # Uploaded images are resized to this max width/height to save VRAM
IMAGE_COMPRESSION_QUALITY = 85      # JPEG compression quality used when downscaling images via Pillow
SCRAPER_TIMEOUT = 15                # Seconds to wait for Jina web scraping OR large native file downloads
WEB_SEARCH_MAX_RESULTS = 3          # Number of DuckDuckGo search result snippets to return to the AI

# --- DISCORD & SYSTEM LIMITS ---
DISCORD_CHUNK_LIMIT = 1980          # Max character limit per Discord message (safely below Discord's 2000 limit)
CHUNK_MESSAGE_DELAY = 1.5           # Seconds to wait between sending message chunks to avoid Discord rate limits
WIPE_REQUEST_EXPIRY = 3600          # Seconds before a pending memory deletion request expires from the cache
DEFAULT_PERSONA = "You are a neutral, conversational AI." # Fallback system prompt if no custom role is set for a server                             

# ==========================================
# 1. CORE DATABASE & UTILITY FUNCTIONS
# ==========================================

async def init_db(db_conn):
    await db_conn.execute('''CREATE TABLE IF NOT EXISTS server_config (server_id TEXT PRIMARY KEY, prompt TEXT)''')
    await db_conn.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, server_id TEXT, role TEXT, content TEXT, user_id TEXT, user_name TEXT)''')
    await db_conn.commit()

@contextlib.asynccontextmanager
async def safe_typing(channel):
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
    def __init__(self, data): 
        self.data = data
    async def read(self): 
        return self.data

def clean_json_response(text):
    """Utility to strip markdown wrappers from LLM JSON outputs."""
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except json.JSONDecodeError:
        return []
    
async def send_chunked_message(target, text: str, is_interaction_followup=False):
    """Chunks and sends long texts to bypass Discord's character limit."""
    remaining_text = text
    is_first = True
    in_code_block = False
    
    while len(remaining_text) > 0:
        chunk_limit = DISCORD_CHUNK_LIMIT
        if len(remaining_text) <= chunk_limit: 
            chunk = remaining_text
            remaining_text = ""
        else:
            split_index = remaining_text.rfind('\n', 0, chunk_limit)
            if split_index == -1: split_index = remaining_text.rfind(' ', 0, chunk_limit)
            if split_index == -1: split_index = chunk_limit
            else: split_index += 1 
            
            chunk = remaining_text[:split_index]
            remaining_text = remaining_text[split_index:]

        code_markers = chunk.count("```")
        if in_code_block: chunk = "```\n" + chunk
        if code_markers % 2 != 0: in_code_block = not in_code_block
        if in_code_block and len(remaining_text) > 0: chunk += "\n```"
        
        try:
            if is_first:
                if is_interaction_followup:
                    await target.followup.send(chunk)
                else:
                    try: 
                        await target.reply(chunk)
                    except discord.HTTPException as e:
                        if e.code == 50035: await target.channel.send(f"<@{target.author.id}> {chunk}")
                        else: raise e
                is_first = False
            else:
                channel = target.channel if hasattr(target, 'channel') else target
                async with safe_typing(channel): 
                    await asyncio.sleep(CHUNK_MESSAGE_DELAY) 
                await channel.send(chunk)
        except discord.Forbidden:
            break

# ==========================================
# 2. MEDIA PROCESSING FUNCTIONS
# ==========================================

def extract_pdf_text(pdf_bytes):
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
    try:
        with Image.open(io.BytesIO(img_bytes)) as pil_img:
            if pil_img.mode in ("RGBA", "P"): 
                pil_img = pil_img.convert("RGB")
            pil_img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=IMAGE_COMPRESSION_QUALITY)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logging.warning(f"⚠️ Pillow failed to process image: {e}")
        return None
    
def process_sticker_bytes(sticker_bytes):
    try:
        return base64.b64encode(sticker_bytes).decode('utf-8')
    except Exception as e:
        logging.warning(f"⚠️ Failed to encode sticker bytes: {e}")
        return None

# ==========================================
# 3. AUTONOMOUS TOOLS & SCRAPING
# ==========================================

async def resolve_tool_error(error_message):
    return error_message

async def perform_web_search(query):
    if not query or not isinstance(query, str): 
        return "Search error: Invalid or missing search query."
    now = datetime.now()
    current_date = now.strftime('%B %d, %Y')
    date_variations = [now.strftime('%B %d, %Y'), now.strftime('%B %d %Y'), f"{now.strftime('%B')} {now.day}, {now.strftime('%Y')}", f"{now.strftime('%B')} {now.day} {now.strftime('%Y')}", now.strftime('%B %d'), f"{now.strftime('%B')} {now.day}", now.strftime('%B %Y'), now.strftime('%Y')]

    # Clean the query of dates
    clean_query = query
    for variation in date_variations: 
        clean_query = clean_query.replace(variation, "")
        
    # <-- NEW: Strip emojis and special characters that crash search engines
    clean_query = re.sub(r'[^\w\s\-\.]', '', clean_query) 
    
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
    direct_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.pdf') 
    is_direct_file = url.split('?')[0].lower().endswith(direct_extensions)
    
    target_url = url if is_direct_file else f"https://r.jina.ai/{url}"
    log_msg = f"📥 Fetching direct file: {url}" if is_direct_file else f"📡 Jina attempting to fetch: {url}"
    
    logging.info(log_msg)
    
    try:
        async with aiohttp.ClientSession() as session:
            # Full browser spoofing to bypass 403 Forbidden firewalls, 
            # but NO 'Accept-Encoding' so aiohttp safely decompresses the data automatically!
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            if not is_direct_file:
                headers.update({"X-Return-Format": "markdown", "X-No-Cache": "true"})
                
            async with session.get(target_url, timeout=SCRAPER_TIMEOUT, headers=headers) as response:
                if response.status not in (200, 206):
                    return {"type": "error", "data": f"Failed to access URL (HTTP {response.status})"}
                
                content_length = response.headers.get('Content-Length')
                if content_length and content_length.isdigit() and int(content_length) > MAX_FILE_SIZE:
                    return {"type": "error", "data": "File skipped: Exceeds the 10MB limit."}
                
                file_bytes = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    file_bytes.extend(chunk)
                    if len(file_bytes) > MAX_FILE_SIZE:
                        return {"type": "error", "data": "File skipped: Exceeds the 10MB limit."}
                
                file_bytes = bytes(file_bytes) 
                
                content_type = response.headers.get('Content-Type', '').lower()
                url_lower = target_url.split('?')[0].lower()
                
                # Robust routing based on BOTH content type and URL extension
                if 'image' in content_type or url_lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                    return {"type": "image", "data": file_bytes}
                elif 'pdf' in content_type or url_lower.endswith('.pdf'):
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
                    except Exception:
                        return {"type": "error", "data": "Webpage content could not be decoded."}
                else:
                    return {"type": "error", "data": f"Unsupported URL media type ({content_type})."}
                    
    except asyncio.TimeoutError:
        return {"type": "error", "data": "The website took too long to respond."}
    except Exception as e:
        return {"type": "error", "data": f"Unexpected scraper error: {str(e)}"}

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
# 4. CHROMA VECTOR MEMORY MANAGEMENT
# ==========================================

async def update_user_memory(server_id, user_id, user_name, forgotten_messages):
    task_start_time = datetime.now().timestamp()
            
    chat_log = ""
    for msg in forgotten_messages:
        if str(msg.get('user_id', '')) != str(user_id):
            continue

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
            
        # Ensure it's a string, then check length against our new constant
        raw_content_str = str(raw_content)
        if len(raw_content_str) > MEMORY_MAX_MSG_CHARS:
            raw_content_str = raw_content_str[:MEMORY_MAX_MSG_CHARS] + "\n...[System Note: Content truncated for memory efficiency]"
            
        chat_log += f"{msg['role'].capitalize()}: {raw_content_str}\n"
        
    memory_prompt = (
            f"You are a strict, automated data-extraction system. Your job is to extract permanent, long-term facts about the user '{user_name}' from the chat log below.\n\n"
            "RULES:\n"
            "1. IGNORE temporary states, moods, current debugging tasks, or conversational greetings.\n"
            "2. ONLY extract immutable facts (e.g., tech stack, geographical location, hobbies, career, strong preferences, relationships).\n"
            f"3. Phrase each fact in the third person, explicitly starting with the user's name (e.g., \"{user_name} prefers dark mode\", \"{user_name} works as a DevOps engineer\").\n"
            "4. Output ONLY a raw, valid JSON array of strings. Do not wrap it in markdown blockquotes, do not use dictionaries, and do not add conversational text.\n"
            "5. If no permanent facts are present in the log, you must output exactly: []\n\n"
            f"CHAT LOG:\n{chat_log}"
        )
    
    try:
        async with client.llm_queue:
            try:
                # 1. Attempt local LLM
                response = await lm_client.chat.completions.create(
                    model=LLM_MODEL_NAME, 
                    messages=[{"role": "user", "content": memory_prompt}], 
                    temperature=MEMORY_TEMPERATURE, 
                    max_tokens=MEMORY_MAX_TOKENS
                )
            except Exception as e:
                # 2. Route to fallback if local fails
                if fallback_client:
                    logging.warning(f"⚠️ Local LLM failed during memory extraction ({e}). Routing to Fallback API...")
                    response = await fallback_client.chat.completions.create(
                        model=FALLBACK_MODEL_NAME, 
                        messages=[{"role": "user", "content": memory_prompt}], 
                        temperature=MEMORY_TEMPERATURE, 
                        max_tokens=MEMORY_MAX_TOKENS
                    )
                else:
                    raise e # Crash normally if no fallback is configured
                    
        content = response.choices[0].message.content
        new_memory_json = content.strip() if content else "[]"
        
        facts_list = clean_json_response(new_memory_json)
        
        if facts_list and isinstance(facts_list, list):
            if client.pending_deletions.get(f"{server_id}_{user_id}", 0) > task_start_time or client.pending_deletions.get(f"wipe_{server_id}", 0) > task_start_time:
                return
            
            # --- NEW: Semantic Deduplication Pipeline ---
            unique_facts = []
            
            # 1. Embed the raw facts to query the database
            raw_embeddings = await asyncio.to_thread(
                client.custom_ef.embed, 
                facts_list, 
                "retrieval.query" # Use query task to search existing memories
            )
            
            # 2. Check each new fact against the user's existing database
            for i, (fact, emb) in enumerate(zip(facts_list, raw_embeddings)):
                try:
                    existing = await asyncio.to_thread(
                        client.memory_collection.query,
                        query_embeddings=[emb],
                        n_results=1,
                        # Strictly limit the search to THIS specific user in THIS server
                        where={"$and": [{"server_id": server_id}, {"user_id": str(user_id)}]}, 
                        include=["distances"]
                    )
                    
                    # 3. If a nearly identical fact exists, skip it
                    if existing and existing['distances'] and existing['distances'][0]:
                        closest_distance = existing['distances'][0][0]
                        if closest_distance < MEMORY_DEDUPLICATION_THRESHOLD:
                            logging.info(f"♻️ [Memory] Skipped duplicate fact (Distance: {closest_distance:.3f}): {fact}")
                            continue 
                except Exception as e:
                    logging.warning(f"⚠️ Deduplication check failed for fact '{fact}': {e}")

                # 4. If it passed the check, format it for permanent storage
                current_date = datetime.now().strftime('%Y-%m-%d')
                unique_facts.append(f"[Recorded on {current_date}]: {fact}")
                
            # 5. Save ONLY the unique facts to ChromaDB
            if unique_facts:
                timestamp = datetime.now().timestamp()
                fact_ids = [f"{server_id}_{user_id}_{timestamp}_{i}" for i in range(len(unique_facts))]
                metadatas = [{"server_id": server_id, "user_id": str(user_id), "user_name": user_name} for _ in unique_facts]

                # Re-embed the final timestamped strings for permanent storage
                doc_embeddings = await asyncio.to_thread(
                    client.custom_ef.embed, 
                    unique_facts, 
                    "retrieval.passage"
                )
                
                async with client.memory_lock: 
                    await asyncio.to_thread(
                        client.memory_collection.add,
                        documents=unique_facts,
                        embeddings=doc_embeddings,
                        metadatas=metadatas,
                        ids=fact_ids
                    )
                logging.info(f"💾 [Memory] Added {len(unique_facts)} new vector facts for {user_name}.")
            else:
                logging.info(f"♻️ [Memory] No new unique facts to add for {user_name}.")
            
    except Exception as e: 
        logging.error(f"Failed to update vector memory for {user_name}: {e}")

async def process_memories_concurrently(server_id, users_dict, forgotten_messages):
    tasks = []
    for uid, uname in users_dict.items():
        tasks.append(update_user_memory(server_id, uid, uname, forgotten_messages))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

# ==========================================
# 5. SLASH COMMANDS
# ==========================================

@tree.command(name="help", description="Learn how to interact with the AI and view system limits.")
async def cmd_help(interaction: discord.Interaction):
    help_text = f"""**How to interact with me:**
• **`@{client.user.name} [message]`** - Chat, ask questions, or analyze attached files and links.
• **Reply to me** and tag me to seamlessly resume an exact topic.

**Slash Commands:**
• **`/help`** - Display this guide.
• **`/status`** - Check diagnostics and ping.
• **`/role`** - View, change, or clear the AI's personality.
• **`/memory`** - View tracked users, read specific memories, or clear your own data.
• **`/clear`** - Clear the temporary conversation history (core facts retained).
• **`/force-forget`** - *(Admin/Owner)* Purge all stored data for a specific user.
• **`/admin_wipe_server`** - *(Admin/Owner)* Factory reset all data for this server.
"""
    await interaction.response.send_message(help_text, ephemeral=True)

@tree.command(name="status", description="Check bot diagnostics, ping, and AI model status.")
async def cmd_status(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False)
    ping_ms = round(client.latency * 1000)
    
    # 1. Ping the local AI node (Using our 2.0s fail-fast timeout!)
    try:
        await lm_client.models.list()
        active_llm = f"🟢 Local ({LLM_MODEL_NAME})"
        active_emb = f"🟢 Local ({EMB_MODEL_NAME})"
    except Exception:
        # 2. If local fails, check if we have a cloud failover ready
        if fallback_client:
            active_llm = f"🟡 Cloud Fallback ({FALLBACK_MODEL_NAME})"
            active_emb = f"🟡 Cloud Fallback (jina-embeddings-v5-text-small)"
        else:
            active_llm = "🔴 Offline (No fallback configured)"
            active_emb = "🔴 Offline (No fallback configured)"
        
    # 3. Fetch Database Stats
    cursor = await client.db_conn.execute("SELECT COUNT(*) FROM chat_history WHERE server_id = ?", (str(interaction.guild_id),))
    history_length = (await cursor.fetchone())[0]
        
    diagnostics = (
        f"**Bot Diagnostics & Status**\n\n"
        f"• **Discord Ping:** `{ping_ms}ms`\n"
        f"• **Active LLM:** `{active_llm}`\n"
        f"• **Active Embeddings:** `{active_emb}`\n"
        f"• **Peak Context Used:** `{client.highest_token_count} tokens`\n"
        f"• **Current History:** `{history_length}/{MAX_HISTORY_LENGTH} messages`"
    )
    await interaction.followup.send(diagnostics)

@tree.command(name="role", description="View or change the AI's personality for this server.")
@app_commands.describe(prompt="The new persona (leave blank to view current, type 'clear' to reset)")
async def cmd_role(interaction: discord.Interaction, prompt: str = None):
    server_id = str(interaction.guild_id)
    await interaction.response.defer()
    
    if not prompt:
        current_role = client.server_personas_cache.get(server_id, DEFAULT_PERSONA)
        await interaction.followup.send(f"**Current Server Persona:**\n> *{current_role}*")
        return

    async with client.db_lock:
        cursor = await client.db_conn.execute("SELECT role, content, user_id, user_name FROM chat_history WHERE server_id = ? ORDER BY id ASC", (server_id,))
        rows = await cursor.fetchall()
        if rows:
            history_to_save = [{"role": r[0], "content": r[1], "user_id": r[2], "user_name": r[3]} for r in rows]
            users_in_history = {msg["user_id"]: msg["user_name"] for msg in history_to_save if msg.get("user_id")}
            task = asyncio.create_task(process_memories_concurrently(server_id, users_in_history, history_to_save))
            client.background_tasks.add(task)
            task.add_done_callback(client.background_tasks.discard)
            
        await client.db_conn.execute("DELETE FROM chat_history WHERE server_id = ?", (server_id,))
        
        if prompt.lower() == 'clear':
            await client.db_conn.execute("INSERT INTO server_config (server_id, prompt) VALUES (?, ?) ON CONFLICT(server_id) DO UPDATE SET prompt=excluded.prompt", (server_id, ""))
            client.server_personas_cache[server_id] = DEFAULT_PERSONA
            reply_text = f"✅ Server persona removed and history cleared! *(Recent memories saved)*\n\n**Current Persona:**\n> {DEFAULT_PERSONA}"
        else:
            await client.db_conn.execute("INSERT INTO server_config (server_id, prompt) VALUES (?, ?) ON CONFLICT(server_id) DO UPDATE SET prompt=excluded.prompt", (server_id, prompt))
            client.server_personas_cache[server_id] = prompt
            reply_text = f"✅ Saved server persona and cleared history for a fresh start! *(Recent memories saved)*\n> *{prompt}*"
            
        await client.db_conn.commit()

    await interaction.followup.send(reply_text)

@tree.command(name="clear", description="Clear the current conversation history (core facts retained).")
@app_commands.default_permissions(manage_messages=True)
async def cmd_clear(interaction: discord.Interaction):
    server_id = str(interaction.guild_id)
    await interaction.response.defer()

    async with client.db_lock:
        cursor = await client.db_conn.execute("SELECT role, content, user_id, user_name FROM chat_history WHERE server_id = ? ORDER BY id ASC", (server_id,))
        rows = await cursor.fetchall()
        if rows:
            history_to_save = [{"role": r[0], "content": r[1], "user_id": r[2], "user_name": r[3]} for r in rows]
            users_in_history = {msg["user_id"]: msg["user_name"] for msg in history_to_save if msg.get("user_id")}
            task = asyncio.create_task(process_memories_concurrently(server_id, users_in_history, history_to_save))
            client.background_tasks.add(task)
            task.add_done_callback(client.background_tasks.discard)
            
        await client.db_conn.execute("DELETE FROM chat_history WHERE server_id = ?", (server_id,))
        await client.db_conn.commit()
        
    await interaction.followup.send("🗑️ Server conversation history cleared! *(Recent memories saved)*")

@tree.command(name="admin_wipe_server", description="[ADMIN/OWNER] Complete factory reset of all data for this server.")
async def cmd_wipe_server(interaction: discord.Interaction):
    if not (interaction.user.id == BOT_OWNER_ID or interaction.permissions.administrator):
        await interaction.response.send_message("⛔ You must be a Server Admin or the Bot Owner to run this.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    server_id = str(interaction.guild_id)
    client.register_deletion(f"wipe_{server_id}")
    
    # Wipe ChromaDB Vector Memory
    async with client.memory_lock: 
        await asyncio.to_thread(client.memory_collection.delete, where={"server_id": server_id})
    # Wipe SQLite History
    async with client.db_lock:
        await client.db_conn.execute("DELETE FROM chat_history WHERE server_id = ?", (server_id,))
        await client.db_conn.commit()
        
    await interaction.followup.send("☢️ **SERVER WIPED.** All core memories and chat histories for **this specific server** have been erased.")


@tree.command(name="force-forget", description="[ADMIN/OWNER] Purge all stored data for a specific user.")
@app_commands.describe(target_user="The user whose memory you want to erase")
async def cmd_force_forget(interaction: discord.Interaction, target_user: discord.User): 
    if not (interaction.user.id == BOT_OWNER_ID or interaction.permissions.administrator):
        await interaction.response.send_message("⛔ You must be a Server Admin or the Bot Owner to run this.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    server_id = str(interaction.guild_id)
    user_id = str(target_user.id)
    client.register_deletion(f"{server_id}_{user_id}")
    
    async with client.memory_lock: 
        await asyncio.to_thread(client.memory_collection.delete, where={"$and": [{"server_id": server_id}, {"user_id": user_id}]})
        
    async with client.db_lock:
        await client.db_conn.execute("DELETE FROM chat_history WHERE server_id = ? AND user_id = ?", (server_id, user_id))
        await client.db_conn.commit()
        
    await interaction.followup.send(f"✅ **Force-Forget Successful:** All memory and chat history for {target_user.mention} has been permanently purged.")

@tree.command(name="memory", description="View tracked users, read specific memories, or clear your own data.")
@app_commands.describe(action="Choose an action", target_user="Name of the user to search for (if reading)")
@app_commands.choices(action=[
    app_commands.Choice(name="List active users", value="list"),
    app_commands.Choice(name="Read a user's memory", value="read"),
    app_commands.Choice(name="Clear my own memory", value="clear")
])
async def cmd_memory(interaction: discord.Interaction, action: app_commands.Choice[str], target_user: str = None):
    server_id = str(interaction.guild_id)
    user_id = str(interaction.user.id)
    await interaction.response.defer(ephemeral=(action.value == "clear"))

    if action.value == 'clear':
        client.register_deletion(f"{server_id}_{user_id}")
        async with client.memory_lock:
            await asyncio.to_thread(client.memory_collection.delete, where={"$and": [{"server_id": server_id}, {"user_id": user_id}]})
            
        async with client.db_lock:
            await client.db_conn.execute("DELETE FROM chat_history WHERE server_id = ? AND user_id = ?", (server_id, user_id))
            await client.db_conn.commit()
            
        await interaction.followup.send("🗑️ **Forget successful.** All your core memories and recent chat history have been erased.")
        return

    if action.value == 'read' and target_user:
        # Search Vector DB for documents matching the user's name
        results = await asyncio.to_thread(
            client.memory_collection.get,
            where={"server_id": server_id},
            include=["documents", "metadatas"]
        )
        
        target_name = target_user.lower()
        found_facts = []
        actual_name = target_user
        
        if results and results['metadatas']:
            for doc, meta in zip(results['documents'], results['metadatas']):
                if target_name in meta.get("user_name", "").lower():
                    found_facts.append(doc)
                    actual_name = meta.get("user_name")
                    
        if not found_facts:
            memory_text = f"I couldn't find any memories for '{target_user}'."
        else:
            facts_list = "\n".join([f"• {f}" for f in found_facts])
            memory_text = f"**Facts known about {actual_name}:**\n{facts_list}"
    else:
        # List action
        results = await asyncio.to_thread(
            client.memory_collection.get,
            where={"server_id": server_id},
            include=["metadatas"]
        )
        
        unique_users = set()
        if results and results['metadatas']:
            for meta in results['metadatas']:
                unique_users.add(meta.get("user_name", "Unknown"))
                
        memory_text = "**Tracked Active Users (with saved facts)**\n\n"
        if unique_users:
            for uname in unique_users:
                memory_text += f"• {uname}\n"
        else:
            memory_text += "*No core memories found for this server yet.*"
    
    await send_chunked_message(interaction, memory_text, is_interaction_followup=True)

# ==========================================
# 6. PIPELINE MODULES
# ==========================================

async def extract_message_context(message, clean_message, user_name):
    ephemeral_context = ""
    image_attachments = []
    
    for att in message.attachments:
        if att.content_type and att.content_type.startswith('image/'):
            if att.size <= MAX_FILE_SIZE: 
                image_attachments.append(att)
            else: 
                clean_message += f"\n[System note: Attached image '{att.filename}' ignored because it exceeds the limit.]"
        elif not att.filename.lower().endswith('.pdf'):
            clean_message += f"\n[System note: The user attached an unsupported file type '{att.filename}'. Politely inform them that you can only read Images, PDFs, and Web Links.]"

    valid_stickers = [s for s in message.stickers if s.format != discord.StickerFormatType.lottie]
    
    pdf_attachments = [att for att in message.attachments if att.filename.lower().endswith('.pdf') and att.size <= MAX_FILE_SIZE]
    if pdf_attachments:
        async with safe_typing(message.channel):
            for pdf in pdf_attachments:
                try:
                    pdf_bytes = await pdf.read()
                    pdf_text = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
                    if len(pdf_text) > MAX_TEXT_EXTRACTION_LENGTH: 
                        pdf_text = pdf_text[:MAX_TEXT_EXTRACTION_LENGTH] + "\n...[Content Truncated due to length limit]"
                    logging.info(f"📎 AI successfully extracted UPLOADED PDF: {pdf.filename}")
                    ephemeral_context += f"\n\n[Extracted PDF Content from {pdf.filename}]:\n{pdf_text}"
                    clean_message += f"\n[System note: User attached PDF '{pdf.filename}']"
                except discord.HTTPException:
                    logging.warning(f"⚠️ Discord CDN failed to provide PDF: {pdf.filename}")
                    clean_message += f"\n[System note: The attached PDF '{pdf.filename}' could not be downloaded from Discord's servers.]"

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
            
            # --- NEW: Extract and process PDFs from the replied message ---
            replied_pdfs = [att for att in replied_msg.attachments if att.filename.lower().endswith('.pdf') and att.size <= MAX_FILE_SIZE]
            if replied_pdfs:
                async with safe_typing(message.channel):
                    for pdf in replied_pdfs:
                        try:
                            pdf_bytes = await pdf.read()
                            pdf_text = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
                            if len(pdf_text) > MAX_TEXT_EXTRACTION_LENGTH: 
                                pdf_text = pdf_text[:MAX_TEXT_EXTRACTION_LENGTH] + "\n...[Content Truncated due to length limit]"
                            logging.info(f"📎 AI successfully extracted REPLIED PDF: {pdf.filename}")
                            ephemeral_context += f"\n\n[Extracted PDF Content from replied message ({pdf.filename})]:\n{pdf_text}"
                        except discord.HTTPException:
                            logging.warning(f"⚠️ Discord CDN failed to provide REPLIED PDF: {pdf.filename}")
                            clean_message += f"\n[System note: The replied PDF '{pdf.filename}' could not be downloaded.]"

        except Exception as e: 
            logging.warning(f"⚠️ Could not fetch the replied message: {e}")

    url_pattern = r'(https?://[^\s<>]+)'
    found_urls = re.findall(url_pattern, clean_message)
    if found_urls:
        scraped_texts = []
        async with safe_typing(message.channel):
            fetch_tasks = [fetch_url_content(url) for url in found_urls]
            results = await asyncio.gather(*fetch_tasks)
            for url, url_result in zip(found_urls, results):
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
    api_user_content = []
    db_user_content_obj = [] 
    
    ai_text_part = f"{user_name}: {clean_message}{ephemeral_context}" if (clean_message or ephemeral_context) else f"{user_name}: What is in this image?"
    db_text_part = f"{user_name}: {clean_message}" if clean_message else f"{user_name}: [Media attached]"

    if image_attachments or valid_stickers:
        api_user_content.append({"type": "text", "text": ai_text_part})
        db_user_content_obj.append({"type": "text", "text": db_text_part}) 
        
        # --- NEW: Check if Vision is enabled before processing ---
        if not VISION_ENABLED:
            api_user_content.append({"type": "text", "text": "\n[System note: The user attached an image or sticker, but your vision capabilities are currently disabled. Politely inform them you cannot see it.]"})
            db_user_content_obj.append({"type": "text", "text": "[Media attached but Vision is disabled]"})
        else:
            # Proceed with normal image processing
            for img in image_attachments:
                try:
                    img_bytes = await img.read()
                    img_b64 = await asyncio.to_thread(process_image_bytes, img_bytes)
                    if img_b64:
                        api_user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
                        db_user_content_obj.append({"type": "text", "text": f"[Image attached: {getattr(img, 'filename', 'URL_Image')}]"})
                    else:
                        api_user_content.append({"type": "text", "text": f"\n[System note: The attached image '{getattr(img, 'filename', 'URL_Image')}' was corrupted and skipped.]"})
                        db_user_content_obj.append({"type": "text", "text": "[Corrupted image skipped]"})
                except discord.HTTPException:
                    api_user_content.append({"type": "text", "text": f"\n[System note: Discord servers failed to provide the image '{getattr(img, 'filename', 'URL_Image')}'.]"})
                    db_user_content_obj.append({"type": "text", "text": "[Failed to download image]"})
                
            for sticker in valid_stickers:
                try:
                    sticker_bytes = await sticker.read()
                    sticker_b64 = await asyncio.to_thread(process_sticker_bytes, sticker_bytes)
                    if sticker_b64:
                        api_user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{sticker.format.name};base64,{sticker_b64}"}})
                        db_user_content_obj.append({"type": "text", "text": f"[Sticker attached: {sticker.name}]"})
                    else:
                        api_user_content.append({"type": "text", "text": f"\n[System note: The attached sticker '{sticker.name}' was corrupted and skipped.]"})
                        db_user_content_obj.append({"type": "text", "text": "[Corrupted sticker skipped]"})
                except discord.HTTPException:
                    api_user_content.append({"type": "text", "text": f"\n[System note: Discord servers failed to provide the sticker '{sticker.name}'.]"})
                    db_user_content_obj.append({"type": "text", "text": "[Failed to download sticker]"})
    else: 
        api_user_content = ai_text_part
        db_user_content_obj = db_text_part 

    return api_user_content, db_user_content_obj

async def build_ai_context(server_id, author_id, api_user_content):
    current_system_prompt = (
        f"Today's date is {datetime.now().strftime('%B %d, %Y')}.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. EXTREME BREVITY: Answer in 1-3 sentences unless asked otherwise.\n"
        "2. DOCUMENT ANALYSIS: You will receive webpage and PDF data in Markdown format. Use headers (#), lists (*), and bold text within that data to identify key information accurately.\n"
        "3. SEARCH POLICY: Only use `web_search` if the provided URL content or attachments do not contain the answer. If a URL is provided, prioritize its content first.\n"
        "4. MULTI-USER CHAT: Address users by their names when appropriate.\n"
        "5. MEMORY USAGE (CRITICAL): Use user facts silently to guide your context. NEVER repeat or summarize these facts back to the user unless they explicitly ask what you remember about them. If contradicting facts exist, always prioritize the most recently recorded data.\n"
        "6. STRICT RULE: Do not use emojis unless your persona requires it.\n"
        "7. MODEL INQUIRIES: If the user asks about your AI model, version, or underlying technology, politely tell them to use the `/status` command.\n"
        "8. IMAGE MEMORY: When a user uploads an image, ALWAYS begin your response with a brief, 1-sentence description of what you see before answering their prompt."
    )
    
    # --- SEMANTIC MEMORY RETRIEVAL (RAG) ---
    user_context_str = ""
    query_text = ""
    
    # Safely extract text string from api_user_content for the semantic search
    if isinstance(api_user_content, str):
        query_text = api_user_content
    elif isinstance(api_user_content, list):
        query_text = " ".join([item.get("text", "") for item in api_user_content if item.get("type") == "text"])

    if query_text.strip():
        try:
            query_embeddings = await asyncio.to_thread(
                client.custom_ef.embed, 
                [query_text], 
                "retrieval.query"
            )
            results = await asyncio.to_thread(
                client.memory_collection.query,
                query_embeddings=query_embeddings,
                n_results=5,
                where={"server_id": server_id},
                include=["documents", "metadatas", "distances"] # <-- NEW: Request distances
            )
            
            if results and results['documents'] and results['documents'][0]:
                retrieved_facts = results['documents'][0]
                retrieved_meta = results['metadatas'][0]
                retrieved_distances = results['distances'][0]
                
                for fact, meta, distance in zip(retrieved_facts, retrieved_meta, retrieved_distances):
                    if distance < MEMORY_DISTANCE_THRESHOLD: 
                        uname = meta.get("user_name", "User")
                        user_context_str += f"- {uname}: {fact}\n"
                        logging.info(f"✅ [Memory INJECTED] Distance: {distance:.3f} | Fact: {fact}")
                    else:
                        logging.info(f"❌ [Memory REJECTED] Distance: {distance:.3f} | Fact: {fact}")
                        
        except Exception as e:
            logging.error(f"Vector search failed: {e}")

    if server_id in client.server_personas_cache:
        base_persona = client.server_personas_cache[server_id]
    else:
        cursor = await client.db_conn.execute("SELECT prompt FROM server_config WHERE server_id = ?", (server_id,))
        row = await cursor.fetchone()
        base_persona = row[0] if row and row[0] else DEFAULT_PERSONA
        client.server_personas_cache[server_id] = base_persona 

    if user_context_str: 
        current_system_prompt += f"\n\nRELEVANT RECALLED FACTS ABOUT USERS (READ ONLY - SILENTLY USE THIS CONTEXT):\n{user_context_str}"
        
    current_system_prompt += f"\n\nYOUR ASSIGNED PERSONA AND ROLE:\n{base_persona}"
    system_message = {"role": "system", "content": current_system_prompt}

    cursor = await client.db_conn.execute("SELECT role, content FROM chat_history WHERE server_id = ? ORDER BY id ASC", (server_id,))
    api_history = []
    for r in await cursor.fetchall():
        try:
            parsed_content = json.loads(r[1])
            if isinstance(parsed_content, list):
                if len(parsed_content) > 0 and not isinstance(parsed_content[0], dict): 
                    parsed_content = r[1]
            elif not isinstance(parsed_content, str): 
                parsed_content = str(parsed_content)
        except (json.JSONDecodeError, TypeError): 
            parsed_content = r[1]
        api_history.append({"role": r[0], "content": parsed_content})
        
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
    
    api_kwargs = {
        "model": LLM_MODEL_NAME, 
        "messages": messages_to_send, 
        "temperature": LLM_TEMPERATURE, 
        "max_tokens": LLM_MAX_TOKENS
    }
    
    if not disable_search:
        api_kwargs["tools"] = tools_schema
        api_kwargs["tool_choice"] = "auto"
        
    # --- NEW: Fallback Wrapper (Latency Optimized) ---
    async def call_llm(**kwargs):
        global local_node_dead_until
        
        # 1. If the breaker is tripped, OR we fell back in a previous loop, use cloud
        if (time.time() < local_node_dead_until or api_kwargs.get("model") == FALLBACK_MODEL_NAME) and fallback_client:
            kwargs["model"] = FALLBACK_MODEL_NAME
            return await fallback_client.chat.completions.create(**kwargs)
            
        # 2. Otherwise, try local
        try:
            result = await lm_client.chat.completions.create(**kwargs)
            local_node_dead_until = 0.0 # Reset breaker on success!
            return result
        except Exception as e:
            if fallback_client:
                logging.warning(f"⚠️ Local LLM failed ({e}). Tripping circuit breaker and routing to cloud...")
                local_node_dead_until = time.time() + CIRCUIT_BREAKER_COOLDOWN
                api_kwargs["model"] = FALLBACK_MODEL_NAME
                kwargs["model"] = FALLBACK_MODEL_NAME
                return await fallback_client.chat.completions.create(**kwargs)
            raise e

    async with safe_typing(message.channel):
        async with client.llm_queue:
            try:
                max_iterations, current_iteration, final_reply = MAX_TOOL_ITERATIONS, 0, ""
                
                # Use our new wrapper instead of lm_client directly
                response = await call_llm(**api_kwargs)
                
                if response.usage and response.usage.total_tokens > client.highest_token_count: 
                    client.highest_token_count = response.usage.total_tokens
                    
                response_message = response.choices[0].message
                
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
                        
                        api_kwargs["messages"] = messages_to_send # Update the payload
                        response = await call_llm(**api_kwargs) # Call the wrapper
                        
                        if response.usage and response.usage.total_tokens > client.highest_token_count: 
                            client.highest_token_count = response.usage.total_tokens
                            
                        response_message = response.choices[0].message
                        current_iteration += 1
                    else:
                        final_reply = response_message.content
                        break
                
                # Check if the AI got stuck in a tool loop
                if current_iteration >= max_iterations and not final_reply and not response_message.content:
                    return "⚠️ *I needed to search too many things at once to answer that. Could you be more specific?*"
                
                return final_reply or response_message.content or "⚠️ *System error: Empty response.*"
            except Exception as e:
                error_str = str(e).lower()
                logging.error(f"Generation Error: {e}")
                if has_media and ("400" in error_str or "vision" in error_str or "image" in error_str):
                    await message.reply("⚠️ **Compatibility Error:** Your local AI model does not support image analysis.")
                else: 
                    await message.reply("Oops! I couldn't process that. Please check my terminal for details.")
                return None

async def save_and_send_response(message, server_id, user_name, db_user_content_obj, final_reply):
    db_user_content = json.dumps(db_user_content_obj) if isinstance(db_user_content_obj, list) else str(db_user_content_obj)
    
    # --- NEW: Apply the lock to the entire database transaction block ---
    async with client.db_lock:
        await client.db_conn.execute("INSERT INTO chat_history (server_id, role, content, user_id, user_name) VALUES (?, ?, ?, ?, ?)", (server_id, "user", db_user_content, str(message.author.id), user_name))
        await client.db_conn.execute("INSERT INTO chat_history (server_id, role, content, user_id, user_name) VALUES (?, ?, ?, ?, ?)", (server_id, "assistant", final_reply, str(message.author.id), user_name))
        
        cursor = await client.db_conn.execute("SELECT COUNT(*) FROM chat_history WHERE server_id = ?", (server_id,))
        
        if (await cursor.fetchone())[0] >= MAX_HISTORY_LENGTH:
            eviction_count = MAX_HISTORY_LENGTH // 2  
            cursor = await client.db_conn.execute('''SELECT id, role, content, user_id, user_name FROM chat_history WHERE server_id = ? ORDER BY id ASC LIMIT ?''', (server_id, eviction_count))
            forgotten_rows = await cursor.fetchall()
            
            if forgotten_rows:
                forgotten_msgs = [{"role": r[1], "content": r[2], "user_id": r[3], "user_name": r[4]} for r in forgotten_rows]
                users_in_forgotten = {msg["user_id"]: msg["user_name"] for msg in forgotten_msgs if msg.get("user_id")}
                if users_in_forgotten:
                    task = asyncio.create_task(process_memories_concurrently(server_id, users_in_forgotten, forgotten_msgs))
                    client.background_tasks.add(task)
                    task.add_done_callback(client.background_tasks.discard)
                    
                ids_to_delete = [r[0] for r in forgotten_rows]
                placeholders = ','.join('?' * len(ids_to_delete))
                await client.db_conn.execute(f"DELETE FROM chat_history WHERE id IN ({placeholders})", ids_to_delete)
        
        # IMPORTANT: Commit happens INSIDE the lock so the next thread sees the updated count
        await client.db_conn.commit()
    # --- END OF LOCK BLOCK ---
    
    await send_chunked_message(message, final_reply)

# ==========================================
# 7. DISCORD EVENTS
# ==========================================

@client.event
async def on_ready():
    logging.info(f'✅ Logged in successfully as {client.user}')
    logging.info('🌐 Bot is fully online and ready!')

@client.event
async def on_message(message):
    # Check if the bot was mentioned directly
    is_mention = client.user in message.mentions
    is_reply_to_bot = False
    
    # Safely check the native cache WITHOUT making API calls
    if message.reference and message.reference.cached_message:
        if message.reference.cached_message.author.id == client.user.id:
            is_reply_to_bot = True

    # If the bot wasn't pinged, AND the message wasn't a cached reply to the bot, ignore it.
    if message.author.bot or not message.guild or not (is_mention or is_reply_to_bot):
        return

    server_id = str(message.guild.id)
    bot_mention = f'<@{client.user.id}>'
    bot_nickname_mention = f'<@!{client.user.id}>' 
    clean_message = message.content.replace(bot_mention, '').replace(bot_nickname_mention, '').strip()
    user_name = f"{message.author.display_name}_{str(message.author.id)[-4:]}"

    # Check for physical files/stickers
    media_tag = " [Media attached]" if message.attachments or message.stickers else ""
    
    # Check for web links in the text
    url_pattern = r'(https?://[^\s<>]+)'
    if re.search(url_pattern, clean_message):
        media_tag += " [Link attached]"
        
    if clean_message:
        truncated_msg = clean_message if len(clean_message) <= 50 else clean_message[:50] + "... [truncated]"
        log_content = f"{truncated_msg}{media_tag}"
    else:
        log_content = media_tag.strip() if media_tag else "[Empty Ping]"
        
    logging.info(f"{message.guild.name} | #{message.channel.name} | {message.author}: {log_content}")

    for mentioned_user in message.mentions:
        if mentioned_user.id != client.user.id:
            memory_formatted_name = f"{mentioned_user.display_name}_{str(mentioned_user.id)[-4:]}"
            clean_message = clean_message.replace(f"<@{mentioned_user.id}>", f"@{memory_formatted_name}").replace(f"<@!{mentioned_user.id}>", f"@{memory_formatted_name}")

    clean_message, image_attachments, valid_stickers, ephemeral_context = await extract_message_context(message, clean_message, user_name)
    
    if not clean_message.strip() and not image_attachments and not valid_stickers:
        await message.reply(f"Hello! I've been upgraded to use Slash Commands. Type `/help` to see what I can do!") 
        return

    api_user_content, db_user_content_obj = await build_user_payloads(clean_message, ephemeral_context, image_attachments, valid_stickers, user_name)
    messages_to_send = await build_ai_context(server_id, str(message.author.id), api_user_content)

    disable_search = bool(ephemeral_context and ("Extracted webpage content from" in ephemeral_context or "Extracted PDF Content" in ephemeral_context))
    has_media = bool(image_attachments or valid_stickers)

    start_time = datetime.now()
    final_reply = await generate_ai_response(messages_to_send, message, disable_search, has_media)

    if final_reply:
        duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"✨ AI Response generated in {duration:.2f}s | Server: {message.guild.name}")
        await save_and_send_response(message, server_id, user_name, db_user_content_obj, final_reply)

if TOKEN: 
    client.run(TOKEN)