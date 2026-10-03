"""Check package compatibility with synthetic data and mocked network clients.

Run with: python scripts/check_dependencies.py
The bot is loaded in a temporary directory with dotenv and Discord login disabled.
"""

import asyncio
import base64
import contextlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx2
from openai import AsyncOpenAI
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class Channel:
    def __init__(self):
        self.messages = []

    @contextlib.asynccontextmanager
    async def typing(self):
        yield

    async def send(self, text):
        self.messages.append(text)


async def check_bot():
    requests_seen = []

    async def respond(request):
        assert request.extensions['timeout'] == {
            'connect': 2.0, 'read': 120.0, 'write': 120.0, 'pool': 120.0,
        }, 'The configured timeout is incompatible with the OpenAI SDK transport'
        requests_seen.append(request.url.path)
        if request.url.path.endswith('/models'):
            return httpx2.Response(200, json={
                'object': 'list',
                'data': [{'id': 'local-model', 'object': 'model', 'created': 0, 'owned_by': 'test'}],
            })

        payload = json.loads(request.content)
        memory_request = 'strict, automated data-extraction system' in str(payload['messages'])
        if memory_request:
            message = {'role': 'assistant', 'content': '["Tester likes Python"]'}
            finish_reason = 'stop'
        elif payload.get('tools') and not any(m['role'] == 'tool' for m in payload['messages']):
            message = {
                'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'search_1', 'type': 'function', 'function': {
                    'name': 'web_search', 'arguments': '{"query":"synthetic test"}',
                }}],
            }
            finish_reason = 'tool_calls'
        else:
            message = {'role': 'assistant', 'content': 'Synthetic answer.'}
            finish_reason = 'stop'
        return httpx2.Response(200, json={
            'id': 'synthetic_completion', 'object': 'chat.completion', 'created': 0,
            'model': payload['model'],
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        })

    def make_model_client(**kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond),
        ))

    def embed(url, *, headers, json, timeout):
        assert url.endswith('/embeddings')
        assert timeout == (2.0, 15.0)
        data = [{'embedding': [1.0, 0.0, 0.0]} for _ in json['input']]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'data': data})

    environment = {
        'DISCORD_BOT_TOKEN': '', 'BOT_OWNER_ID': '0',
        'LLM_BASE_URL': 'https://dependency-check.invalid/v1',
        'LLM_API_KEY': 'offline-test', 'LLM_MODEL_NAME': 'local-model',
        'EMB_MODEL_NAME': 'local-embedding', 'VISION_ENABLED': 'true',
        'FALLBACK_BASE_URL': '', 'FALLBACK_API_KEY': '',
        'FALLBACK_MODEL_NAME': '', 'FALLBACK_EMB_API_KEY': '',
        'ANONYMIZED_TELEMETRY': 'False',
    }
    previous_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='discord-bot-dependencies-') as directory:
        try:
            os.chdir(directory)
            with (
                patch.dict(os.environ, environment, clear=True),
                patch('dotenv.load_dotenv', return_value=False),
                patch('openai.AsyncOpenAI', side_effect=make_model_client),
                patch('discord.Client.run', side_effect=AssertionError('Discord login is disabled')),
                patch('socket.socket.connect', side_effect=AssertionError('Network access is disabled')),
                patch('socket.socket.connect_ex', side_effect=AssertionError('Network access is disabled')),
            ):
                spec = importlib.util.spec_from_file_location('bot_dependency_check', ROOT / 'bot.py')
                bot = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bot)
                bot.client.config = bot.Config(environment)
                bot.tree.sync = AsyncMock(return_value=[])
                try:
                    await bot.client.setup_hook()
                    models = await bot.client.lm_client.models.list()
                    assert models.data[0].id == 'local-model'
                    print('PASS SDK import, request serialization, and timeout configuration')

                    names = {command.name for command in bot.tree.get_commands()}
                    assert {'help', 'status', 'role', 'clear', 'memory', 'remember', 'force-forget', 'admin_wipe_server'} <= names
                    # Verify the published command schema no longer offers individual edits/deletions.
                    command = bot.tree.get_command('memory').to_dict(bot.tree)
                    options = {option['name']: option for option in command['options']}
                    assert set(options) == {'action', 'target_user'}
                    assert {choice['value'] for choice in options['action']['choices']} == {'list', 'read', 'clear'}
                    print('PASS Discord command registration and database initialization')

                    channel = Channel()
                    message = SimpleNamespace(author=SimpleNamespace(id=42), channel=channel, reply=channel.send)
                    search = AsyncMock(return_value='Synthetic search result. Source: https://example.com/')
                    with patch('requests.post', side_effect=embed), patch.dict(bot.AVAILABLE_TOOLS, web_search=search):
                        content, stored = await bot.build_user_payloads('Hello', '', [], [], 'Tester')
                        context = await bot.build_ai_context('1', content)
                        answer = await bot.generate_ai_response(context, message, False)
                        assert answer == 'Synthetic answer.'
                        search.assert_awaited_once_with(query='synthetic test')
                        await bot.save_and_send_response(message, '1', 'Tester', stored, answer)
                        assert channel.messages == ['Synthetic answer.']
                        cursor = await bot.client.db_conn.execute('SELECT COUNT(*) FROM chat_history')
                        assert (await cursor.fetchone())[0] == 2
                        print('PASS tool calling, response parsing, chat persistence, and reply delivery')

                        await bot.update_user_memory('1', '42', 'Tester', [
                            {'role': 'user', 'content': 'I like Python', 'user_id': '42'},
                        ])
                        assert bot.client.memory_collection.count() == 1
                        recalled = await bot.build_ai_context('1', 'What do I like?')
                        assert 'Tester likes Python' in recalled[0]['content']
                        print('PASS embedding requests and Chroma memory write and retrieval')

                        # Use real temporary Chroma and SQLite to check explicit saves and full forget,
                        # including an indexing write whose SQLite acknowledgement fails.
                        bot.client.memory_worker.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await bot.client.memory_worker
                        bot.client.memory_worker = None
                        key = await bot.remember_fact('1', '42', 'Tester', 'I like hiking')
                        await bot.client.db_conn.execute(
                            "CREATE TEMP TRIGGER fail_index_ack BEFORE UPDATE OF indexed ON explicit_memories "
                            "BEGIN SELECT RAISE(ABORT, 'synthetic acknowledgement failure'); END",
                        )
                        await bot.sync_manual_memories()
                        cursor = await bot.client.db_conn.execute('SELECT indexed FROM explicit_memories')
                        assert (await cursor.fetchone())[0] == 0
                        assert 'hiking' in (await bot.list_memories('1', '42'))[key][0]
                        await bot.client.db_conn.execute('DROP TRIGGER fail_index_ack')
                        await bot.sync_manual_memories()
                        assert bot.client.memory_collection.count() == 2
                        assert 'hiking' in bot.client.memory_collection.get(ids=[key])['documents'][0]
                        await bot.forget_memories('1', '42')
                        assert bot.client.memory_collection.count() == 0
                        assert not await bot.list_memories('1', '42')
                        cursor = await bot.client.db_conn.execute('SELECT COUNT(*) FROM explicit_memories')
                        assert (await cursor.fetchone())[0] == 0
                        print('PASS explicit memory save, retry after failed acknowledgement, and scoped cleanup')

                    with Image.new('RGBA', (1200, 600), (20, 40, 80, 255)) as image:
                        buffer = io.BytesIO()
                        image.save(buffer, format='PNG')
                    encoded = bot.process_image_bytes(buffer.getvalue())
                    assert encoded
                    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                        assert image.format == 'JPEG' and max(image.size) <= bot.MAX_IMAGE_DIMENSION
                    with bot.pymupdf.open() as document:
                        document.new_page().insert_text((72, 72), 'Synthetic PDF text')
                        pdf = document.tobytes()
                    assert 'Synthetic PDF text' in bot.extract_pdf_text(pdf)
                    assert callable(bot.DDGS().text)
                    print('PASS Pillow, PyMuPDF, and search client interfaces')
                    assert len(requests_seen) == 4
                finally:
                    await bot.client.close()
                    logging.shutdown()
        finally:
            os.chdir(previous_directory)


if __name__ == '__main__':
    asyncio.run(check_bot())
