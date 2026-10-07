"""Check package compatibility with synthetic data and mocked network clients.

Run with: python scripts/check_dependencies.py
The bot is loaded in a temporary directory with dotenv and Discord login disabled.
"""

import asyncio
import base64
import builtins
import contextlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx2
from openai import APIStatusError, AsyncOpenAI
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Channel:
    def __init__(self):
        self.messages = []

    @contextlib.asynccontextmanager
    async def typing(self):
        yield

    async def send(self, text, **kwargs):
        self.messages.append(text)


async def check_bot():
    requests_seen = []
    primary_offline = False

    async def respond(request):
        assert request.extensions['timeout'] == {
            'connect': 2.0, 'read': 120.0, 'write': 120.0, 'pool': 120.0,
        }, 'The configured timeout is incompatible with the OpenAI SDK transport'
        assert request.url.path in ('/v1/models', '/v1/chat/completions'), 'Unexpected model or embedding request'
        requests_seen.append((request.url.host, request.url.path))
        if request.url.path.endswith('/models'):
            return httpx2.Response(200, json={
                'object': 'list',
                'data': [{'id': 'local-model', 'object': 'model', 'created': 0, 'owned_by': 'test'}],
            })
        if primary_offline and request.url.host == 'dependency-check.invalid':
            return httpx2.Response(503, json={'error': {'message': 'Synthetic outage'}})

        payload = json.loads(request.content)
        if payload.get('tools') and not any(m['role'] == 'tool' for m in payload['messages']):
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

    original_import = builtins.__import__

    def import_without_chroma(name, *args, **kwargs):
        assert name.split('.')[0] != 'chromadb', 'Chat must run without Chroma installed'
        return original_import(name, *args, **kwargs)

    environment = {
        'DISCORD_BOT_TOKEN': '',
        'LLM_BASE_URL': 'https://dependency-check.invalid/v1',
        'LLM_API_KEY': 'offline-test', 'LLM_MODEL_NAME': 'local-model',
        'VISION_ENABLED': 'true',
    }
    previous_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='discord-bot-dependencies-') as directory:
        try:
            os.chdir(directory)
            with (
                patch.dict(os.environ, environment, clear=True),
                patch('dotenv.load_dotenv', side_effect=AssertionError('Tests must not read .env')),
                patch('builtins.__import__', side_effect=import_without_chroma),
                patch('openai.AsyncOpenAI', side_effect=make_model_client) as model_client_factory,
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
                    model_client_factory.assert_called_once()
                    models = await bot.client.lm_client.models.list()
                    assert models.data[0].id == 'local-model'
                    print('PASS SDK import, request serialization, and timeout configuration')

                    names = {command.name for command in bot.tree.get_commands()}
                    assert names == {'help', 'status', 'role', 'clear', 'imagegen'}
                    bot.tree.sync.assert_awaited_once()
                    cursor = await bot.client.db_conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                    tables = {row[0] for row in await cursor.fetchall() if not row[0].startswith('sqlite_')}
                    assert tables == {'server_config', 'chat_history'}
                    assert not Path('chroma_storage').exists()
                    print('PASS retained Discord commands and conversation-only database initialization')

                    channel = Channel()
                    message = SimpleNamespace(author=SimpleNamespace(id=42), channel=channel, reply=channel.send)
                    search = AsyncMock(return_value='Synthetic search result. Source: https://example.com/')
                    with patch.dict(bot.AVAILABLE_TOOLS, web_search=search):
                        content, stored = await bot.build_user_payloads('Hello', '', [], [], 'Tester')
                        context = await bot.build_ai_context('channel:10', content)
                        answer = await bot.generate_ai_response(context, message, False)
                        assert answer == 'Synthetic answer.'
                        search.assert_awaited_once_with(query='synthetic test')
                        await bot.save_and_send_response(message, 'channel:10', stored, answer)
                        assert channel.messages == ['Synthetic answer.']
                        cursor = await bot.client.db_conn.execute('SELECT COUNT(*) FROM chat_history')
                        assert (await cursor.fetchone())[0] == 2
                    assert len(requests_seen) == 3
                    print('PASS tool calling, response parsing, chat persistence, and reply delivery')

                    async with bot.history_transaction():
                        await bot.client.db_conn.execute(
                            "INSERT INTO server_config VALUES ('channel:10', 'Persisted persona')",
                        )
                    await bot.client.close()
                    bot.client = bot.MyAIClient(intents=bot.intents)
                    bot.client.config = bot.Config({
                        **environment,
                        'FALLBACK_BASE_URL': 'https://cloud-check.invalid/v1',
                        'FALLBACK_API_KEY': 'offline-cloud-key', 'FALLBACK_MODEL_NAME': 'cloud-chat',
                    })
                    await bot.client.setup_hook()
                    assert model_client_factory.call_count == 2, 'Each startup must create only one chat client'
                    assert all(call.kwargs['base_url'] == environment['LLM_BASE_URL']
                               for call in model_client_factory.call_args_list)
                    assert all(call.kwargs['max_retries'] == 0 for call in model_client_factory.call_args_list)
                    context = await bot.build_ai_context('channel:10', 'Follow up')
                    assert 'Persisted persona' in context[0]['content']
                    assert context[1:] == [
                        {'role': 'user', 'content': stored},
                        {'role': 'assistant', 'content': answer},
                        {'role': 'user', 'content': 'Follow up'},
                    ]
                    assert len(requests_seen) == 3, 'Startup and context assembly must not invoke models'
                    assert not Path('chroma_storage').exists()
                    print('PASS history/persona restart persistence without Chroma or embedding calls')

                    primary_offline = True
                    try:
                        await bot.request_completion(messages=[{'role': 'user', 'content': 'Hello'}])
                    except APIStatusError as exc:
                        assert exc.status_code == 503
                    else:
                        raise AssertionError('The configured endpoint failure must propagate')
                    assert len(requests_seen) == 4, 'Failures must not retry or use another endpoint'
                    primary_offline = False
                    response = await bot.request_completion(messages=[{'role': 'user', 'content': 'Hello'}])
                    assert response.model == 'local-model'
                    assert response.choices[0].message.content == 'Synthetic answer.'
                    assert all(host == 'dependency-check.invalid' for host, _ in requests_seen)
                    print('PASS single-endpoint failures, no retries, and recovery despite obsolete fallback settings')

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
                    assert len(requests_seen) == 5
                finally:
                    await bot.client.close()
                    logging.shutdown()
        finally:
            os.chdir(previous_directory)


if __name__ == '__main__':
    asyncio.run(check_bot())
