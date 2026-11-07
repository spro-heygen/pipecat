#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for HeyGenTTSService and HeyGenHttpTTSService."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
import websockets
from aiohttp import web
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.heygen.tts import (
    HeyGenHttpTTSService,
    HeyGenTTSService,
    language_to_heygen_language,
)
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory

STREAM_PATH = "/v3/models/audio/tts/stream"
DEFAULT_VOICE_ID = "83548de556df48ba8c09c42a57c51d85"
_ENVELOPE = (
    '{"error":{"code":"invalid_parameter","message":"Voice not found","request_id":"req-7"}}'
)


@asynccontextmanager
async def _http_tts(client, sample_rate: int = 24000, **kwargs):
    async with aiohttp.ClientSession() as session:
        yield HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=sample_rate,
            **kwargs,
        )


def _make_app(requests: list, audio: bytes = b"", status: int = 200, error: str = ""):
    async def handler(request):
        requests.append({"headers": dict(request.headers), "body": await request.json()})
        if status != 200:
            envelope = {
                "error": {"code": "rate_limit_exceeded", "message": error, "request_id": "req-9"}
            }
            return web.json_response(envelope, status=status)

        response = web.StreamResponse(
            status=200, headers={"Content-Type": "audio/pcm", "X-Request-Id": "req-123"}
        )
        await response.prepare(request)
        # Split mid-sample to check that emitted frames stay sample-aligned.
        await response.write(audio[:2047])
        await asyncio.sleep(0.01)
        await response.write(audio[2047:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post(STREAM_PATH, handler)
    return app


@pytest.mark.asyncio
async def test_heygen_tts_streams_pcm(aiohttp_client):
    """The service sends the documented request and emits sample-aligned PCM frames."""
    requests = []
    pcm_audio = b"\x00\x01\x02\x03" * 1024
    client = await aiohttp_client(_make_app(requests, pcm_audio))

    async with _http_tts(client) as tts:
        down_frames, _ = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hello from HeyGen.")]
        )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio_frames)
    assert all(len(f.audio) % 2 == 0 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == pcm_audio

    assert len(requests) == 1
    assert requests[0]["headers"]["x-api-key"] == "test-key"
    assert requests[0]["body"] == {
        "text": "Hello from HeyGen.",
        "voice_id": DEFAULT_VOICE_ID,
        "model": "heygen-voice-1-turbo",
        "output_format": {"encoding": "pcm", "sample_rate_hertz": 24000},
    }


@pytest.mark.asyncio
async def test_heygen_tts_settings_in_body(aiohttp_client):
    """Voice, model and language settings reach the request body."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 2048))

    async with _http_tts(
        client,
        sample_rate=16000,
        settings=HeyGenHttpTTSService.Settings(
            voice="my-voice", model="heygen-voice-1-turbo-free", language=Language.EN_US
        ),
    ) as tts:
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    body = requests[0]["body"]
    assert body["voice_id"] == "my-voice"
    assert body["model"] == "heygen-voice-1-turbo-free"
    assert body["language"] == "en"
    assert body["output_format"]["sample_rate_hertz"] == 16000


@pytest.mark.asyncio
async def test_heygen_tts_unsupported_sample_rate_is_resampled(aiohttp_client):
    """A pipeline rate HeyGen doesn't offer requests 24 kHz and resamples to the pipeline rate."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 24000))

    async with _http_tts(client, sample_rate=32000) as tts:
        down_frames, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert requests[0]["body"]["output_format"]["sample_rate_hertz"] == 24000
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 32000 for f in audio_frames)


@pytest.mark.asyncio
async def test_heygen_tts_http_error(aiohttp_client):
    """A non-200 response produces an ErrorFrame with HeyGen's code, message and request ID."""
    requests = []
    client = await aiohttp_client(_make_app(requests, status=429, error="Request limit reached."))

    async with _http_tts(client) as tts:
        _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "429" in errors[0].error
    assert "rate_limit_exceeded: Request limit reached." in errors[0].error
    assert "req-9" in errors[0].error


def test_language_mapping():
    """Regional variants resolve to HeyGen's base ISO 639-1 codes."""
    assert language_to_heygen_language(Language.EN) == "en"
    assert language_to_heygen_language(Language.EN_GB) == "en"
    assert language_to_heygen_language(Language.ZH_HK) == "zh"
    assert language_to_heygen_language(Language.PT_BR) == "pt"


# --- HeyGenTTSService (WebSocket) ---


def _alignment(text: str) -> dict:
    chars = list(text)
    return {
        "chars": chars,
        "charStartTimesMs": [i * 50 for i in range(len(chars))],
        "charDurationsMs": [50 for _ in chars],
    }


class _FakeHeyGen:
    """A fake ``multi-stream-input`` socket: audio for each text, isFinal on close_context.

    Parameters:
        refused_sockets: How many sockets, in order, close with ``close_code``
            at their first text instead of speaking.
        close_code: The close code of a refused socket.
        final_delay: How long each isFinal is held back, as real synthesis takes time.
        final_after_text: Whether a context ends itself after its first text, as
            HeyGen ends one left silent mid-turn.
        trim_alignment: Whether alignment leaves out trailing spaces.
    """

    def __init__(
        self,
        refused_sockets: int = 0,
        close_code: int = 1013,
        final_delay: float = 0.0,
        final_after_text: bool = False,
        trim_alignment: bool = False,
    ):
        self.refused_sockets = refused_sockets
        self.close_code = close_code
        self.final_delay = final_delay
        self.final_after_text = final_after_text
        self.trim_alignment = trim_alignment
        self.audio = b"\x00\x01" * 2400
        self.request = None
        self.sockets: list[list[dict]] = []
        self.peak_open = 0
        self.stray: list[dict] = []
        self._open_contexts: set[str] = set()
        self._accepting_contexts: set[str] = set()

    @property
    def messages(self) -> list[dict]:
        return [message for socket in self.sockets for message in socket]

    async def _final(self, ws, context_id: str):
        self._accepting_contexts.discard(context_id)
        await asyncio.sleep(self.final_delay)
        self._open_contexts.discard(context_id)
        await ws.send(json.dumps({"isFinal": True, "contextId": context_id}))

    async def _speak(self, ws, context_id: str, text: str):
        spoken = text.rstrip() if self.trim_alignment else text
        await ws.send(
            json.dumps(
                {
                    "audio": base64.b64encode(self.audio).decode(),
                    "alignment": _alignment(spoken),
                    "contextId": context_id,
                    "isFinal": None,
                }
            )
        )

    async def __call__(self, ws):
        self.request = ws.request
        refused = len(self.sockets) < self.refused_sockets
        messages: list[dict] = []
        self.sockets.append(messages)
        try:
            async for raw in ws:
                message = json.loads(raw)
                messages.append(message)
                context_id = message.get("context_id")
                text = message.get("text") or ""
                if text == " ":
                    self._open_contexts.add(context_id)
                    self._accepting_contexts.add(context_id)
                    self.peak_open = max(self.peak_open, len(self._open_contexts))
                elif context_id not in self._accepting_contexts and not message.get(
                    "close_context"
                ):
                    self.stray.append(message)
                if text.strip() and refused:
                    await ws.send(
                        json.dumps(
                            {
                                "message": "Free Turbo capacity is full.",
                                "error": {
                                    "code": "rate_limit_exceeded",
                                    "message": "Free Turbo capacity is full.",
                                    "request_id": "req-123",
                                },
                            }
                        )
                    )
                    await ws.close(self.close_code, "Free Turbo capacity is full")
                    return
                if text.strip():
                    await self._speak(ws, context_id, text)
                    if self.final_after_text:
                        await self._final(ws, context_id)
                if message.get("close_context"):
                    asyncio.create_task(self._final(ws, context_id))
        except websockets.ConnectionClosed:
            pass


@pytest.fixture
def no_backoff():
    with patch("pipecat.services.heygen.tts.exponential_backoff_time", return_value=0.01):
        yield


def _service(server, **kwargs) -> HeyGenTTSService:
    host, port = next(iter(server.sockets)).getsockname()[:2]
    kwargs.setdefault("sample_rate", 24000)
    return HeyGenTTSService(api_key="test-key", url=f"ws://{host}:{port}", **kwargs)


async def _connection_query(**kwargs) -> dict[str, list[str]]:
    heygen = _FakeHeyGen()
    async with serve(heygen, "127.0.0.1", 0) as server:
        await run_test(_service(server, **kwargs), frames_to_send=[])
    return parse_qs(urlparse(heygen.request.path).query)


def _spoken_contexts(heygen: _FakeHeyGen) -> list[str]:
    return [m["context_id"] for m in heygen.messages if (m.get("text") or "").strip()]


@pytest.mark.asyncio
async def test_heygen_websocket_streams_audio_and_words():
    """The service connects with the v3 parameters and turns alignment into word frames."""
    heygen = _FakeHeyGen()

    async with serve(heygen, "127.0.0.1", 0) as server:
        tts = _service(server, settings=HeyGenTTSService.Settings(language=Language.EN_US))
        down_frames, _ = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from HeyGen."), SleepFrame(sleep=0.5)],
        )

    assert heygen.request.headers.get("x-api-key") == "test-key"
    parsed = urlparse(heygen.request.path)
    assert parsed.path == "/v3/models/audio/tts/multi-stream-input"
    assert parse_qs(parsed.query) == {
        "voice_id": [DEFAULT_VOICE_ID],
        "model": ["heygen-voice-1-turbo"],
        "output_format": ["pcm_24000"],
        "auto_mode": ["true"],
        "language": ["en"],
    }

    sent = heygen.messages
    context_id = sent[0]["context_id"]
    assert sent == [
        {"text": " ", "context_id": context_id},
        {"text": "Hello from HeyGen. ", "context_id": context_id},
        {"flush": True, "context_id": context_id},
        {"close_context": True, "context_id": context_id},
    ]

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert b"".join(f.audio for f in audio_frames) == heygen.audio
    words = [f.text for f in down_frames if isinstance(f, TTSTextFrame)]
    assert words == ["Hello", "from", "HeyGen."]


@pytest.mark.asyncio
async def test_heygen_websocket_speaks_a_last_word_without_a_trailing_space():
    """A last word that no space ends still reaches the assistant when the context ends."""
    heygen = _FakeHeyGen(trim_alignment=True)

    async with serve(heygen, "127.0.0.1", 0) as server:
        down_frames, _ = await run_test(
            _service(server),
            frames_to_send=[TTSSpeakFrame(text="Hello there."), SleepFrame(sleep=0.5)],
        )

    assert [f.text for f in down_frames if isinstance(f, TTSTextFrame)] == ["Hello", "there."]


@pytest.mark.asyncio
async def test_heygen_websocket_refused_context_reports_code_and_request_id():
    """HeyGen's error message becomes a recoverable ErrorFrame and the next turn still speaks."""
    heygen = _FakeHeyGen(refused_sockets=1, close_code=1008)

    async with serve(heygen, "127.0.0.1", 0) as server:
        tts = _service(server, reconnect_backoff_min_wait=0.01, reconnect_backoff_max_wait=0.01)
        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[
                TTSSpeakFrame(text="Lost."),
                SleepFrame(sleep=0.5),
                TTSSpeakFrame(text="Spoken."),
                SleepFrame(sleep=0.5),
            ],
        )

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert "rate_limit_exceeded" in errors[0].error and "req-123" in errors[0].error
    assert errors[0].category is ErrorCategory.RATE_LIMIT
    assert tts.is_usable
    assert len(heygen.sockets) == 2
    assert any(isinstance(f, TTSAudioRawFrame) for f in down_frames)


@pytest.mark.parametrize("close_code", [1012, 1013])
@pytest.mark.asyncio
async def test_heygen_websocket_retries_a_turn_after_a_retryable_close(no_backoff, close_code):
    """A context refused before any audio is sent again on a new socket, once, and is spoken."""
    heygen = _FakeHeyGen(refused_sockets=2, close_code=close_code)

    async with serve(heygen, "127.0.0.1", 0) as server:
        tts = _service(
            server, settings=HeyGenTTSService.Settings(model="heygen-voice-1-turbo-free")
        )
        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello there."), SleepFrame(sleep=0.5)],
        )

    assert "model=heygen-voice-1-turbo-free" in heygen.request.path
    assert len(heygen.sockets) == 3
    retried = heygen.sockets[-1]
    context_id = retried[0]["context_id"]
    assert retried == [
        {"text": " ", "context_id": context_id},
        {"text": "Hello there. ", "context_id": context_id},
        {"flush": True, "context_id": context_id},
        {"close_context": True, "context_id": context_id},
    ]
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert b"".join(f.audio for f in audio_frames) == heygen.audio
    assert [f.text for f in down_frames if isinstance(f, TTSTextFrame)] == ["Hello", "there."]
    assert tts.is_usable
    assert all(
        f.category is ErrorCategory.RATE_LIMIT for f in up_frames if isinstance(f, ErrorFrame)
    )


@pytest.mark.asyncio
async def test_heygen_websocket_gives_up_on_a_turn_but_not_the_service(no_backoff):
    """Once the retries run out the turn is lost, and the next turn gets a new socket."""
    heygen = _FakeHeyGen(refused_sockets=4)

    async with serve(heygen, "127.0.0.1", 0) as server:
        tts = _service(server, reconnect_backoff_min_wait=0.01, reconnect_backoff_max_wait=0.01)
        down_frames, _ = await run_test(
            tts,
            frames_to_send=[
                TTSSpeakFrame(text="Lost."),
                SleepFrame(sleep=0.5),
                TTSSpeakFrame(text="Spoken."),
                SleepFrame(sleep=0.5),
            ],
        )

    assert tts.is_usable
    assert len(heygen.sockets) == 5  # the first send, three retries, the next turn
    assert [m["text"] for m in heygen.sockets[3] if (m.get("text") or "").strip()] == ["Lost. "]
    assert [f.text for f in down_frames if isinstance(f, TTSTextFrame)] == ["Spoken."]


@pytest.mark.asyncio
async def test_heygen_websocket_interruption_cancels_a_pending_retry():
    """A turn interrupted while waiting out a refusal is not sent again."""
    heygen = _FakeHeyGen(refused_sockets=1)

    with patch("pipecat.services.heygen.tts.exponential_backoff_time", return_value=0.3):
        async with serve(heygen, "127.0.0.1", 0) as server:
            await run_test(
                _service(server),
                frames_to_send=[
                    TTSSpeakFrame(text="Interrupted."),
                    SleepFrame(sleep=0.1),
                    InterruptionFrame(),
                    SleepFrame(sleep=0.5),
                ],
            )

    assert not [m for socket in heygen.sockets[1:] for m in socket if m.get("text")]


@pytest.mark.asyncio
async def test_heygen_websocket_reconnect_waits_as_long_as_a_refused_handshake_asks():
    """A handshake refused with Retry-After is retried once, after that wait."""
    heygen = _FakeHeyGen(refused_sockets=1, close_code=1008)
    handshakes = []

    def rate_limit_the_reconnect(connection, request):
        handshakes.append(asyncio.get_running_loop().time())
        if len(handshakes) == 2:
            response = connection.respond(429, _ENVELOPE)
            response.headers["Retry-After"] = "1"
            return response

    async with serve(heygen, "127.0.0.1", 0, process_request=rate_limit_the_reconnect) as server:
        tts = _service(server, reconnect_backoff_min_wait=0.01, reconnect_backoff_max_wait=0.01)
        down_frames, _ = await run_test(
            tts,
            frames_to_send=[
                TTSSpeakFrame(text="Lost."),
                SleepFrame(sleep=0.2),
                TTSSpeakFrame(text="Not connected."),
                SleepFrame(sleep=1.2),
                TTSSpeakFrame(text="Spoken."),
                SleepFrame(sleep=0.5),
            ],
        )

    assert tts.is_usable
    assert len(handshakes) == 3
    assert handshakes[2] - handshakes[1] >= 1.0
    assert [f.text for f in down_frames if isinstance(f, TTSTextFrame)] == ["Spoken."]


@pytest.mark.asyncio
async def test_heygen_websocket_never_reuses_a_context_after_its_final_message():
    """After HeyGen ends a context mid-turn, the turn's next sentence opens a new one."""
    heygen = _FakeHeyGen(final_after_text=True)

    async with serve(heygen, "127.0.0.1", 0) as server:
        await run_test(
            _service(server),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame("First sentence. "),
                LLMTextFrame("Next one."),
                SleepFrame(sleep=0.2),
                LLMTextFrame(" And the last."),
                SleepFrame(sleep=0.2),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
            ],
        )

    contexts = _spoken_contexts(heygen)
    assert len(contexts) == 3
    assert len(set(contexts)) == 3


@pytest.mark.asyncio
async def test_heygen_websocket_interruption_closes_context_and_keeps_socket():
    """On barge-in the service closes the HeyGen context and keeps the socket open."""
    heygen = _FakeHeyGen()
    heygen.final_delay = 5.0

    async with serve(heygen, "127.0.0.1", 0) as server:
        await run_test(
            _service(server, text_aggregation_mode=TextAggregationMode.TOKEN),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame("Still speaking"),
                SleepFrame(sleep=0.2),
                InterruptionFrame(),
                SleepFrame(sleep=0.2),
            ],
        )

    context_id = heygen.messages[0]["context_id"]
    assert {"close_context": True, "context_id": context_id} in heygen.messages
    assert len(heygen.sockets) == 1


@pytest.mark.asyncio
async def test_heygen_websocket_keepalive_never_names_a_closed_context():
    """Keepalives name only an open context; HeyGen opens a context for any other."""
    heygen = _FakeHeyGen(final_delay=0.3)

    with patch("pipecat.services.heygen.tts._KEEPALIVE_INTERVAL_S", 0.05):
        async with serve(heygen, "127.0.0.1", 0) as server:
            await run_test(
                _service(server),
                frames_to_send=[
                    SleepFrame(sleep=0.2),
                    LLMFullResponseStartFrame(),
                    LLMTextFrame("Hello there. "),
                    SleepFrame(sleep=0.2),
                    LLMFullResponseEndFrame(),
                    SleepFrame(sleep=0.5),
                ],
            )

    context_id = _spoken_contexts(heygen)[0]
    assert {"text": "", "context_id": context_id} in heygen.messages
    assert any(m.get("close_context") and m["context_id"] != context_id for m in heygen.messages)
    assert heygen.stray == []


@pytest.mark.asyncio
async def test_heygen_websocket_turn_without_text_sends_nothing():
    """A turn that produced no speech, such as a tool call alone, opens no HeyGen context."""
    heygen = _FakeHeyGen()

    async with serve(heygen, "127.0.0.1", 0) as server:
        await run_test(
            _service(server),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.2),
            ],
        )

    assert heygen.messages == []


@pytest.mark.asyncio
async def test_heygen_websocket_flush_names_only_an_open_context():
    """Flushing a context the service never opened would open it at HeyGen."""
    tts = HeyGenTTSService(api_key="test-key", sample_rate=24000)
    websocket = AsyncMock()
    tts._websocket = websocket

    await tts.flush_audio("never-opened")

    websocket.send.assert_not_called()


@pytest.mark.parametrize(
    "aggregation, auto_mode",
    [(TextAggregationMode.TOKEN, "false"), (TextAggregationMode.SENTENCE, "true")],
)
@pytest.mark.asyncio
async def test_heygen_websocket_auto_mode_follows_aggregation(aggregation, auto_mode):
    """Token streaming leaves generation to HeyGen's chunk schedule."""
    query = await _connection_query(text_aggregation_mode=aggregation)
    assert query["auto_mode"] == [auto_mode]


@pytest.mark.asyncio
async def test_heygen_websocket_unsupported_sample_rate_is_resampled():
    """At a rate HeyGen doesn't offer, the socket asks for 24 kHz and frames carry the pipeline rate."""
    heygen = _FakeHeyGen()

    async with serve(heygen, "127.0.0.1", 0) as server:
        down_frames, _ = await run_test(
            _service(server, sample_rate=11025),
            frames_to_send=[TTSSpeakFrame(text="Hello."), SleepFrame(sleep=0.5)],
        )

    assert parse_qs(urlparse(heygen.request.path).query)["output_format"] == ["pcm_24000"]
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 11025 for f in audio_frames)
    resampled = sum(len(f.audio) for f in audio_frames)
    assert resampled < len(heygen.audio) * 0.6  # 24 kHz -> 11.025 kHz is about 0.46x the bytes


@pytest.mark.asyncio
async def test_heygen_websocket_url_options():
    """Connection options and settings reach the URL only when given."""
    plain = await _connection_query()
    for name in (
        "enable_logging",
        "enable_ssml_parsing",
        "end_user_id",
        "apply_text_normalization",
        "expressiveness_boost",
    ):
        assert name not in plain

    query = await _connection_query(
        enable_logging=False,
        enable_ssml_parsing=True,
        end_user_id="user-42",
        settings=HeyGenTTSService.Settings(
            apply_text_normalization="off", expressiveness_boost=0.5
        ),
    )
    assert query["enable_logging"] == ["false"]
    assert query["enable_ssml_parsing"] == ["true"]
    assert query["end_user_id"] == ["user-42"]
    assert query["apply_text_normalization"] == ["off"]
    assert query["expressiveness_boost"] == ["0.5"]


@pytest.mark.asyncio
async def test_heygen_unsupported_language_is_not_sent(aiohttp_client):
    """A language HeyGen doesn't speak is left out, so HeyGen detects the language."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 2048))
    async with _http_tts(
        client, settings=HeyGenHttpTTSService.Settings(language=Language.NO)
    ) as http_tts:
        await run_test(http_tts, frames_to_send=[TTSSpeakFrame(text="Hei.")])
    assert "language" not in requests[0]["body"]

    query = await _connection_query(settings=HeyGenTTSService.Settings(language=Language.NO))
    assert "language" not in query


@pytest.mark.asyncio
async def test_heygen_websocket_handshake_refusal_reports_reason():
    """A refused handshake reaches the ErrorFrame with HeyGen's reason and leaves the service unusable."""

    def refuse(connection, request):
        return connection.respond(400, _ENVELOPE)

    async with serve(lambda ws: None, "127.0.0.1", 0, process_request=refuse) as server:
        tts = _service(server)
        _, up_frames = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hello."), SleepFrame(sleep=0.3)]
        )

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert "Voice not found" in errors[0].error
    assert "req-7" in errors[0].error
    assert errors[0].category is ErrorCategory.INVALID_REQUEST
    assert not tts.is_usable


@pytest.mark.asyncio
async def test_heygen_websocket_back_to_back_speech_stays_under_context_limit():
    """Seven queued sentences never hold more than HeyGen's 5 contexts open at once."""
    heygen = _FakeHeyGen(final_delay=0.2)

    async with serve(heygen, "127.0.0.1", 0) as server:
        await run_test(
            _service(server),
            frames_to_send=[TTSSpeakFrame(text=f"Sentence {i}.") for i in range(7)]
            + [SleepFrame(sleep=2.5)],
        )

    assert len(set(_spoken_contexts(heygen))) == 7
    assert heygen.peak_open <= 5
