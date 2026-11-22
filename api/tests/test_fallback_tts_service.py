"""Runtime behaviour of the TTS voice-fallback switcher."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.tests.mock_tts_service import MockTTSService
from pipecat.utils.asyncio.task_manager import TaskManager

from api.services.pipecat.fallback_tts import FallbackTTSService


class Capture(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames = []
        self.errors = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            self.errors.append(frame)
        else:
            self.frames.append(frame)
        await self.push_frame(frame, direction)


def http_error(status):
    request = httpx.Request("GET", "https://example.com")
    return httpx.HTTPStatusError(
        str(status), request=request, response=httpx.Response(status, request=request)
    )


class RecordingPrimary(MockTTSService):
    def __init__(self, **kwargs):
        kwargs.setdefault("stop_frame_timeout_s", 0.2)
        super().__init__(frame_delay=0, **kwargs)
        self.fail_in_setup = False

    async def setup(self, setup):
        await super().setup(setup)
        if self.fail_in_setup:
            await self.push_error("handshake failed", exception=http_error(402))


class FailingPrimary(RecordingPrimary):
    def __init__(self, *, audio_seconds=0.0, **kwargs):
        kwargs.setdefault("pause_frame_processing", False)
        super().__init__(**kwargs)
        self.audio_seconds = audio_seconds

    async def run_tts(self, text, context_id):
        self.received_texts.append(text)
        await asyncio.sleep(0)
        if self.audio_seconds:
            yield TTSAudioRawFrame(
                b"\x00\x00" * int(self.audio_seconds * self.sample_rate),
                self.sample_rate,
                1,
                context_id=context_id,
            )
            await asyncio.sleep(0.1)
        await self.push_error("synthesis failed", fatal=False)
        yield None


class RecordingBackup(MockTTSService):
    """A backup that connects after its base setup, as websocket services do."""

    def __init__(self, **kwargs):
        super().__init__(stop_frame_timeout_s=0.2, frame_delay=0, **kwargs)
        self.slow_connect_s = 0.0
        self.connect_gate: asyncio.Event | None = None
        self.connect_error = False
        self.fail_auth_in_setup = False

    async def setup(self, setup):
        await super().setup(setup)
        if self.slow_connect_s:
            await asyncio.sleep(self.slow_connect_s)
        if self.connect_gate:
            await self.connect_gate.wait()
        if self.connect_error:
            raise RuntimeError("connect refused")
        if self.fail_auth_in_setup:
            await self.push_error("unauthorized", exception=http_error(401))
            await self.set_usable(False)


class StreamingPrimary(RecordingPrimary):
    """Like a websocket provider: run_tts only sends the request, the audio comes
    later through the audio context, and the context stays open until it is
    closed. The test closes it (remove_audio_context), which runs the same
    _end_audio_context path as the idle stop timeout, at a moment it picks."""

    def __init__(self, *, push_stop_frames=True, **kwargs):
        # The idle timer is out of play: the test closes the context itself.
        super().__init__(stop_frame_timeout_s=3600.0, **kwargs)
        self._push_stop_frames = push_stop_frames
        self.context_closed = asyncio.Event()
        self.arrived = []
        self.pushed = []

    async def queue_frame(self, frame, *args, **kwargs):
        self.arrived.append(frame)
        await super().queue_frame(frame, *args, **kwargs)

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        self.pushed.append(frame)

    async def run_tts(self, text, context_id):
        self.received_texts.append(text)
        yield None

    async def on_audio_context_completed(self, context_id):
        self.context_closed.set()


class TwoHalvesBackup(RecordingBackup):
    """Speaks one utterance in two halves, the test holding the gap between them."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.second_half = asyncio.Event()

    def _chunk(self, context_id):
        return TTSAudioRawFrame(
            b"\x00\x01" * 160, self.sample_rate, 1, context_id=context_id
        )

    async def run_tts(self, text, context_id):
        self.received_texts.append(text)
        yield self._chunk(context_id)
        await self.second_half.wait()
        yield self._chunk(context_id)


def make_harness(
    *,
    fail_in_setup=False,
    slow_backup_setup=0.0,
    backup_connect_gate=None,
    fail_backup_auth=False,
    backup_connect_error=False,
    primary=None,
    backup=None,
):
    if primary is None:
        primary = RecordingPrimary(name="PrimaryTTS")
    primary.fail_in_setup = fail_in_setup
    if backup is None:
        backup = RecordingBackup(name="BackupTTS")
        backup.slow_connect_s = slow_backup_setup
        backup.connect_gate = backup_connect_gate
        backup.fail_auth_in_setup = fail_backup_auth
        backup.connect_error = backup_connect_error
    tts = FallbackTTSService(primary, backup)
    upstream, downstream = Capture(), Capture()
    pipeline = Pipeline([upstream, tts, downstream])
    manager = TaskManager()
    processor_setup = FrameProcessorSetup(
        clock=SystemClock(),
        task_manager=manager,
        pipeline_worker=SimpleNamespace(app_resources={}, worker_runner=None),
        enable_metrics=True,
        enable_usage_metrics=True,
    )

    async def setup():
        await pipeline.setup(processor_setup)

    return SimpleNamespace(
        tts=tts,
        primary=primary,
        backup=backup,
        pipeline=pipeline,
        upstream=upstream,
        downstream=downstream,
        manager=manager,
        setup=setup,
    )


async def build_harness(**options):
    harness = make_harness(**options)
    await harness.setup()
    return harness


async def eventually(predicate, *, timeout=3.0, what="the condition"):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)


async def start(harness):
    """Queue the StartFrame and wait until it has crossed the switcher."""
    await harness.pipeline.queue_frame(StartFrame())
    await eventually(
        lambda: any(isinstance(f, StartFrame) for f in harness.downstream.frames),
        what="the StartFrame to reach downstream",
    )



@pytest.mark.asyncio
async def test_greeting_is_replayed_once_after_early_primary_failure():
    primary = FailingPrimary(name="PrimaryTTS")
    harness = await build_harness(primary=primary)
    await start(harness)
    await harness.pipeline.queue_frame(TTSSpeakFrame("AI and recording disclosure"))
    await eventually(
        lambda: harness.backup.received_texts == ["AI and recording disclosure"],
        what="the backup to speak the greeting once",
    )
    assert harness.tts.fallback_metrics["switched"] == 1
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_ordinary_llm_sentence_is_replayed_after_early_primary_failure():
    primary = FailingPrimary(name="PrimaryTTS")
    harness = await build_harness(primary=primary)
    await start(harness)
    await harness.pipeline.queue_frame(LLMFullResponseStartFrame())
    await harness.pipeline.queue_frame(LLMTextFrame("An ordinary sentence."))
    await harness.pipeline.queue_frame(LLMFullResponseEndFrame())
    await eventually(
        lambda: harness.backup.received_texts == ["An ordinary sentence."],
        what="the backup to speak the LLM sentence once",
    )
    assert harness.tts.fallback_metrics["switched"] == 1
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_primary_failure_after_three_seconds_does_not_replay():
    primary = FailingPrimary(name="PrimaryTTS", audio_seconds=3.1)
    harness = await build_harness(primary=primary)
    await start(harness)
    await harness.pipeline.queue_frame(TTSSpeakFrame("Already heard sentence."))
    await eventually(
        lambda: harness.tts.fallback_metrics["switched"] == 1,
        what="the provider switch",
    )
    assert harness.backup.received_texts == []
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_two_errors_in_one_tick_switch_once():
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    switched_events = []

    @harness.tts.strategy.event_handler("on_service_switched")
    async def on_switched(_strategy, _service):
        switched_events.append(True)

    await asyncio.gather(
        harness.primary.push_error("one", fatal=False),
        harness.primary.push_error("two", fatal=False),
    )
    await asyncio.sleep(0.2)
    assert harness.tts.fallback_metrics["switched"] == 1
    assert len(switched_events) == 1
    await harness.pipeline.cleanup()
