"""Runtime behaviour of the STT fallback switcher, including Flux turn handoff."""

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    ProposedUserStartedSpeakingFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.utils.asyncio.task_manager import TaskManager

from api.services.pipecat.fallback_stt import FallbackSTTService
from api.services.pipecat.greeting import GreetingController
from api.services.pipecat.run_pipeline import _bind_stt_fallback_turns


class FakeSTT(FrameProcessor):
    def __init__(self, name):
        super().__init__(name=name)
        self.audio = []
        self.reconnects = 0
        self._reconnect_on_error = True
        self._reconnecting = False
        self._need_reconnect = False
        self._disconnecting = False
        self._reconnect_audio_buffer = []

    async def _reconnect(self):
        self.reconnects += 1
        self._reconnecting = True
        self._reconnect_audio_buffer.append("would-replay")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self.audio.append(frame)
        await self.push_frame(frame, direction)


def _audio(payload: bytes) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=payload, sample_rate=16000, num_channels=1)


async def _pipeline(primary, backup):
    stt = FallbackSTTService(primary, backup)
    pipeline = Pipeline([stt])
    manager = TaskManager()
    await pipeline.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=manager,
            pipeline_worker=SimpleNamespace(app_resources={}, worker_runner=None),
            enable_metrics=True,
            enable_usage_metrics=True,
        )
    )
    await pipeline.queue_frame(StartFrame())
    await asyncio.sleep(0.05)
    return stt, pipeline


@pytest.mark.asyncio
async def test_connection_error_switches_once_and_does_not_replay_audio():
    primary = FakeSTT("primary")
    backup = FakeSTT("backup")
    stt, pipeline = await _pipeline(primary, backup)

    assert primary._reconnect_on_error is False
    first = _audio(b"\x00\x00")
    await pipeline.queue_frame(first)
    await asyncio.sleep(0.05)
    assert primary.audio == [first]
    assert backup.audio == []

    await primary.push_error("connection closed by server")
    await asyncio.sleep(0.05)
    assert stt.strategy.active_service is backup
    assert stt.fallback_metrics == {"switched": 1}
    assert primary._reconnect_audio_buffer == []
    assert not primary.is_usable

    second = _audio(b"\x01\x00")
    await pipeline.queue_frame(second)
    await asyncio.sleep(0.05)
    assert second not in primary.audio
    assert backup.audio == [second]
    assert first not in backup.audio

    await backup.push_error("connection closed by server")
    await asyncio.sleep(0.05)
    assert stt.strategy.active_service is backup
    assert stt.fallback_metrics == {"switched": 1}
    await pipeline.cleanup()


@pytest.mark.asyncio
async def test_replaced_reconnect_does_not_replay_or_retry():
    primary = FakeSTT("primary")
    backup = FakeSTT("backup")
    stt, pipeline = await _pipeline(primary, backup)
    primary._reconnect_audio_buffer.append("already-buffered")

    await primary._reconnect()
    await asyncio.sleep(0.05)

    assert primary.reconnects == 0
    assert primary._reconnect_audio_buffer == []
    assert stt.strategy.active_service is backup
    await pipeline.cleanup()


@pytest.mark.asyncio
async def test_deepgram_503_connection_error_switches_before_retry(monkeypatch):
    import pipecat.services.deepgram.stt as deepgram_stt

    attempts = 0

    class ApiError(Exception):
        status_code = 503

        def __init__(self):
            super().__init__("503 Service Unavailable")

    class UnavailableEndpoint:
        def connect(self, **_kwargs):
            class Connection:
                async def __aenter__(self):
                    nonlocal attempts
                    attempts += 1
                    raise ApiError()

                async def __aexit__(self, *_args):
                    return False

            return Connection()

    monkeypatch.setattr(
        deepgram_stt, "exponential_backoff_time", lambda *_args, **_kwargs: 0.01
    )
    monkeypatch.setattr(deepgram_stt, "ApiError", ApiError)
    primary = DeepgramSTTService(api_key="disposable", sample_rate=16000)
    primary._client = SimpleNamespace(
        listen=SimpleNamespace(v1=SimpleNamespace(connect=UnavailableEndpoint().connect))
    )
    backup = FakeSTT("backup")
    stt, pipeline = await _pipeline(primary, backup)

    assert stt.strategy.active_service is backup
    assert stt.fallback_metrics == {"switched": 1}
    assert attempts == 1
    audio = _audio(b"\x02\x00")
    await pipeline.queue_frame(audio)
    await asyncio.sleep(0.05)
    assert backup.audio == [audio]
    await pipeline.cleanup()


@pytest.mark.asyncio
async def test_ordinary_error_does_not_switch():
    primary = FakeSTT("primary")
    backup = FakeSTT("backup")
    stt, pipeline = await _pipeline(primary, backup)

    await primary.push_error("one bad frame")
    await asyncio.sleep(0.05)

    assert stt.strategy.active_service is primary
    assert stt.fallback_metrics == {"switched": 0}
    assert isinstance(ErrorFrame(error="one bad frame"), ErrorFrame)
    await pipeline.cleanup()


def _controller(strategies):
    return UserTurnController(user_turn_strategies=strategies)


async def _speak_while_flux_still_reports_speech(controller):
    """Open a turn, then leave Flux's 'user is speaking' flag set.

    A finalized transcript plus a local VAD stop is enough for the Korean
    speech-timeout stop. It is not enough while Flux still reports speech and
    never sends its own end-of-turn, which is the state after Flux dies.
    """
    await controller.process_frame(
        TranscriptionFrame(
            text="hello", user_id="caller", timestamp="0", finalized=True
        )
    )
    await controller.process_frame(ProposedUserStartedSpeakingFrame())
    await controller.process_frame(VADUserStartedSpeakingFrame())
    await controller.process_frame(VADUserStoppedSpeakingFrame())
    await asyncio.sleep(0.8)


@pytest.mark.asyncio
async def test_flux_fallback_ends_turns_like_the_transcription_path():
    from api.services.pipecat.run_pipeline import (
        _create_non_realtime_user_turn_strategies,
    )

    flux = _create_non_realtime_user_turn_strategies({}, uses_external_turns=True)
    controller = _controller(flux)
    manager = TaskManager()
    await controller.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=manager,
            pipeline_worker=SimpleNamespace(app_resources={}, worker_runner=None),
            enable_metrics=True,
            enable_usage_metrics=True,
        )
    )
    stopped = []
    controller.add_event_handler(
        "on_user_turn_stopped", lambda *_args: stopped.append(1)
    )

    await _speak_while_flux_still_reports_speech(controller)
    assert stopped == []
    assert isinstance(
        controller.user_turn_strategies.stop[0], ExternalUserTurnStopStrategy
    )

    primary = FakeSTT("primary")
    backup = FakeSTT("backup")
    stt = FallbackSTTService(primary, backup)
    greeting = GreetingController.__new__(GreetingController)
    greeting._user = SimpleNamespace(user_turn_controller=controller)
    greeting._normal_strategies = flux
    greeting._greeting_strategy = None
    _bind_stt_fallback_turns(
        stt,
        greeting._user,
        greeting,
        {},
        uses_external_turns=True,
    )

    await stt.on_switched()
    assert isinstance(
        controller.user_turn_strategies.stop[0], SpeechTimeoutUserTurnStopStrategy
    )
    assert greeting._normal_strategies is not flux
    assert isinstance(
        greeting._normal_strategies.stop[0], SpeechTimeoutUserTurnStopStrategy
    )

    await _speak_while_flux_still_reports_speech(controller)
    assert stopped == [1]
    await controller.cleanup()
