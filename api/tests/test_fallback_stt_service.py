"""Runtime behaviour of the STT fallback switcher, including Flux turn handoff."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.utils.asyncio.task_manager import TaskManager

from api.services.pipecat.fallback_stt import FallbackSTTService
from api.services.pipecat import run_pipeline as run_pipeline_module
from api.services.pipecat.greeting import GreetingController
from api.services.pipecat.run_pipeline import _bind_stt_fallback_turns
from api.services.pipecat.termination_funnel_processor import (
    TerminationFunnelProcessor,
)


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


async def _pipeline(primary, backup, *processors):
    stt = FallbackSTTService(primary, backup)
    pipeline = Pipeline([stt, *processors])
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


@pytest.mark.asyncio
async def test_assemblyai_error_message_then_close_fails_over_without_termination():
    primary = AssemblyAISTTService(api_key="disposable", sample_rate=16000)
    primary._connect = AsyncMock()
    backup = FakeSTT("backup")
    funnel = TerminationFunnelProcessor()
    terminations = []

    async def record_termination(reason, error):
        terminations.append((reason, error))

    funnel.set_termination_handler(record_termination)
    stt, pipeline = await _pipeline(primary, backup, funnel)

    with pytest.raises(ValueError, match="Unknown message type: Error") as parsed:
        primary._parse_message({"type": "Error", "error": "session rejected"})
    assert primary.is_usable
    await primary.push_frame(
        ErrorFrame(
            error=f"Unknown error occurred: {parsed.value}",
            processor=primary,
            exception=parsed.value,
        ),
        FrameDirection.UPSTREAM,
    )
    await asyncio.sleep(0.05)

    assert stt.strategy.active_service is backup
    assert stt.fallback_metrics == {"switched": 1}

    await primary.push_error(
        "AssemblyAI websocket closed with code 3006",
        exception=ConnectionError("websocket close code 3006"),
    )
    await asyncio.sleep(0.05)

    assert stt.strategy.active_service is backup
    assert stt.fallback_metrics == {"switched": 1}
    assert terminations == []
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
async def test_external_turn_failover_installs_local_turn_analyzer():
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
        controller.user_turn_strategies.stop[0], TurnAnalyzerUserTurnStopStrategy
    )
    assert greeting._normal_strategies is not flux
    assert isinstance(
        greeting._normal_strategies.stop[0], TurnAnalyzerUserTurnStopStrategy
    )

    await _speak_while_flux_still_reports_speech(controller)
    assert stopped == []
    await controller.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prefix", "continuation", "pause_probability"),
    [
        ("I want a reservation for Friday at", "seven for two people", 0.982),
        ("다음 주 금요일 오후 세 시에 두 명으로 예약하고", "싶습니다.", 0.926),
    ],
)
async def test_stt_fallback_keeps_a_turn_open_across_a_1_5_second_pause(
    monkeypatch, prefix, continuation, pause_probability
):
    probabilities = iter((pause_probability, 0.999))

    def predict_endpoint(_analyzer, _audio):
        probability = next(probabilities)
        return {
            "probability": probability,
            "prediction": int(probability > 0.5),
        }

    monkeypatch.setattr(
        run_pipeline_module.LocalSmartTurnAnalyzerV3,
        "_predict_endpoint",
        predict_endpoint,
    )
    original = run_pipeline_module._create_non_realtime_user_turn_strategies(
        {"turn_stop_strategy": "assemblyai"}, uses_external_turns=True
    )
    controller = _controller(original)
    await controller.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=SimpleNamespace(app_resources={}, worker_runner=None),
            enable_metrics=True,
            enable_usage_metrics=True,
        )
    )
    stopped = []
    loop = asyncio.get_running_loop()
    stopped_event = asyncio.Event()

    def record_turn_stop(*_args):
        stopped.append(loop.time())
        stopped_event.set()

    controller.add_event_handler("on_user_turn_stopped", record_turn_stop)

    greeting = GreetingController.__new__(GreetingController)
    greeting._user = SimpleNamespace(user_turn_controller=controller)
    greeting._normal_strategies = original
    greeting._greeting_strategy = None
    stt = FallbackSTTService(FakeSTT("primary"), FakeSTT("backup"))
    _bind_stt_fallback_turns(
        stt, greeting._user, greeting, {}, uses_external_turns=True
    )
    await stt.on_switched()

    await controller.process_frame(
        TranscriptionFrame(text=prefix, user_id="caller", timestamp="0", finalized=True)
    )
    await controller.process_frame(ProposedUserStartedSpeakingFrame())
    await controller.process_frame(VADUserStartedSpeakingFrame())
    await controller.process_frame(_audio(b"\x00\x00" * 320))
    await controller.process_frame(VADUserStoppedSpeakingFrame())
    await asyncio.sleep(1.5)
    assert stopped == []

    await controller.process_frame(VADUserStartedSpeakingFrame())
    await controller.process_frame(_audio(b"\x00\x00" * 320))
    await controller.process_frame(
        TranscriptionFrame(
            text=f"{prefix} {continuation}",
            user_id="caller",
            timestamp="1",
            finalized=True,
        )
    )
    final_speech_at = loop.time()
    await controller.process_frame(VADUserStoppedSpeakingFrame())
    await asyncio.wait_for(stopped_event.wait(), timeout=1.2)
    assert len(stopped) == 1
    assert stopped[0] - final_speech_at <= 1.2
    await controller.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn_stop_strategy", "smart_turn_stop_secs", "expected_stop_secs"),
    [
        ("assemblyai", 1.4, 2.0),
        ("transcription", 2.4, 2.4),
        ("turn_analyzer", 1.7, 2.0),
    ],
)
async def test_external_stt_failover_uses_smart_turn_pause_window(
    turn_stop_strategy, smart_turn_stop_secs, expected_stop_secs
):
    controller = SimpleNamespace(_user_turn_stop_timeout=None)
    user_aggregator = SimpleNamespace(user_turn_controller=controller)
    greeting = SimpleNamespace(use_transcription_turns=AsyncMock())
    stt = FallbackSTTService(FakeSTT("primary"), FakeSTT("backup"))

    _bind_stt_fallback_turns(
        stt,
        user_aggregator,
        greeting,
        {
            "turn_stop_strategy": turn_stop_strategy,
            "smart_turn_stop_secs": smart_turn_stop_secs,
        },
        uses_external_turns=True,
    )
    await stt.on_switched()

    strategies = greeting.use_transcription_turns.await_args.args[0]
    stop_strategy = strategies.stop[0]
    assert isinstance(stop_strategy, TurnAnalyzerUserTurnStopStrategy)
    assert stop_strategy._turn_analyzer.params.stop_secs == expected_stop_secs
