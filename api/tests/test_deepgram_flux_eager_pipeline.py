"""Flux protocol events through Dograh's real call/agent pipeline assembly.

Only the Flux socket is stubbed. LLM and TTS use Pipecat's in-process fakes;
turn aggregation, the worker bus, speculation gates and tool runners are real.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

from api.schemas.llm_fallback import NoOutputCondition
from api.services.configuration.registry import DeepgramSTTConfiguration
from api.services.pipecat.agent_bridge import AgentBridgeProcessor
from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.fallback_llm import FallbackLLMProcessor, FallbackRoute
from api.services.pipecat.pipeline_builder import (
    build_agent_generation_pipeline,
    build_pipeline,
    create_agent_worker,
    create_pipeline_task,
)
from api.services.pipecat.run_pipeline import _create_non_realtime_user_turn_strategies
from api.services.pipecat.service_factory import create_stt_service
from api.services.pipecat.worker_runner import (
    create_worker_runner,
    wait_for_pipeline_worker_started,
)
from pipecat.tests import MockLLMService, MockTTSService


class Probe(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest_asyncio.fixture(params=[False, True], ids=["standalone", "fallback"])
async def eager_pipeline(request, monkeypatch):
    monkeypatch.setattr(DeepgramFluxSTTService, "_connect", AsyncMock())
    monkeypatch.setattr(DeepgramFluxSTTService, "_disconnect", AsyncMock())
    config = SimpleNamespace(
        stt=DeepgramSTTConfiguration(
            api_key="test-key",
            model="flux-general-en",
            enable_eager_end_of_turn=True,
            eager_eot_threshold=0.4,
        )
    )
    audio = AudioConfig(transport_in_sample_rate=16000, transport_out_sample_rate=16000)
    stt = create_stt_service(config, audio)
    primary = MockLLMService(chunk_delay=0, enable_direct_mode=request.param)
    llm = primary
    if request.param:
        llm = FallbackLLMProcessor(
            primary,
            routes=[
                FallbackRoute(
                    NoOutputCondition(after_ms=1200),
                    MockLLMService(chunk_delay=0, enable_direct_mode=True),
                )
            ],
        )
    spoken = []
    tts = MockTTSService(
        mock_audio_duration_ms=20,
        frame_delay=0,
        pause_frame_processing=False,
    )
    original_tts = tts.run_tts

    async def synthesize(text, context_id):
        spoken.append(text)
        async for frame in original_tts(text, context_id):
            yield frame

    monkeypatch.setattr(tts, "run_tts", synthesize)
    context = LLMContext()
    user, assistant = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=_create_non_realtime_user_turn_strategies(
                {},
                uses_external_turns=True,
                track_eager_user_turns=True,
            ),
            user_idle_timeout=0,
        ),
    )
    runner = create_worker_runner()
    monitor, output = Probe(), Probe()
    transport = SimpleNamespace(input=lambda: Probe(), output=lambda: output)
    bridge = AgentBridgeProcessor(
        bus=runner.bus,
        worker_name="eager-call",
        selected_visit=lambda: "eager-agent",
        allow_inference=lambda: True,
    )
    call_pipeline = build_pipeline(
        transport,
        stt,
        None,
        user,
        assistant,
        monitor,
        [bridge],
        None,
        None,
    )
    call = create_pipeline_task(
        call_pipeline,
        workflow_run_id=1,
        audio_config=audio,
        name="eager-call",
    )
    child = create_agent_worker(
        build_agent_generation_pipeline(llm, tts, Probe()),
        "eager-agent",
        audio,
        call_worker_name=call.name,
    )
    await runner.add_workers(call)
    running = asyncio.create_task(runner.run())
    try:
        await wait_for_pipeline_worker_started(call, run_task=running)
        await call.add_workers(child)
        await wait_for_pipeline_worker_started(child, run_task=running)
        await call.activate_worker(child.name)
        await wait_until(lambda: child.active)

        async def event(name, text=""):
            await stt._handle_message(
                {"type": "TurnInfo", "event": name, "transcript": text}
            )
            # Drain the turn-start interruption before replaying the next
            # provider event, as in Pipecat's eager strategy tests.
            await asyncio.sleep(0.05)

        yield SimpleNamespace(
            primary=primary,
            llm=llm,
            spoken=spoken,
            event=event,
            requests=lambda: [
                f for f in monitor.frames if isinstance(f, LLMContextFrame)
            ],
        )
    finally:
        await runner.cancel()
        await asyncio.wait_for(running, 5)


@pytest.mark.parametrize("withdrawal", ["TurnResumed", "different transcript"])
async def test_withdrawn_flux_guess_never_reaches_tts(eager_pipeline, withdrawal):
    h = eager_pipeline
    h.primary.set_mock_steps(
        [
            MockLLMService.create_text_chunks("Wrong guess."),
            MockLLMService.create_text_chunks("Correct answer."),
        ]
    )
    await h.event("StartOfTurn", "Book")
    await h.event("EagerEndOfTurn", "Book a flight")
    await wait_until(lambda: h.primary.get_current_step() == 1)
    await asyncio.sleep(0.05)
    assert h.spoken == []
    if withdrawal == "TurnResumed":
        await h.event("TurnResumed")
    await h.event("EndOfTurn", "Book a hotel instead")
    await wait_until(lambda: bool(h.spoken))
    assert h.spoken == ["Correct answer."]
    assert [f.speculation for f in h.requests()] == [True, False]


async def test_confirmed_flux_guess_releases_once(eager_pipeline):
    h = eager_pipeline
    h.primary.set_mock_steps([MockLLMService.create_text_chunks("Prepared answer.")])
    await h.event("StartOfTurn", "Book")
    await h.event("EagerEndOfTurn", "Book a flight")
    await wait_until(lambda: h.primary.get_current_step() == 1)
    await asyncio.sleep(0.05)
    assert h.spoken == []
    await h.event("EndOfTurn", "Book a flight")
    await wait_until(lambda: bool(h.spoken))
    assert h.spoken == ["Prepared answer."]
    assert [f.speculation for f in h.requests()] == [True]
    await asyncio.sleep(0.1)
    assert h.spoken == ["Prepared answer."]
    assert [f.speculation for f in h.requests()] == [True]


async def test_speculative_tools_wait_for_flux_commit(eager_pipeline):
    h = eager_pipeline
    tool_chunks = MockLLMService.create_function_call_chunks("book", {})
    h.primary.set_mock_steps(
        [
            tool_chunks,
            tool_chunks,
            MockLLMService.create_text_chunks("Booked."),
        ]
    )
    handler = AsyncMock()
    h.llm.register_function("book", handler)
    await h.event("StartOfTurn", "Book")
    await h.event("EagerEndOfTurn", "Book a flight")
    await wait_until(lambda: h.primary.get_current_step() == 1)
    await asyncio.sleep(0.05)
    handler.assert_not_awaited()
    assert h.spoken == []
    await h.event("EndOfTurn", "Book a flight")
    await wait_until(lambda: handler.await_count > 0)
    handler.assert_awaited_once()
    assert [f.speculation for f in h.requests()][:2] == [True, False]
