"""Exercise Dograh's race through unchanged Gemini services and frame queues."""

import ast
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from google.genai import Client
from google.genai.types import (
    Candidate,
    Content,
    FunctionCall,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HttpOptions,
    Part,
)
from loguru import logger
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallsFromLLMInfoFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.asyncio.task_manager import TaskManager
from pydantic import TypeAdapter

from api.schemas.llm_fallback import ErrorCondition, NoOutputCondition
from api.schemas.tool import ToolMessages
from api.services.pipecat import fallback_llm
from api.services.pipecat.fallback_llm import FallbackLLMProcessor, FallbackRoute

PRIMARY = "gemini-3.5-flash"
BACKUP = "gemini-3.1-flash-lite"
DOWN = FrameDirection.DOWNSTREAM


def chunk(text="answer", *, parts=None, tokens=None):
    return GenerateContentResponse(
        candidates=[
            Candidate(content=Content(role="model", parts=parts or [Part(text=text)]))
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            candidates_token_count=tokens
        )
        if tokens is not None
        else None,
    )


class Stream:
    def __init__(self, *steps):
        self.steps = steps
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def __call__(self):
        self.started.set()
        try:
            for step in self.steps:
                if isinstance(step, asyncio.Event):
                    await step.wait()
                elif isinstance(step, Exception):
                    raise step
                else:
                    yield step
        finally:
            self.closed.set()


class Capture(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames = []
        self.started = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        if isinstance(frame, StartFrame):
            self.started.set()
        await self.push_frame(frame, direction)


@pytest_asyncio.fixture
async def harness(request):
    service_class = getattr(request, "param", GoogleLLMService)
    services = [
        service_class(
            api_key="test-key",
            settings=service_class.Settings(model=model),
            enable_direct_mode=True,
            stream_idle_timeout_secs=0.3,
        )
        for model in (PRIMARY, BACKUP)
    ]
    llm = FallbackLLMProcessor(
        services[0],
        routes=[
            FallbackRoute(NoOutputCondition.model_construct(after_ms=20), services[1]),
            FallbackRoute(ErrorCondition(), services[1]),
        ],
        first_output_timeout_secs=0.3,
    )
    upstream, downstream = Capture(), Capture()
    pipeline = Pipeline([upstream, llm, downstream])
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
    await asyncio.wait_for(downstream.started.wait(), 1)
    yield SimpleNamespace(
        llm=llm,
        pipeline=pipeline,
        upstream=upstream,
        downstream=downstream,
        manager=manager,
    )
    await pipeline.cleanup()
    assert not manager.current_tasks()


def install_streams(harness, monkeypatch, primary, backup):
    for service, stream in zip(
        (harness.llm.primary, harness.llm._services[1]), (primary, backup)
    ):
        monkeypatch.setattr(
            service, "_stream_content", AsyncMock(side_effect=lambda _, s=stream: s())
        )


@pytest.fixture
def fallback_logs():
    messages = []

    def capture(message):
        text = message.record["message"]
        if text.startswith("LLM fallback:"):
            messages.append(text)

    sink = logger.add(capture, level="INFO")
    try:
        yield messages
    finally:
        logger.remove(sink)


def assert_fallback_log(messages, *, winner, interrupted, outcomes):
    (message,) = messages
    prefix = f"LLM fallback: winner={winner}, interrupted={interrupted}, attempts="
    assert message.startswith(prefix)
    attempts = ast.literal_eval(message.removeprefix(prefix))
    assert [(a["index"], a["failed"], a["cancelled"]) for a in attempts] == [
        (index, failed, cancelled) for index, (failed, cancelled) in enumerate(outcomes)
    ]


async def generate(harness, *, context=None, speculation=False):
    done = asyncio.Event()

    async def completed(*_):
        done.set()

    await harness.llm.queue_frame(
        LLMContextFrame(context or LLMContext(), speculation=speculation),
        callback=completed,
    )
    await asyncio.wait_for(done.wait(), 1)
    return [
        frame.text
        for frame in harness.downstream.frames
        if isinstance(frame, LLMTextFrame)
    ]


@pytest.mark.asyncio
async def test_fast_primary_never_starts_backup(harness, monkeypatch):
    primary, backup = Stream(chunk("primary")), Stream(chunk("backup"))
    install_streams(harness, monkeypatch, primary, backup)
    assert await generate(harness) == ["primary"]
    assert not backup.started.is_set()
    assert primary.closed.is_set()
    assert harness.llm.fallback_metrics == {"started": 0, "won": 0}


@pytest.mark.asyncio
async def test_backup_wins_and_next_generation_restarts_primary(
    harness, monkeypatch, fallback_logs
):
    primary, backup = Stream(asyncio.Event()), Stream(chunk("backup"))
    install_streams(harness, monkeypatch, primary, backup)
    assert await generate(harness) == ["backup"]
    assert primary.closed.is_set() and backup.closed.is_set()
    assert harness.llm.fallback_metrics == {"started": 1, "won": 1}
    assert_fallback_log(
        fallback_logs,
        winner=1,
        interrupted=False,
        outcomes=[(False, True), (False, False)],
    )
    primary.steps = (chunk("primary"),)
    assert await generate(harness) == ["backup", "primary"]
    frames = harness.downstream.frames
    assert sum(isinstance(f, LLMFullResponseStartFrame) for f in frames) == 2
    assert sum(isinstance(f, LLMFullResponseEndFrame) for f in frames) == 2


@pytest.mark.asyncio
async def test_primary_wins_after_backup_starts(harness, monkeypatch, fallback_logs):
    release = asyncio.Event()
    primary, backup = Stream(release, chunk("primary")), Stream(asyncio.Event())
    install_streams(harness, monkeypatch, primary, backup)
    task = asyncio.create_task(generate(harness))
    await asyncio.wait_for(backup.started.wait(), 1)
    release.set()
    assert await task == ["primary"]
    assert backup.closed.is_set()
    assert harness.llm.fallback_metrics == {"started": 1, "won": 0}
    assert_fallback_log(
        fallback_logs,
        winner=0,
        interrupted=False,
        outcomes=[(False, False), (False, True)],
    )


@pytest.mark.asyncio
async def test_error_starts_backup_immediately(harness, monkeypatch):
    harness.llm._fallback_after_secs = 10
    install_streams(
        harness,
        monkeypatch,
        Stream(ConnectionError("primary")),
        Stream(chunk("backup")),
    )
    assert await generate(harness) == ["backup"]
    assert not any(isinstance(f, ErrorFrame) for f in harness.upstream.frames)


@pytest.mark.asyncio
async def test_backup_error_does_not_abandon_primary(harness, monkeypatch):
    release = asyncio.Event()
    primary, backup = (
        Stream(release, chunk("primary")),
        Stream(ConnectionError("backup")),
    )
    install_streams(harness, monkeypatch, primary, backup)
    task = asyncio.create_task(generate(harness))
    await asyncio.wait_for(backup.closed.wait(), 1)
    release.set()
    assert await task == ["primary"]
    assert not any(isinstance(f, ErrorFrame) for f in harness.upstream.frames)


@pytest.mark.asyncio
async def test_both_fail_emit_one_error_and_one_response(
    harness, monkeypatch, fallback_logs
):
    install_streams(
        harness,
        monkeypatch,
        Stream(TimeoutError("primary")),
        Stream(ConnectionError("backup")),
    )
    assert await generate(harness) == []
    assert sum(isinstance(f, ErrorFrame) for f in harness.upstream.frames) == 1
    assert_fallback_log(
        fallback_logs,
        winner=None,
        interrupted=False,
        outcomes=[(True, False), (True, False)],
    )
    assert (
        sum(isinstance(f, LLMFullResponseStartFrame) for f in harness.downstream.frames)
        == 1
    )
    assert (
        sum(isinstance(f, LLMFullResponseEndFrame) for f in harness.downstream.frames)
        == 1
    )


@pytest.mark.asyncio
async def test_winner_error_never_switches_after_text(harness, monkeypatch):
    backup = Stream(chunk("backup"))
    install_streams(
        harness,
        monkeypatch,
        Stream(chunk("partial"), ConnectionError("broken")),
        backup,
    )
    assert await generate(harness) == ["partial"]
    assert not backup.started.is_set()
    assert sum(isinstance(f, ErrorFrame) for f in harness.upstream.frames) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("start_backup", [False, True])
async def test_interruption_closes_requests_and_next_turn_works(
    harness, monkeypatch, start_backup
):
    primary, backup = Stream(asyncio.Event()), Stream(asyncio.Event())
    install_streams(harness, monkeypatch, primary, backup)
    await harness.llm.queue_frame(LLMContextFrame(LLMContext()))
    await asyncio.wait_for((backup if start_backup else primary).started.wait(), 1)
    done = asyncio.Event()

    async def interrupted(*_):
        done.set()

    await harness.llm.queue_frame(InterruptionFrame(), callback=interrupted)
    await asyncio.wait_for(done.wait(), 1)
    assert primary.closed.is_set()
    assert backup.closed.is_set() == start_backup
    assert not any(isinstance(f, ErrorFrame) for f in harness.upstream.frames)
    primary.steps = (chunk("next"),)
    assert await generate(harness) == ["next"]


@pytest.mark.asyncio
async def test_empty_stream_fails_over(harness, monkeypatch, fallback_logs):
    install_streams(
        harness, monkeypatch, Stream(GenerateContentResponse()), Stream(chunk("backup"))
    )
    assert await generate(harness) == ["backup"]
    assert_fallback_log(
        fallback_logs,
        winner=1,
        interrupted=False,
        outcomes=[(True, False), (False, False)],
    )


@pytest.mark.asyncio
async def test_setup_deadline_covers_both_requests(harness, monkeypatch, fallback_logs):
    harness.llm._first_output_timeout_secs = 0.04

    async def stuck(_):
        await asyncio.Event().wait()

    for service in (harness.llm.primary, harness.llm._services[1]):
        monkeypatch.setattr(service, "_stream_content", stuck)
    assert await generate(harness) == []
    assert sum(isinstance(f, ErrorFrame) for f in harness.upstream.frames) == 1
    assert_fallback_log(
        fallback_logs,
        winner=None,
        interrupted=False,
        outcomes=[(True, False), (True, False)],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_errors", [False, True])
async def test_interrupted_race_logs_cancellation_without_hiding_errors(
    harness, monkeypatch, fallback_logs, primary_errors
):
    primary = Stream(
        ConnectionError("429 RESOURCE_EXHAUSTED") if primary_errors else asyncio.Event()
    )
    backup = Stream(asyncio.Event())
    install_streams(harness, monkeypatch, primary, backup)
    await harness.llm.queue_frame(LLMContextFrame(LLMContext()))
    await asyncio.wait_for(backup.started.wait(), 1)
    await control(harness, InterruptionFrame())

    assert primary.closed.is_set() and backup.closed.is_set()
    assert not any(isinstance(f, ErrorFrame) for f in harness.upstream.frames)
    assert harness.llm.fallback_metrics == {"started": 1, "won": 0}
    assert_fallback_log(
        fallback_logs,
        winner=None,
        interrupted=True,
        outcomes=[(primary_errors, not primary_errors), (False, True)],
    )


@pytest.mark.asyncio
async def test_interrupted_streaming_winner_logs_cancellation(
    harness, monkeypatch, fallback_logs
):
    streaming = asyncio.Event()

    async def backup_stream():
        yield chunk("partial")
        streaming.set()
        await asyncio.Event().wait()

    primary = Stream(asyncio.Event())
    install_streams(harness, monkeypatch, primary, backup_stream)
    await harness.llm.queue_frame(LLMContextFrame(LLMContext()))
    await asyncio.wait_for(streaming.wait(), 1)
    await control(harness, InterruptionFrame())

    assert not any(isinstance(f, ErrorFrame) for f in harness.upstream.frames)
    assert harness.llm.fallback_metrics == {"started": 1, "won": 1}
    assert_fallback_log(
        fallback_logs,
        winner=1,
        interrupted=True,
        outcomes=[(False, True), (False, True)],
    )


@pytest.mark.asyncio
async def test_only_winner_tools_signatures_and_usage_escape(harness, monkeypatch):
    release = asyncio.Event()
    tool_parts = [
        Part(
            function_call=FunctionCall(name="book", args={}, id="booking"),
            thought_signature=b"signature",
        )
    ]
    primary = Stream(release, chunk(parts=tool_parts, tokens=10))
    backup = Stream(release, chunk(parts=tool_parts, tokens=20))
    install_streams(harness, monkeypatch, primary, backup)
    handler = AsyncMock()
    prepared = AsyncMock()
    harness.llm.register_function("book", handler)
    harness.llm.add_event_handler("on_function_calls_prepared", prepared)
    task = asyncio.create_task(generate(harness))
    await asyncio.wait_for(backup.started.wait(), 1)
    release.set()
    await task
    await asyncio.sleep(0)
    handler.assert_awaited_once()
    prepared.assert_awaited_once()
    for kind in (
        FunctionCallsFromLLMInfoFrame,
        LLMMessagesAppendFrame,
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    ):
        assert sum(isinstance(f, kind) for f in harness.downstream.frames) == 1
    usage = [
        data
        for f in harness.downstream.frames
        if isinstance(f, MetricsFrame)
        for data in f.data
        if isinstance(data, LLMUsageMetricsData)
    ]
    assert len(usage) == 1
    assert (usage[0].model, usage[0].value.completion_tokens) in [
        (PRIMARY, 10),
        (BACKUP, 20),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, "Sure."])
async def test_tool_hook_runs_after_the_response_frames_were_pushed(
    harness, monkeypatch, text
):
    """A tool hook sees the response already delivered, as with a direct service."""
    tool = Part(function_call=FunctionCall(name="book", args={}, id="booking"))
    parts = [Part(text=text), tool] if text else [tool]
    install_streams(
        harness, monkeypatch, Stream(chunk(parts=parts)), Stream(asyncio.Event())
    )
    pushed, at_hook = [], []
    harness.llm.add_event_handler(
        "on_before_push_frame", lambda _, frame: pushed.append(type(frame))
    )

    async def prepared(*_):
        at_hook.extend(pushed)

    harness.llm.register_function("book", AsyncMock())
    harness.llm.add_event_handler("on_function_calls_prepared", prepared)
    await generate(harness)
    assert LLMFullResponseStartFrame in at_hook
    assert (LLMTextFrame in at_hook) == bool(text)


@pytest.mark.asyncio
async def test_interruption_while_response_frames_flush_drops_tool_batch(
    harness, monkeypatch
):
    tool = Part(function_call=FunctionCall(name="book", args={}, id="booking"))
    install_streams(
        harness, monkeypatch, Stream(chunk(parts=[Part(text="Sure."), tool])), Stream()
    )
    pushed_start, release_start = asyncio.Event(), asyncio.Event()
    push_frame = LLMService.push_frame

    async def blocked_push_frame(self, frame, direction):
        if self is harness.llm and isinstance(frame, LLMFullResponseStartFrame):
            pushed_start.set()
            await release_start.wait()
        await push_frame(self, frame, direction)

    monkeypatch.setattr(LLMService, "push_frame", blocked_push_frame)
    handler, prepared = AsyncMock(), AsyncMock()
    harness.llm.register_function("book", handler)
    harness.llm.add_event_handler("on_function_calls_prepared", prepared)

    generation = asyncio.create_task(
        harness.llm.queue_frame(LLMContextFrame(LLMContext()))
    )
    await asyncio.wait_for(pushed_start.wait(), 1)
    race = harness.llm._active_race
    assert race is not None
    interruption = asyncio.create_task(control(harness, InterruptionFrame()))
    try:
        async with asyncio.timeout(1):
            while not race.interrupted:
                await asyncio.sleep(0)
    finally:
        release_start.set()
    await asyncio.wait_for(interruption, 1)
    await generation

    prepared.assert_not_awaited()
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_instructions_reach_both_and_inference_uses_primary(
    harness, monkeypatch
):
    await harness.llm._update_settings(
        LLMSettings(system_instruction="new instructions")
    )
    assert harness.llm.primary._settings.system_instruction == "new instructions"
    assert harness.llm._services[1]._settings.system_instruction == "new instructions"
    assert harness.llm._services[1]._settings.model == BACKUP
    infer = AsyncMock(return_value="one shot")
    monkeypatch.setattr(harness.llm.primary, "run_inference", infer)
    assert await harness.llm.run_inference(LLMContext()) == "one shot"


@pytest.mark.asyncio
async def test_backup_ttfb_includes_delay(harness, monkeypatch):
    install_streams(
        harness, monkeypatch, Stream(asyncio.Event()), Stream(chunk("backup"))
    )
    await generate(harness)
    timings = [
        data
        for f in harness.downstream.frames
        if isinstance(f, MetricsFrame)
        for data in f.data
        if isinstance(data, TTFBMetricsData)
    ]
    assert len(timings) == 1
    assert timings[0].value >= harness.llm._fallback_after_secs


@pytest.mark.asyncio
@pytest.mark.parametrize("simultaneous_output", [False, True])
async def test_real_sdk_closes_both_http_responses(harness, simultaneous_output):
    both_started = asyncio.Event()
    bodies = []

    class ResponseBody(httpx.AsyncByteStream):
        def __init__(self, primary):
            self.primary = primary
            self.closed = asyncio.Event()

        async def __aiter__(self):
            if self.primary:
                await (both_started if simultaneous_output else asyncio.Event()).wait()
            yield b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"answer"}]}}]}\n\n'
            # Keep a losing response open after its first chunk, too.
            if self.primary:
                await asyncio.Event().wait()

        async def aclose(self):
            self.closed.set()

    async def handle(request):
        body = ResponseBody(PRIMARY in request.url.path)
        bodies.append(body)
        if len(bodies) == 2:
            both_started.set()
        return httpx.Response(
            200, stream=body, headers={"Content-Type": "text/event-stream"}
        )

    clients = []
    try:
        for service in (harness.llm.primary, harness.llm._services[1]):
            await service._close_client()
            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            clients.append(client)
            service._client = Client(
                api_key="test", http_options=HttpOptions(httpx_async_client=client)
            )
        assert await generate(
            harness, context=LLMContext([{"role": "user", "content": "Hi"}])
        ) == ["answer"]
        assert len(bodies) == 2
        # Check before closing the clients: client teardown would mask leaks.
        async with asyncio.timeout(0.5):
            for body in bodies:
                await body.closed.wait()
    finally:
        for client in clients:
            await client.aclose()


@pytest.mark.asyncio
async def test_slow_loser_cleanup_does_not_delay_output(harness, monkeypatch):
    cleanup_started, release_cleanup, output = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def primary():
        try:
            await asyncio.Event().wait()
            yield chunk("unused")
        finally:
            cleanup_started.set()
            await release_cleanup.wait()

    monkeypatch.setattr(
        harness.llm.primary,
        "_stream_content",
        AsyncMock(side_effect=lambda _: primary()),
    )
    backup = Stream(chunk("backup"))
    monkeypatch.setattr(
        harness.llm._services[1],
        "_stream_content",
        AsyncMock(side_effect=lambda _: backup()),
    )

    async def on_frame(_, frame):
        if isinstance(frame, LLMTextFrame):
            output.set()

    harness.downstream.add_event_handler("on_after_process_frame", on_frame)
    task = asyncio.create_task(generate(harness))
    try:
        await asyncio.wait_for(output.wait(), 0.5)
        await asyncio.wait_for(cleanup_started.wait(), 0.5)
        assert not task.done()
    finally:
        release_cleanup.set()
    assert await task == ["backup"]


@pytest.mark.asyncio
async def test_metadata_trickle_cannot_extend_first_output_deadline(
    harness, monkeypatch
):
    harness.llm._first_output_timeout_secs = 0.06

    async def metadata():
        while True:
            yield GenerateContentResponse()
            await asyncio.sleep(0.005)

    for service in (harness.llm.primary, harness.llm._services[1]):
        monkeypatch.setattr(
            service, "_stream_content", AsyncMock(side_effect=lambda _: metadata())
        )
    assert await generate(harness) == []
    assert sum(isinstance(f, ErrorFrame) for f in harness.upstream.frames) == 1


@pytest.mark.asyncio
async def test_disabled_public_metrics_still_arbitrate_tools(harness, monkeypatch):
    harness.llm._setup.enable_metrics = False
    harness.llm._setup.enable_usage_metrics = False
    handler = AsyncMock()
    harness.llm.register_function("book", handler)
    install_streams(
        harness,
        monkeypatch,
        Stream(asyncio.Event()),
        Stream(chunk(parts=[Part(function_call=FunctionCall(name="book", args={}))])),
    )
    await generate(harness)
    await asyncio.sleep(0)
    handler.assert_awaited_once()
    assert harness.llm.fallback_metrics == {"started": 1, "won": 1}
    assert not any(isinstance(f, MetricsFrame) for f in harness.downstream.frames)


@pytest.mark.asyncio
async def test_worker_flush_waits_for_race_and_upstream_continuation(monkeypatch):
    from pipecat.frames.frames import EndFrame
    from pipecat.pipeline.worker import PipelineWorker

    from api.services.pipecat.worker_runner import (
        run_pipeline_worker,
        wait_for_pipeline_worker_started,
    )

    services = [
        GoogleLLMService(
            api_key="test",
            enable_direct_mode=True,
            settings=GoogleLLMService.Settings(model=m),
        )
        for m in (PRIMARY, BACKUP)
    ]
    llm = FallbackLLMProcessor(
        services[0],
        routes=[
            FallbackRoute(NoOutputCondition.model_construct(after_ms=20), services[1]),
            FallbackRoute(ErrorCondition(), services[1]),
        ],
    )
    release = asyncio.Event()
    primary, backup = Stream(asyncio.Event()), Stream(release, chunk("backup"))
    install_streams(SimpleNamespace(llm=llm), monkeypatch, primary, backup)

    class ContinueOnce(Capture):
        def __init__(self):
            super().__init__()
            self.continued = False

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMFullResponseEndFrame) and not self.continued:
                self.continued = True
                primary.steps = (chunk("continuation"),)
                await self.push_frame(
                    LLMContextFrame(LLMContext()), FrameDirection.UPSTREAM
                )

    sink = ContinueOnce()
    worker = PipelineWorker(Pipeline([llm, sink]), enable_rtvi=False)
    running = asyncio.create_task(run_pipeline_worker(worker))
    try:
        await wait_for_pipeline_worker_started(worker, run_task=running)
        await worker.queue_frame(LLMContextFrame(LLMContext()))
        await asyncio.wait_for(backup.started.wait(), 1)
        flush = asyncio.create_task(worker.flush_pipeline(timeout=0.5))
        await asyncio.sleep(0.02)
        assert not flush.done()
        release.set()
        assert await asyncio.wait_for(flush, 2)
        assert [f.text for f in sink.frames if isinstance(f, LLMTextFrame)] == [
            "backup",
            "continuation",
        ]
        assert await worker.flush_pipeline(timeout=0.5)
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(running, 2)
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)


async def control(harness, frame):
    done = asyncio.Event()

    async def completed(*_):
        done.set()

    await harness.llm.queue_frame(frame, callback=completed)
    await asyncio.wait_for(done.wait(), 1)


@pytest.mark.asyncio
async def test_interruption_during_loser_cleanup_waits_before_reusing_services(
    harness, monkeypatch, fallback_logs
):
    closing, release = asyncio.Event(), asyncio.Event()

    async def primary():
        try:
            await asyncio.Event().wait()
            yield chunk()
        finally:
            closing.set()
            await release.wait()

    monkeypatch.setattr(
        harness.llm.primary,
        "_stream_content",
        AsyncMock(side_effect=lambda _: primary()),
    )
    monkeypatch.setattr(
        harness.llm._services[1],
        "_stream_content",
        AsyncMock(side_effect=lambda _: Stream(chunk("backup"))()),
    )
    await harness.llm.queue_frame(LLMContextFrame(LLMContext()))
    await asyncio.wait_for(closing.wait(), 1)
    interrupt = asyncio.create_task(control(harness, InterruptionFrame()))
    try:
        await asyncio.sleep(0.01)
        assert not interrupt.done()
    finally:
        release.set()
    await interrupt
    assert_fallback_log(
        fallback_logs,
        winner=1,
        interrupted=True,
        outcomes=[(False, True), (False, False)],
    )
    install_streams(
        harness, monkeypatch, Stream(chunk("next")), Stream(chunk("unused"))
    )
    assert (await generate(harness))[-1] == "next"


@pytest.mark.asyncio
async def test_new_context_retires_backup_deferred_transition(harness, monkeypatch):
    from pipecat.frames.frames import BotStoppedSpeakingFrame

    transition = AsyncMock()
    harness.llm.register_function("move", transition, is_node_transition=True)
    primary = Stream(asyncio.Event())
    backup = Stream(
        chunk(
            parts=[
                Part(text="Moving you"),
                Part(function_call=FunctionCall(name="move", args={})),
            ]
        )
    )
    install_streams(harness, monkeypatch, primary, backup)
    assert await generate(harness) == ["Moving you"]
    assert harness.llm._services[1]._pending_node_transition_function_calls
    primary.steps = (chunk("new turn"),)
    await generate(harness)
    await control(harness, BotStoppedSpeakingFrame())
    transition.assert_not_awaited()
    assert not harness.llm._services[1]._pending_node_transition_function_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_output", "first_byte", "reported_ttfb", "ttfb", "thinking_time"),
    [
        # Speculation starts at 100.0 and is committed at 100.5, so the TTFAT
        # frame at 101.0 measures 0.5 s from the commit. first_output is
        # relative to the speculation start.
        pytest.param(0.6, None, 10.0, 0.1, 0.4, id="output-after-commit"),
        pytest.param(0.2, None, 10.0, 0.0, 0.5, id="output-before-commit"),
        pytest.param(None, 100.7, 10.0, 0.2, 0.3, id="first-byte-only"),
        # No output or first byte: the reported ttfb is the service's own,
        # measured from the speculation start, so it is rebased to the commit
        # (reported - 0.5 s) and kept within [0, ttfat].
        pytest.param(None, None, 10.0, 0.5, 0.0, id="no-output-capped-at-ttfat"),
        pytest.param(None, None, 0.8, 0.3, 0.2, id="no-output-after-commit"),
        pytest.param(None, None, 0.3, 0.0, 0.5, id="no-output-before-commit"),
    ],
)
async def test_committed_ttfat_measures_ttfb_from_the_commit(
    harness, monkeypatch, first_output, first_byte, reported_ttfb, ttfb, thinking_time
):
    race = fallback_llm._Race(started=100.0, hedge_started=100.5)
    attempt = fallback_llm._Attempt(
        index=0, race=race, first_output=first_output, first_byte=first_byte
    )
    race.winner = attempt
    race.attempts = [attempt]
    token = fallback_llm._attempt.set(attempt)
    monkeypatch.setattr(
        fallback_llm,
        "time",
        SimpleNamespace(monotonic=lambda: 101.0),
    )
    try:
        ttfat = TTFATMetricsData(
            processor="test",
            model="gemini",
            ttfb=reported_ttfb,
            ttfat=0.0,
            thinking_time=0.0,
        )
        frame = MetricsFrame(data=[ttfat])
        await harness.llm._receive(0, frame, FrameDirection.DOWNSTREAM)
        assert ttfat.ttfat == pytest.approx(0.5)
        assert ttfat.ttfb == pytest.approx(ttfb)
        assert ttfat.thinking_time == pytest.approx(thinking_time)
        assert 0 <= ttfat.ttfb <= ttfat.ttfat
        assert ttfat.thinking_time >= 0
        assert not any(isinstance(data, TTFBMetricsData) for data in frame.data)
    finally:
        fallback_llm._attempt.reset(token)


@pytest.mark.asyncio
async def test_buffered_speculative_metrics_rebase_when_turn_commits(
    harness, monkeypatch
):
    from pipecat.frames.frames import UserStoppedSpeakingFrame

    now = [100.25]
    monkeypatch.setattr(
        fallback_llm, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    race = fallback_llm._Race(started=100.0)
    attempts = [fallback_llm._Attempt(index=i, race=race) for i in range(2)]
    race.attempts.extend(attempts)
    harness.llm._active_race = race
    harness.llm._latest[:] = attempts
    harness.llm._speculation_gate.begin_speculation(True)

    token = fallback_llm._attempt.set(attempts[0])
    try:
        ttfb = TTFBMetricsData(processor="test", model="gemini", value=0.0)
        await harness.llm._receive(
            0, MetricsFrame(data=[ttfb]), FrameDirection.DOWNSTREAM
        )
    finally:
        fallback_llm._attempt.reset(token)

    now[0] = 100.3
    token = fallback_llm._attempt.set(attempts[1])
    try:
        ttfb = TTFBMetricsData(processor="test", model="gemini", value=0.0)
        await harness.llm._receive(
            1, MetricsFrame(data=[ttfb]), FrameDirection.DOWNSTREAM
        )
        now[0] = 100.4
        ttfat = TTFATMetricsData(
            processor="test",
            model="gemini",
            ttfb=0.0,
            ttfat=0.0,
            thinking_time=0.0,
        )
        await harness.llm._receive(
            1, MetricsFrame(data=[ttfat]), FrameDirection.DOWNSTREAM
        )
    finally:
        fallback_llm._attempt.reset(token)

    now[0] = 100.5
    await control(harness, UserStoppedSpeakingFrame())

    assert race.hedge_started == pytest.approx(100.5)
    assert attempts[0].buffered[0][0].data[0].value == pytest.approx(0)
    assert attempts[1].buffered[0][0].data[0].value == pytest.approx(0)
    ttfat = attempts[1].buffered[1][0].data[0]
    assert ttfat.ttfat == pytest.approx(0)
    assert ttfat.ttfb == pytest.approx(0)
    assert ttfat.thinking_time == pytest.approx(0)


@pytest.mark.asyncio
async def test_winner_metrics_rebase_after_generation_finishes_before_commit(
    harness, monkeypatch
):
    from pipecat.frames.frames import UserStoppedSpeakingFrame

    install_streams(
        harness, monkeypatch, Stream(chunk("answer")), Stream(chunk("backup"))
    )
    assert await generate(harness, speculation=True) == []
    assert harness.llm._active_race is None
    assert harness.llm._pending_speculative_race is not None

    await control(harness, UserStoppedSpeakingFrame())

    metrics = [
        data
        for frame in harness.downstream.frames
        if isinstance(frame, MetricsFrame)
        for data in frame.data
        if isinstance(data, (TTFBMetricsData, TTFATMetricsData))
    ]
    assert len(metrics) == 2
    ttfb = next(data for data in metrics if isinstance(data, TTFBMetricsData))
    ttfat = next(data for data in metrics if isinstance(data, TTFATMetricsData))
    assert ttfb.value == pytest.approx(0)
    assert ttfat.ttfat == pytest.approx(0)
    assert ttfat.ttfb == pytest.approx(0)
    assert ttfat.thinking_time == pytest.approx(0)
    assert [
        frame.text
        for frame in harness.downstream.frames
        if isinstance(frame, LLMTextFrame)
    ] == ["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_backup", [False, True], ids=["primary", "backup"])
async def test_speculative_tools_cancel_both_directions_without_execution(
    harness, monkeypatch, use_backup
):
    tools = Stream(
        chunk(
            parts=[Part(function_call=FunctionCall(name="book", args={}, id="booking"))]
        )
    )
    primary = Stream(ConnectionError("primary")) if use_backup else tools
    backup = tools if use_backup else Stream(chunk("unused"))
    install_streams(harness, monkeypatch, primary, backup)
    handler = AsyncMock()
    harness.llm.register_function("book", handler)

    assert await generate(harness, speculation=True) == []

    handler.assert_not_awaited()
    assert tools.closed.is_set()
    assert not harness.llm._speculation_gate.is_speculating
    for capture in (harness.upstream, harness.downstream):
        cancellations = [
            frame
            for frame in capture.frames
            if isinstance(frame, UserTurnResponseCancelFrame)
        ]
        assert len(cancellations) == 1
        assert type(cancellations[0]) is UserTurnResponseCancelFrame
        assert not any(
            isinstance(frame, FunctionCallsFromLLMInfoFrame) for frame in capture.frames
        )


@pytest.mark.asyncio
async def test_speculative_winner_releases_once_on_confirmed_turn(harness, monkeypatch):
    from pipecat.frames.frames import UserStoppedSpeakingFrame

    install_streams(
        harness,
        monkeypatch,
        Stream(ConnectionError("primary")),
        Stream(chunk("backup")),
    )
    assert await generate(harness, speculation=True) == []
    await control(harness, UserStoppedSpeakingFrame())
    assert [
        f.text for f in harness.downstream.frames if isinstance(f, LLMTextFrame)
    ] == ["backup"]
    assert (
        sum(isinstance(f, LLMFullResponseStartFrame) for f in harness.downstream.frames)
        == 1
    )
    assert (
        sum(isinstance(f, LLMFullResponseEndFrame) for f in harness.downstream.frames)
        == 1
    )


@pytest.mark.asyncio
async def test_superseded_speculative_backup_cannot_release_old_text(
    harness, monkeypatch
):
    from pipecat.frames.frames import UserStoppedSpeakingFrame

    primary = Stream(ConnectionError("primary"))
    install_streams(harness, monkeypatch, primary, Stream(chunk("old")))
    assert await generate(harness, speculation=True) == []
    primary.steps = (chunk("new"),)
    assert await generate(harness) == ["new"]
    await control(harness, UserStoppedSpeakingFrame())
    assert [
        f.text for f in harness.downstream.frames if isinstance(f, LLMTextFrame)
    ] == ["new"]


@pytest.mark.asyncio
async def test_speculative_no_output_hedge_gets_full_1200ms_after_commit(
    harness, monkeypatch
):
    from pipecat.frames.frames import UserStoppedSpeakingFrame

    harness.llm._fallback_after_secs = 1.2
    harness.llm._first_output_timeout_secs = 5
    harness.llm.primary._stream_idle_timeout_secs = 5
    primary, backup = Stream(asyncio.Event()), Stream(chunk("backup"))
    install_streams(harness, monkeypatch, primary, backup)
    await harness.llm.queue_frame(LLMContextFrame(LLMContext(), speculation=True))
    await asyncio.wait_for(primary.started.wait(), 1)
    # More than a hedge interval spent guessing must not launch a backup.
    await asyncio.sleep(1.25)
    assert not backup.started.is_set()
    committed = asyncio.get_running_loop().time()
    await control(harness, UserStoppedSpeakingFrame())
    await asyncio.wait_for(backup.started.wait(), 2)
    assert asyncio.get_running_loop().time() - committed >= 1.2
    async with asyncio.timeout(1):
        while not any(isinstance(f, LLMTextFrame) for f in harness.downstream.frames):
            await asyncio.sleep(0.001)
    assert [
        f.text for f in harness.downstream.frames if isinstance(f, LLMTextFrame)
    ] == ["backup"]
    # Loser cleanup runs beside the winner's output so it never delays the
    # reply; the first text can therefore arrive before the primary closes.
    await asyncio.wait_for(primary.closed.wait(), 1)
    timings = [
        data
        for f in harness.downstream.frames
        if isinstance(f, MetricsFrame)
        for data in f.data
        if isinstance(data, (TTFBMetricsData, TTFATMetricsData))
    ]
    assert len(timings) == 2
    for data in timings:
        # The 1.25 seconds spent speculating are excluded from public latency.
        elapsed = data.value if isinstance(data, TTFBMetricsData) else data.ttfat
        assert 1.2 <= elapsed < 2
    ttfb = next(data for data in timings if isinstance(data, TTFBMetricsData))
    ttfat = next(data for data in timings if isinstance(data, TTFATMetricsData))
    assert ttfat.ttfb == pytest.approx(ttfb.value, abs=0.02)
    assert ttfat.thinking_time == pytest.approx(ttfat.ttfat - ttfat.ttfb)


@pytest.mark.asyncio
async def test_speculation_started_during_commit_flush_keeps_hedge(
    harness, monkeypatch
):
    from pipecat.frames.frames import UserStoppedSpeakingFrame

    harness.llm._first_output_timeout_secs = 5
    harness.llm.primary._stream_idle_timeout_secs = 5
    primary, backup = Stream(asyncio.Event()), Stream(chunk("backup"))
    install_streams(harness, monkeypatch, primary, backup)
    entered, release = asyncio.Event(), asyncio.Event()
    original_flush = harness.llm._flush_output

    async def flush():
        if asyncio.current_task() is committing:
            entered.set()
            await release.wait()
        await original_flush()

    monkeypatch.setattr(harness.llm, "_flush_output", flush)
    committing = asyncio.create_task(
        harness.llm.process_frame(UserStoppedSpeakingFrame(), DOWN)
    )
    generating = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        generating = asyncio.create_task(
            harness.llm._generate(LLMContextFrame(LLMContext(), speculation=True))
        )
        await asyncio.wait_for(primary.started.wait(), 1)
        assert harness.llm._speculation_gate.is_speculating
        release.set()
        await asyncio.wait_for(committing, 1)
        await asyncio.wait_for(backup.started.wait(), 1)
        await asyncio.wait_for(generating, 1)
        assert [
            f.text for f in harness.downstream.frames if isinstance(f, LLMTextFrame)
        ] == ["backup"]
        assert primary.closed.is_set()
    finally:
        release.set()
        tasks = [committing, *([generating] if generating else [])]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_withdrawn_speculation_retires_request_and_hedge(harness, monkeypatch):
    from pipecat.frames.frames import (
        UserStoppedSpeakingFrame,
        UserTurnResponseCancelFrame,
    )

    harness.llm._first_output_timeout_secs = 5
    primary, backup = Stream(asyncio.Event()), Stream(chunk("backup"))
    install_streams(harness, monkeypatch, primary, backup)
    await harness.llm.queue_frame(LLMContextFrame(LLMContext(), speculation=True))
    await asyncio.wait_for(primary.started.wait(), 1)
    await control(harness, UserTurnResponseCancelFrame())
    assert primary.closed.is_set()
    await asyncio.sleep(0.04)
    assert not backup.started.is_set()
    # The final turn has its own full hedge budget.
    await control(harness, UserStoppedSpeakingFrame())
    primary.steps = (chunk("committed answer"),)
    assert await generate(harness) == ["committed answer"]
    assert not backup.started.is_set()


@pytest.mark.asyncio
async def test_summary_uses_primary_even_after_backup_won(harness, monkeypatch):
    from pipecat.frames.frames import (
        LLMContextSummaryRequestFrame,
        LLMContextSummaryResultFrame,
    )

    install_streams(
        harness, monkeypatch, Stream(asyncio.Event()), Stream(chunk("backup"))
    )
    await generate(harness)
    monkeypatch.setattr(
        harness.llm.primary, "_generate_summary", AsyncMock(return_value=("summary", 3))
    )
    await control(
        harness,
        LLMContextSummaryRequestFrame(
            request_id="summary",
            context=LLMContext(),
            min_messages_to_keep=1,
            target_context_tokens=100,
            summarization_prompt="summarize",
        ),
    )
    async with asyncio.timeout(1):
        while not any(
            isinstance(f, LLMContextSummaryResultFrame)
            for f in harness.downstream.frames
        ):
            await asyncio.sleep(0.001)
    results = [
        f
        for f in harness.downstream.frames
        if isinstance(f, LLMContextSummaryResultFrame)
    ]
    assert len(results) == 1 and results[0].summary == "summary"


@pytest.mark.asyncio
async def test_only_winner_is_accepted_by_dograh_tool_owner(
    harness, monkeypatch, simple_workflow
):
    from api.services.workflow.pipecat_engine import PipecatEngine

    engine = PipecatEngine(
        llm=harness.llm,
        workflow=simple_workflow,
        context=LLMContext(),
        call_context_vars={},
        workflow_run_id=123,
    )
    saved = []

    async def save(params):
        saved.append(params.tool_call_id)
        await params.result_callback({"saved": True})

    harness.llm.register_function("book", engine.active_agent.bind_tool(engine, save))
    release = asyncio.Event()
    primary = Stream(
        release,
        chunk(
            parts=[Part(function_call=FunctionCall(name="book", args={}, id="primary"))]
        ),
    )
    backup = Stream(
        release,
        chunk(
            parts=[Part(function_call=FunctionCall(name="book", args={}, id="backup"))]
        ),
    )
    install_streams(harness, monkeypatch, primary, backup)
    task = asyncio.create_task(generate(harness))
    await asyncio.wait_for(backup.started.wait(), 1)
    release.set()
    await task
    await engine.active_agent.finish_tool_calls()
    assert len(saved) == 1
    assert len(engine._gathered_context["tool_results"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spoke_now", "start_lines"), [(False, ["Checking."]), (True, [])]
)
async def test_start_line_follows_the_response_that_called_the_tool(
    harness, monkeypatch, simple_workflow, spoke_now, start_lines
):
    """Only this response's lead-in replaces the start line, never an earlier one's."""
    from api.services.workflow.pipecat_engine import PipecatEngine

    engine = PipecatEngine(
        llm=harness.llm,
        workflow=simple_workflow,
        context=LLMContext(),
        call_context_vars={},
        workflow_run_id=123,
    )
    engine.speech_playback.observe_responses(harness.llm)
    engine.queue_speech = AsyncMock()
    messages = TypeAdapter(ToolMessages).validate_python(
        [{"event": "start", "type": "text", "text": "Checking."}]
    )
    handled = []

    async def handler(params):
        handled.append(params.tool_call_id)
        await params.result_callback({"saved": True})

    harness.llm.register_function(
        "book", engine.active_agent.bind_tool(engine, handler, messages=messages)
    )
    # The previous response spoke; the backup answers this one after the primary stalls.
    install_streams(
        harness, monkeypatch, Stream(chunk("Let me check.")), Stream(chunk("unused"))
    )
    await generate(harness)
    tool = Part(function_call=FunctionCall(name="book", args={}, id="booking"))
    parts = [Part(text="One moment."), tool] if spoke_now else [tool]
    install_streams(
        harness, monkeypatch, Stream(asyncio.Event()), Stream(chunk(parts=parts))
    )
    await generate(harness)
    await engine.active_agent.finish_tool_calls()
    assert handled == ["booking"]
    assert [
        c.kwargs["text"] for c in engine.queue_speech.await_args_list
    ] == start_lines
