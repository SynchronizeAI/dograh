"""A response that picks an edge with transition speech asks no question of its own.

The edge's fixed line is the turn's question. Question sentences wait for the
end of the response: they are dropped when the response picked such an edge and
spoken in order otherwise. Statements before the first held question stream
immediately; following sentences wait with it to preserve their order.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from loguru import logger
from pipecat.frames.frames import (
    AggregatedTextFrame,
    FunctionCallFromLLM,
    FunctionCallsFromLLMInfoFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.mock_transport import MockTransport
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_transport import TransportParams

from api.services.pipecat import agent_runtime_factory as factory_module
from api.services.pipecat.agent_generation_processor import AgentGenerationProcessor
from api.services.pipecat.agent_runtime_factory import AgentRuntimeFactory
from api.services.pipecat.recording_router_processor import RecordingRouterProcessor
from api.services.workflow.dto import (
    AgentNodeData,
    EdgeDataDTO,
    EndCallNodeData,
    Position,
    ReactFlowDTO,
    RFEdgeDTO,
    RFNodeDTO,
    StartCallNodeData,
)
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.conftest import (
    AGENT_SYSTEM_PROMPT,
    END_CALL_SYSTEM_PROMPT,
    START_CALL_SYSTEM_PROMPT,
)
from api.tests.pipecat_test_utils import run_engine_test_pipeline, stub_agent_runtime
from pipecat.tests import MockLLMService, MockTTSService

FIXED_LINE = "Is the number you are calling from the best one to reach you?"
STATEMENT = "Your visit is noted."
QUESTION = "Which sink is leaking?"


def _workflow(*, recorded_line: bool = False) -> WorkflowGraph:
    """Start has two edges: "Collect Info" speaks a fixed line, "End Call" does not.

    ``recorded_line`` adds a third that plays a recording as its line.
    """

    def edge(id, source, target, label, **data):
        return RFEdgeDTO(
            id=id,
            source=source,
            target=target,
            data=EdgeDataDTO(label=label, condition=label, **data),
        )

    flags = dict(allow_interrupt=False, add_global_prompt=False)
    return WorkflowGraph(
        ReactFlowDTO(
            nodes=[
                RFNodeDTO(
                    id="start",
                    type="startCall",
                    position=Position(x=0, y=0),
                    data=StartCallNodeData(
                        name="Start",
                        prompt=START_CALL_SYSTEM_PROMPT,
                        is_start=True,
                        **flags,
                    ),
                ),
                RFNodeDTO(
                    id="agent",
                    type="agentNode",
                    position=Position(x=0, y=200),
                    data=AgentNodeData(
                        name="Agent", prompt=AGENT_SYSTEM_PROMPT, **flags
                    ),
                ),
                RFNodeDTO(
                    id="end",
                    type="endCall",
                    position=Position(x=0, y=400),
                    data=EndCallNodeData(
                        name="End", prompt=END_CALL_SYSTEM_PROMPT, is_end=True, **flags
                    ),
                ),
            ],
            edges=[
                edge(
                    "s-a",
                    "start",
                    "agent",
                    "Collect Info",
                    transition_speech=FIXED_LINE,
                ),
                edge("s-e", "start", "end", "End Call"),
                edge("a-e", "agent", "end", "End Call"),
                *(
                    [
                        edge(
                            "s-r",
                            "start",
                            "end",
                            "Recorded Line",
                            transition_speech_type="audio",
                            transition_speech_recording_id="7",
                        )
                    ]
                    if recorded_line
                    else []
                ),
            ],
        )
    )


class _NoInfoFrameLLM(MockLLMService):
    """Runs its tool calls at once and sends no info frame: the Anthropic and Bedrock way."""

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if not isinstance(frame, FunctionCallsFromLLMInfoFrame):
            await super().push_frame(frame, direction)

    async def _run_or_defer_function_calls(self, function_calls, **_):
        await self.run_function_calls(function_calls)


async def _spoken(
    first_step: list,
    *,
    then: list[list],
    llm_class: type[MockLLMService] = MockLLMService,
    chunk_delay: float = 0.001,
) -> tuple[list[str], list[str]]:
    """Run the engine with the generation stage in place; return what TTS spoke and the logs."""
    llm = llm_class(mock_steps=[first_step, *then], chunk_delay=chunk_delay)
    tts = MockTTSService(mock_audio_duration_ms=40, frame_delay=0)
    transport = MockTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_end_silence_secs=0,
        ),
    )
    context = LLMContext()
    assistant = LLMContextAggregatorPair(
        context, assistant_params=LLMAssistantAggregatorParams()
    ).assistant()
    engine = PipecatEngine(
        llm=llm,
        context=context,
        workflow=_workflow(),
        call_context_vars={},
        workflow_run_id=1,
    )
    generation = AgentGenerationProcessor(
        fixed_line_tools=lambda: engine.active_agent.fixed_line_tools()
    )
    task = PipelineWorker(
        Pipeline(
            [transport.input(), llm, generation, tts, transport.output(), assistant]
        ),
        params=PipelineParams(),
        enable_rtvi=False,
    )
    engine.call_worker = task

    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), level="INFO")
    try:
        with patch(
            "api.db:db_client.get_organization_id_by_workflow_run_id",
            new_callable=AsyncMock,
            return_value=1,
        ):
            await run_engine_test_pipeline(task, engine, transport)
    finally:
        logger.remove(sink)
    return [text.strip() for text in tts.received_texts], logs


def _call(name: str, call_id: str) -> list:
    return MockLLMService.create_function_call_chunks(
        function_name=name, arguments={}, tool_call_id=call_id
    )


def _text_then_call(text: str, name: str) -> list:
    return MockLLMService.create_text_chunks(text)[:-1] + _call(name, "call_1")


class TestFixedLineTurnQuestion:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_class", [MockLLMService, _NoInfoFrameLLM])
    @pytest.mark.parametrize("chunk_delay", [0.001, 0])
    async def test_question_is_dropped_when_the_edge_speaks_a_fixed_line(
        self, llm_class, chunk_delay
    ):
        spoken, logs = await _spoken(
            _text_then_call(f"{STATEMENT} {QUESTION}", "collect_info"),
            then=[
                _call("end_call", "call_2"),
                MockLLMService.create_text_chunks("Bye."),
            ],
            llm_class=llm_class,
            chunk_delay=chunk_delay,
        )

        assert spoken[:2] == [STATEMENT, FIXED_LINE]
        assert QUESTION not in spoken
        assert any("dropped 1 model question(s)" in line for line in logs)

    @pytest.mark.asyncio
    async def test_question_is_spoken_after_the_statements_without_such_an_edge(self):
        spoken, logs = await _spoken(
            _text_then_call(f"{STATEMENT} {QUESTION}", "end_call"),
            then=[MockLLMService.create_text_chunks("Bye.")],
        )

        assert spoken[:2] == [STATEMENT, QUESTION]
        assert FIXED_LINE not in spoken
        assert not any("dropped" in line for line in logs)

    @pytest.mark.asyncio
    async def test_response_without_a_question_is_unchanged(self):
        spoken, logs = await _spoken(
            _text_then_call(f"{STATEMENT} We will call you back.", "collect_info"),
            then=[
                _call("end_call", "call_2"),
                MockLLMService.create_text_chunks("Bye."),
            ],
        )

        assert spoken[:3] == [STATEMENT, "We will call you back.", FIXED_LINE]
        assert not any("dropped" in line for line in logs)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("chunk_delay", [0.001, 0])
    async def test_speech_after_a_held_question_keeps_its_order(self, chunk_delay):
        text = f"{QUESTION} {STATEMENT}"
        released, _ = await _spoken(
            _text_then_call(text, "end_call"),
            then=[MockLLMService.create_text_chunks("Bye.")],
            chunk_delay=chunk_delay,
        )
        dropped, _ = await _spoken(
            _text_then_call(text, "collect_info"),
            then=[
                _call("end_call", "call_2"),
                MockLLMService.create_text_chunks("Bye."),
            ],
            chunk_delay=chunk_delay,
        )

        assert released[:2] == [QUESTION, STATEMENT]
        assert dropped[:2] == [STATEMENT, FIXED_LINE]


# -- the generation stage on its own ---------------------------------------


def _text(text: str, size: int = 7) -> list[LLMTextFrame]:
    return [LLMTextFrame(text[i : i + size]) for i in range(0, len(text), size)]


def _picks_collect_info(frame_type, call_id: str = "call_1") -> TextFrame:
    return frame_type(
        function_calls=[
            FunctionCallFromLLM(
                function_name="collect_info",
                tool_call_id=call_id,
                arguments={},
                context=None,
            )
        ]
    )


def _stage() -> AgentGenerationProcessor:
    return AgentGenerationProcessor(fixed_line_tools=lambda: {"collect_info"})


async def _through_stage(frames: list) -> list[str]:
    """The sentences the stage passes on for these frames."""
    down, _ = await run_test(_stage(), frames_to_send=frames)
    return [f.text for f in down if isinstance(f, AggregatedTextFrame)]


SENTENCES = [
    (STATEMENT, QUESTION),
    ("지금 확인할게요.", "어느 싱크대가 샜나요?"),
    ("지금 확인할게요.", "어느 싱크대가 샜나요？"),
]


class TestGenerationStage:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("statement, question", SENTENCES)
    @pytest.mark.parametrize(
        "frame_type", [FunctionCallsFromLLMInfoFrame, FunctionCallsStartedFrame]
    )
    async def test_question_is_dropped_whichever_frame_announces_the_call(
        self, statement, question, frame_type
    ):
        frames = [
            LLMFullResponseStartFrame(),
            *_text(f"{statement} {question}"),
            _picks_collect_info(frame_type),
            LLMFullResponseEndFrame(),
        ]

        assert await _through_stage(frames) == [statement]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("statement, question", SENTENCES)
    async def test_question_is_spoken_when_no_fixed_line_is_picked(
        self, statement, question
    ):
        frames = [
            LLMFullResponseStartFrame(),
            *_text(f"{statement} {question}"),
            LLMFullResponseEndFrame(),
        ]

        assert await _through_stage(frames) == [statement, question]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "quoted_question",
        ["“Which sink is leaking?”", "«Quel évier fuit ?»"],
    )
    async def test_a_quoted_question_is_still_a_question(self, quoted_question):
        frames = [
            LLMFullResponseStartFrame(),
            *_text(f"{quoted_question} Noted."),
            _picks_collect_info(FunctionCallsFromLLMInfoFrame),
            LLMFullResponseEndFrame(),
        ]

        assert await _through_stage(frames) == ["Noted."]

    @pytest.mark.asyncio
    async def test_a_started_frame_after_the_response_has_ended_picks_nothing(self):
        """Where the info frame exists, a transition waits for the speech to end."""
        frames = [
            LLMFullResponseStartFrame(),
            *_text(STATEMENT),
            _picks_collect_info(FunctionCallsFromLLMInfoFrame),
            LLMFullResponseEndFrame(),
            SleepFrame(0.1),
            _picks_collect_info(FunctionCallsStartedFrame),
            SleepFrame(0.1),
            LLMFullResponseStartFrame(),
            *_text(QUESTION),
            LLMFullResponseEndFrame(),
        ]

        assert await _through_stage(frames) == [STATEMENT, QUESTION]

    @pytest.mark.asyncio
    async def test_empty_fixed_line_snapshot_is_not_read_again_at_response_start(self):
        snapshot_reads = 0

        def fixed_line_tools():
            nonlocal snapshot_reads
            snapshot_reads += 1
            return set() if snapshot_reads == 1 else {"collect_info"}

        stage = AgentGenerationProcessor(fixed_line_tools=fixed_line_tools)
        down, _ = await run_test(
            stage,
            frames_to_send=[
                _picks_collect_info(FunctionCallsStartedFrame),
                LLMFullResponseStartFrame(),
                LLMTextFrame(QUESTION),
                LLMFullResponseEndFrame(),
            ],
        )

        assert snapshot_reads == 1
        assert [f.text for f in down if isinstance(f, LLMTextFrame)] == [QUESTION]
        assert not any(isinstance(f, AggregatedTextFrame) for f in down)

    @pytest.mark.asyncio
    async def test_started_only_pick_is_seen_after_an_info_frame_response(self):
        frames = [
            LLMFullResponseStartFrame(),
            *_text(STATEMENT),
            _picks_collect_info(FunctionCallsFromLLMInfoFrame),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(),
            *_text(QUESTION),
            _picks_collect_info(FunctionCallsStartedFrame, "call_2"),
            LLMFullResponseEndFrame(),
        ]

        assert await _through_stage(frames) == [STATEMENT]

    @pytest.mark.asyncio
    async def test_empty_snapshot_is_cleared_before_the_next_response(self):
        snapshot_reads = 0

        def fixed_line_tools():
            nonlocal snapshot_reads
            snapshot_reads += 1
            return set() if snapshot_reads == 1 else {"collect_info"}

        stage = AgentGenerationProcessor(fixed_line_tools=fixed_line_tools)
        down, _ = await run_test(
            stage,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame("The first node has no fixed line."),
                LLMFullResponseEndFrame(),
                # System tool-call frames overtake queued text frames. Keep
                # the second response after the first response's end so its
                # call is not attributed to the preceding node snapshot.
                SleepFrame(0.1),
                LLMFullResponseStartFrame(),
                LLMTextFrame(QUESTION),
                _picks_collect_info(FunctionCallsStartedFrame, "call_2"),
                LLMFullResponseEndFrame(),
            ],
        )

        assert snapshot_reads == 2
        assert [f.text for f in down if isinstance(f, LLMTextFrame)] == [
            "The first node has no fixed line."
        ]
        assert not any(isinstance(f, AggregatedTextFrame) for f in down)

    @pytest.mark.asyncio
    async def test_interruption_clears_the_pick_for_the_next_response(self):
        frames = [
            LLMFullResponseStartFrame(),
            *_text("Which sink is leaking?"),
            _picks_collect_info(FunctionCallsFromLLMInfoFrame),
            SleepFrame(0.1),
            InterruptionFrame(),
            SleepFrame(0.1),
            LLMFullResponseStartFrame(),
            *_text("Which room?"),
            LLMFullResponseEndFrame(),
        ]

        assert await _through_stage(frames) == ["Which room?"]

    @pytest.mark.asyncio
    async def test_restarted_response_drops_what_the_abandoned_one_held(self):
        frames = [
            LLMFullResponseStartFrame(),
            *_text(QUESTION),
            LLMFullResponseStartFrame(),
            *_text(STATEMENT),
            LLMFullResponseEndFrame(),
        ]

        assert await _through_stage(frames) == [STATEMENT]

    @pytest.mark.asyncio
    async def test_text_marked_skip_tts_passes_through_unheld(self):
        marked = LLMTextFrame(QUESTION)
        marked.skip_tts = True

        down, _ = await run_test(
            _stage(),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                marked,
                LLMFullResponseEndFrame(),
            ],
        )

        texts = [f for f in down if isinstance(f, TextFrame)]
        assert [(type(f), f.text, f.skip_tts) for f in texts] == [
            (LLMTextFrame, QUESTION, True)
        ]

    @pytest.mark.asyncio
    async def test_statement_is_spoken_when_the_next_sentence_begins_not_when_the_response_ends(
        self,
    ):
        """English TTS holds a text frame's sentence until the next one begins."""
        tts = _TimedTTS(
            mock_audio_duration_ms=40, frame_delay=0, pause_frame_processing=False
        )

        await run_test(
            Pipeline([_stage(), tts]),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                *_text(STATEMENT),
                LLMTextFrame(" Which"),
                SleepFrame(0.5),
                LLMTextFrame(" sink is leaking?"),
                LLMFullResponseEndFrame(),
            ],
        )

        assert tts.spoken_at[QUESTION] - tts.spoken_at[STATEMENT] > 0.3


class _TimedTTS(MockTTSService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spoken_at: dict[str, float] = {}

    async def run_tts(self, text, context_id):
        self.spoken_at[text.strip()] = time.monotonic()
        async for frame in super().run_tts(text, context_id):
            yield frame


# -- how the stage is told which tools carry a fixed line --------------------


def test_fixed_line_tools_are_the_edges_that_speak_text_or_play_a_recording():
    runtime = stub_agent_runtime()
    runtime.current_node = _workflow(recorded_line=True).nodes["start"]

    assert runtime.fixed_line_tools() == {"collect_info", "recorded_line"}


def test_no_fixed_line_tools_while_the_recording_router_is_routing():
    runtime = stub_agent_runtime()
    runtime.current_node = _workflow().nodes["start"]
    runtime.recording_router = RecordingRouterProcessor(
        audio_sample_rate=16000, fetch_recording_audio=AsyncMock()
    )
    runtime.recording_router.set_enabled(True)

    assert runtime.fixed_line_tools() == set()


@pytest.mark.asyncio
async def test_an_agent_visits_generation_stage_gets_the_runtimes_fixed_line_tools(
    monkeypatch,
):
    runtime = stub_agent_runtime()
    runtime.worker = None
    stage = Mock()
    monkeypatch.setattr(factory_module, "AgentGenerationProcessor", stage)
    monkeypatch.setattr(factory_module, "build_agent_generation_pipeline", Mock())
    monkeypatch.setattr(factory_module, "create_agent_worker", Mock())
    factory = AgentRuntimeFactory(
        organization_id=1,
        workflow_run_id=1,
        call_worker=Mock(add_workers=AsyncMock()),
        audio_config=None,
        callbacks_factory=lambda visit_id: Mock(),
    )

    await factory.attach(runtime)

    assert stage.call_args.kwargs["fixed_line_tools"] == runtime.fixed_line_tools
