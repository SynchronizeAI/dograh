"""An end-node tool result must not close a still-producing voice session."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    SpeechBoundaryFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.enums import EndTaskReason

from api.services.pipecat.speech_playback import PlaybackOutcome
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.text_chat_runner import (
    _ResponseWindowState,
    _TextChatCaptureProcessor,
)


@pytest.fixture
def engine(three_node_workflow_no_variable_extraction):
    workflow = three_node_workflow_no_variable_extraction
    engine = PipecatEngine(
        task=SimpleNamespace(queue_frame=AsyncMock()),
        workflow=workflow,
        call_context_vars={},
    )
    engine.active_agent.current_node = workflow.nodes["agent"]
    engine.set_node = AsyncMock()
    engine._perform_variable_extraction_if_needed = AsyncMock()
    engine.perform_final_variable_extraction = AsyncMock()
    return engine


async def transition_to_end(engine):
    transition = await engine._create_transition_func("move_to_end_call", "end")
    result = AsyncMock()
    await transition(SimpleNamespace(arguments={}, result_callback=result))
    assert result.await_args.args == ({"status": "done"},)
    return result.await_args.kwargs["properties"].on_context_updated


@pytest.mark.asyncio
async def test_transition_from_inactive_node_is_rejected(engine):
    transition = await engine._create_transition_func(
        "stale_transition",
        "end",
        source_node=engine.active_agent.workflow.nodes["start"],
    )
    result = AsyncMock()

    await transition(SimpleNamespace(arguments={}, result_callback=result))

    engine._perform_variable_extraction_if_needed.assert_not_awaited()
    engine.set_node.assert_not_awaited()
    assert result.await_args.args == (
        {"status": "error", "error": "Transition is no longer available"},
    )
    assert result.await_args.kwargs["properties"].run_llm is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "expected_run_llm"),
    [("", None), (None, None), ("How can I help?", False)],
)
async def test_recorded_transition_wait_requires_context_text(
    engine, transcript, expected_run_llm
):
    engine.active_agent.workflow.nodes["agent"].respond_immediately = False
    engine._fetch_recording_audio = object()
    engine.queue_speech = AsyncMock(
        return_value=SimpleNamespace(outcome=PlaybackOutcome.PLAYED, text=transcript)
    )
    transition = await engine._create_transition_func(
        "move_to_agent",
        "agent",
        transition_speech_type="audio",
        transition_speech_recording_id="1",
        source_node=engine.active_agent.workflow.nodes["agent"],
    )
    result = AsyncMock()

    await transition(SimpleNamespace(arguments={}, result_callback=result))

    assert result.await_args.kwargs["properties"].run_llm is expected_run_llm


@pytest.mark.asyncio
@pytest.mark.parametrize("is_realtime", [True, False])
@pytest.mark.parametrize("playback", ["delayed", "already_started", "already_finished"])
async def test_end_node_waits_for_goodbye_playback(engine, is_realtime, playback):
    engine._is_realtime = is_realtime
    # Old playback must not count as the end node's goodbye.
    await engine.should_mute_user(BotStartedSpeakingFrame())
    await engine.should_mute_user(BotStoppedSpeakingFrame())
    finish = await transition_to_end(engine)
    assert engine._mute_pipeline

    # Realtime may start (or finish) before the context callback is scheduled.
    if playback != "delayed":
        engine.speech_playback.after_output(None, LLMFullResponseStartFrame())
        engine.speech_playback.note_output()
    if playback == "already_finished":
        engine.speech_playback.after_output(None, LLMFullResponseEndFrame())

    finishing = asyncio.create_task(finish())
    try:
        if playback != "already_finished":
            await asyncio.sleep(0)
            engine.call_worker.queue_frame.assert_not_awaited()
            if playback == "delayed":
                engine.speech_playback.after_output(None, LLMFullResponseStartFrame())
                engine.speech_playback.note_output()
            # Finishing inference does not mean the audio has reached the caller.
            await engine.should_mute_user(LLMFullResponseEndFrame())
            await asyncio.sleep(0)
            engine.call_worker.queue_frame.assert_not_awaited()
            engine.perform_final_variable_extraction.assert_not_awaited()
            engine.speech_playback.after_output(None, LLMFullResponseEndFrame())

        await asyncio.wait_for(finishing, timeout=1)
        engine.perform_final_variable_extraction.assert_awaited_once()
        terminal = engine.call_worker.queue_frame.await_args.args[0]
        assert isinstance(terminal, EndFrame)
        assert terminal.reason == EndTaskReason.END_CALL.value
        assert not engine._bot_is_speaking
    finally:
        if not finishing.done():
            finishing.cancel()
            await asyncio.gather(finishing, return_exceptions=True)


@pytest.mark.asyncio
async def test_caller_hangup_can_cancel_while_goodbye_is_pending(engine):
    finish = await transition_to_end(engine)
    finishing = asyncio.create_task(finish())
    try:
        await asyncio.sleep(0)
        await engine.end_call_with_reason(
            EndTaskReason.USER_HANGUP.value, abort_immediately=True
        )
        terminal = engine.call_worker.queue_frame.await_args.args[0]
        assert isinstance(terminal, CancelFrame)
        assert terminal.reason == EndTaskReason.USER_HANGUP.value
    finally:
        finishing.cancel()
        await asyncio.gather(finishing, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("starts", [False, True])
async def test_end_node_wait_is_bounded_when_audio_stalls(engine, starts):
    finish = await transition_to_end(engine)
    speech = next(iter(engine.speech_playback.pending.values()))
    asyncio.get_running_loop().call_later(
        0.01, speech.finish, PlaybackOutcome.TIMED_OUT
    )
    if starts:
        engine.speech_playback.after_output(None, LLMFullResponseStartFrame())
        engine.speech_playback.note_output()
    await asyncio.wait_for(finish(), timeout=1)
    assert isinstance(engine.call_worker.queue_frame.await_args.args[0], EndFrame)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["static", "llm", "recording"])
async def test_text_delivery_completes_the_shared_playback_wait(engine, source):
    context = LLMContext()
    window = _ResponseWindowState()
    capture = _TextChatCaptureProcessor(window, context, engine)
    capture.push_frame = AsyncMock()
    speech = (
        engine.speech_playback.expect_response()
        if source == "llm"
        else engine.speech_playback.create()
    )
    waiting = asyncio.create_task(speech.wait())
    if source != "llm":
        await capture.process_frame(
            SpeechBoundaryFrame(speech.id, True), FrameDirection.DOWNSTREAM
        )
    try:
        if source == "static":
            await capture.process_frame(
                TTSSpeakFrame("Goodbye!"), FrameDirection.DOWNSTREAM
            )
            assert window.outputs == ["Goodbye!"]
            await capture.process_frame(
                SpeechBoundaryFrame(speech.id, False), FrameDirection.DOWNSTREAM
            )
        else:
            start = (
                LLMFullResponseStartFrame() if source == "llm" else TTSStartedFrame()
            )
            stop = LLMFullResponseEndFrame() if source == "llm" else TTSStoppedFrame()
            await capture.process_frame(start, FrameDirection.DOWNSTREAM)
            await capture.process_frame(
                LLMTextFrame("Goodbye!"), FrameDirection.DOWNSTREAM
            )
            await asyncio.sleep(0)
            assert not waiting.done()
            await capture.process_frame(stop, FrameDirection.DOWNSTREAM)
            if source != "llm":
                await capture.process_frame(
                    SpeechBoundaryFrame(speech.id, False), FrameDirection.DOWNSTREAM
                )
        assert await asyncio.wait_for(waiting, timeout=1)
    finally:
        if not waiting.done():
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)


async def respond(engine, text, *, play=False):
    """Emit one LLM response from the active agent's LLM; optionally play it out."""
    llm = engine.active_agent.llm
    playback = engine.speech_playback
    start = LLMFullResponseStartFrame()
    await llm._call_event_handler("on_before_push_frame", start)
    if text is not None:
        await llm._call_event_handler("on_before_push_frame", LLMTextFrame(text))
    speech_id = start.metadata["dograh_speech_id"]
    if play:
        playback.after_output(None, SpeechBoundaryFrame(speech_id, True))
        playback.note_output()
        playback.after_output(None, SpeechBoundaryFrame(speech_id, False))
    return speech_id


async def call_end_transition(engine):
    transition = await engine._create_transition_func("move_to_end_call", "end")
    result = AsyncMock()
    await transition(SimpleNamespace(arguments={}, result_callback=result))
    properties = result.await_args.kwargs["properties"]
    return properties.run_llm, asyncio.create_task(properties.on_context_updated())


@pytest.fixture
def llm_engine(engine):
    llm = FrameProcessor()
    engine.active_agent.llm = llm
    engine.speech_playback.observe_responses(llm)
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize("playback", ["still_playing", "already_played"])
async def test_speech_in_the_transition_response_is_the_closing(llm_engine, playback):
    engine = llm_engine
    speech_id = await respond(
        engine, "You're welcome. Goodbye.", play=playback == "already_played"
    )

    run_llm, finishing = await call_end_transition(engine)
    try:
        # The end node must not generate a second goodbye after this one.
        assert run_llm is False
        if playback == "still_playing":
            await asyncio.sleep(0)
            # Hanging up now would cut the goodbye that is still being played.
            engine.call_worker.queue_frame.assert_not_awaited()
            engine.speech_playback.after_output(
                None, SpeechBoundaryFrame(speech_id, True)
            )
            engine.speech_playback.note_output()
            engine.speech_playback.after_output(
                None, SpeechBoundaryFrame(speech_id, False)
            )
        await asyncio.wait_for(finishing, timeout=1)
        engine.perform_final_variable_extraction.assert_awaited_once()
        terminal = engine.call_worker.queue_frame.await_args.args[0]
        assert isinstance(terminal, EndFrame)
        assert terminal.reason == EndTaskReason.END_CALL.value
        assert not engine.speech_playback.pending
    finally:
        if not finishing.done():
            finishing.cancel()
            await asyncio.gather(finishing, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier", ["none", "earlier_turn_spoke", "cut_off"])
async def test_end_node_still_closes_when_its_response_was_not_spoken(
    llm_engine, earlier
):
    engine = llm_engine
    if earlier == "earlier_turn_spoke":
        await respond(engine, "Anything else?", play=True)
    if earlier == "cut_off":
        # The caller interrupted the goodbye before the transition ran.
        await respond(engine, "Goodbye now.")
        engine.speech_playback.cancel_all(PlaybackOutcome.INTERRUPTED)
    else:
        await respond(engine, " ... ")  # this turn only called the transition

    run_llm, finishing = await call_end_transition(engine)
    try:
        assert run_llm is None
        await asyncio.sleep(0)
        assert not finishing.done()
        await respond(engine, "Thank you for calling. Goodbye.", play=True)
        await asyncio.wait_for(finishing, timeout=1)
        assert isinstance(engine.call_worker.queue_frame.await_args.args[0], EndFrame)
    finally:
        if not finishing.done():
            finishing.cancel()
            await asyncio.gather(finishing, return_exceptions=True)
