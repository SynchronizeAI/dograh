"""One agent visit's generation, reported back to the engine.

Rides with the generation stage rather than with the call, so it sits in the
agent worker in a cascade call and on the call pipeline in a realtime one.
Both callbacks are bound to the visit that owns this processor, so a retired
agent finishing its last generation cannot disturb the running one.
"""

from typing import Awaitable, Callable, Optional

from loguru import logger

from api.services.pipecat.korean_sentence_aggregator import (
    _CLOSING_DELIMITERS,
    KoreanSentenceAggregator,
)
from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    FunctionCallsFromLLMInfoFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.base_text_aggregator import TextType


def _is_question(sentence: str) -> bool:
    closing = " " + "".join(_CLOSING_DELIMITERS)
    return sentence.rstrip(closing).endswith(("?", "？"))


class AgentGenerationProcessor(FrameProcessor):
    """Tell the engine when this agent starts generating and what it says.

    With ``fixed_line_tools`` it also keeps a response that picks an edge with
    transition speech from asking a question of its own: the fixed line is the
    turn's question. Question sentences wait for the end of the response, and
    are dropped if the response chose such an edge. Everything after the first
    held question waits with it, so the speech keeps the order it was written in.
    Sentences go out as whole sentences: as text frames, English TTS would hold
    each one until the next sentence began.
    """

    def __init__(
        self,
        generation_started_callback: Optional[Callable[[], Awaitable[None]]] = None,
        llm_text_frame_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        fixed_line_tools: Optional[Callable[[], set[str]]] = None,
    ):
        super().__init__()
        self._generation_started_callback = generation_started_callback
        self._llm_text_frame_callback = llm_text_frame_callback
        self._fixed_line_tools = fixed_line_tools
        self._sentences = KoreanSentenceAggregator()
        self._lines: Optional[set[str]] = None
        self._picked = False
        self._held: list[str] = []
        self._sends_info = False
        self._info_call_ids: set[str] = set()
        self._pending_picked = False
        self._response_started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            preserve_snapshot = not self._response_started and self._lines is not None
            # A distinct started frame can overtake this response's start.
            defer_pending_pick = self._sends_info
            pending_picked = self._pending_picked and not defer_pending_pick
            await self._reset(
                preserve_snapshot=preserve_snapshot,
                preserve_pending_pick=defer_pending_pick,
            )
            self._picked |= pending_picked
            self._response_started = True
            self._snapshot()
            if self._generation_started_callback:
                await self._generation_started_callback()
        elif (
            isinstance(frame, (LLMTextFrame, TTSSpeakFrame))
            and self._llm_text_frame_callback
        ):
            # Static nodes speak through TTSSpeakFrame rather than the LLM, so
            # both carry reference text for correcting the aggregated transcript.
            await self._llm_text_frame_callback(frame.text)

        if isinstance(frame, LLMTextFrame) and self._lines and not frame.skip_tts:
            async for sentence in self._sentences.aggregate(frame.text):
                await self._release(sentence.text)
            return
        if isinstance(frame, FunctionCallsFromLLMInfoFrame):
            self._sends_info = True
            self._info_call_ids.update(
                call.tool_call_id for call in frame.function_calls
            )
            self._note_picks(frame)
        elif isinstance(frame, FunctionCallsStartedFrame):
            # Anthropic and Bedrock send only this one. Where the info frame
            # exists it comes first, and the started frame can arrive after the
            # response has ended (Google, OpenAI), too late to mean anything.
            call_ids = {call.tool_call_id for call in frame.function_calls}
            # A different ID is a new response, even if it arrives before this
            # response's end frame.
            is_next_response = self._sends_info and not call_ids.issubset(
                self._info_call_ids
            )
            self._note_picks(
                frame,
                self._info_call_ids,
                next_response=is_next_response,
            )
            self._info_call_ids.difference_update(call_ids)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._finish()
        elif isinstance(frame, InterruptionFrame):
            await self._reset()

        await self.push_frame(frame, direction)

    def _snapshot(self) -> set[str]:
        """The node's fixed-line tools, read once per response.

        System frames overtake the response's own frames, so the call can be
        seen before the response starts, and a tool that runs at once may have
        left the node by the time the start is seen. Whichever comes first
        reads the node.
        """
        if self._lines is None:
            self._lines = self._fixed_line_tools() if self._fixed_line_tools else set()
        return self._lines

    def _note_picks(
        self,
        frame: FunctionCallsFromLLMInfoFrame | FunctionCallsStartedFrame,
        ignored_call_ids: Optional[set[str]] = None,
        *,
        next_response: bool = False,
    ):
        ignored_call_ids = ignored_call_ids or set()
        calls = [
            call
            for call in frame.function_calls
            if call.tool_call_id not in ignored_call_ids
        ]
        if frame.function_calls and not calls:
            return
        lines = self._snapshot()
        picked = any(call.function_name in lines for call in calls)
        if next_response:
            self._pending_picked |= picked
        else:
            self._picked |= picked

    async def _reset(
        self, *, preserve_snapshot: bool = False, preserve_pending_pick: bool = False
    ):
        await self._sentences.reset()
        self._held.clear()
        self._sends_info = False
        if not preserve_pending_pick:
            self._pending_picked = False
        if not preserve_snapshot:
            self._lines = None
            self._picked = False
        self._response_started = False

    async def _release(self, sentence: str):
        if self._held or _is_question(sentence):
            self._held.append(sentence)
        else:
            await self._say(sentence)

    async def _finish(self):
        if tail := await self._sentences.flush():
            await self._release(tail.text)
        kept = [s for s in self._held if not (self._picked and _is_question(s))]
        if len(kept) < len(self._held):
            logger.info(
                "Fixed line follows; dropped {} model question(s)",
                len(self._held) - len(kept),
            )
        for sentence in kept:
            await self._say(sentence)
        await self._reset(preserve_pending_pick=True)

    async def _say(self, sentence: str):
        await self.push_frame(AggregatedTextFrame(sentence, TextType.SENTENCE))
