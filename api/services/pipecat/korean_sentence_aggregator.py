"""Preserve Korean punctuation clusters with minimal boundary lookahead."""

from pipecat.utils.text.base_text_aggregator import Aggregation, TextType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

_TERMINAL_PUNCTUATION = frozenset(".!?。？！…")
_CLOSING_DELIMITERS = frozenset("\"'”’»›)]}）］｝」』》〉】〕〗〙〛")
_CLUSTER_CONTINUATION = _TERMINAL_PUNCTUATION | _CLOSING_DELIMITERS


class KoreanSentenceAggregator(SimpleTextAggregator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chunk_tail = ""

    async def aggregate(self, text: str):
        if self._aggregation_type == TextType.TOKEN:
            async for aggregation in super().aggregate(text):
                yield aggregation
            return

        try:
            for index, char in enumerate(text):
                self._text += char
                self._chunk_tail = text[index + 1 :]
                result = await self._check_sentence_with_lookahead(char)
                if result:
                    yield result
        finally:
            self._chunk_tail = ""

    async def _check_sentence_with_lookahead(
        self, char: str, *, protected_end: int = 0
    ):
        boundary_start = _pending_korean_boundary(self._text)
        if boundary_start is not None:
            if (
                self._chunk_tail
                and self._chunk_tail[0] in _CLUSTER_CONTINUATION
            ) or (
                self._text.endswith("..") and not self._text.endswith("...")
            ):
                return None
            sentence = self._text.strip()
            await self.reset()
            return Aggregation(text=sentence, type=TextType.SENTENCE)
        if _pending_korean_boundary(self._text[:-1]) is not None:
            sentence = self._text[:-1].strip()
            remainder = self._text[-1:]
            await self.reset()
            self._text = remainder
            return Aggregation(text=sentence, type=TextType.SENTENCE)
        return await super()._check_sentence_with_lookahead(
            char, protected_end=protected_end
        )


def _pending_korean_boundary(text: str) -> int | None:
    end = len(text)
    while end and text[end - 1] in _CLOSING_DELIMITERS:
        end -= 1
    punctuation_end = end
    while end and text[end - 1] in _TERMINAL_PUNCTUATION:
        end -= 1
    punctuation = text[end:punctuation_end]
    if (
        end == punctuation_end
        or not any(char in _TERMINAL_PUNCTUATION for char in punctuation)
        or end == 0
        or not _is_hangul(text[end - 1])
    ):
        return None
    return end


def _is_hangul(char: str) -> bool:
    return "\uac00" <= char <= "\ud7a3"
