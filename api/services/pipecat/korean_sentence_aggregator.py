"""Preserve Korean punctuation clusters with minimal boundary lookahead."""

from pipecat.utils.text.base_text_aggregator import Aggregation, TextType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

_TERMINAL_PUNCTUATION = frozenset(".!?。？！")
_CLUSTER_PUNCTUATION = frozenset(".,!?;:…。？！；：")
_CLOSING_DELIMITERS = frozenset("\"'”’»›)]}）］｝」』》〉】〕〗〙〛")


class KoreanSentenceAggregator(SimpleTextAggregator):
    async def _check_sentence_with_lookahead(
        self, char: str, *, protected_end: int = 0
    ):
        if _pending_korean_boundary(self._text) is not None:
            return None
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
    while end and text[end - 1] in _CLUSTER_PUNCTUATION:
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
