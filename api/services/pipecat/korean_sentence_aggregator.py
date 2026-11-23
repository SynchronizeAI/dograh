"""Release complete Korean sentences on terminal punctuation without lookahead."""

from pipecat.utils.text.base_text_aggregator import Aggregation, TextType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class KoreanSentenceAggregator(SimpleTextAggregator):
    async def _check_sentence_with_lookahead(
        self, char: str, *, protected_end: int = 0
    ):
        if char in ".!?。？！" and len(self._text) > 1 and _is_hangul(self._text[-2]):
            sentence = self._text.strip()
            await self.reset()
            return Aggregation(text=sentence, type=TextType.SENTENCE)
        return await super()._check_sentence_with_lookahead(
            char, protected_end=protected_end
        )


def _is_hangul(char: str) -> bool:
    return "\uac00" <= char <= "\ud7a3"
