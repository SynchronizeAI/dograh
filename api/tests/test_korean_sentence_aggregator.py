import pytest
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from api.services.pipecat.korean_sentence_aggregator import KoreanSentenceAggregator
from api.services.pipecat.service_factory import _enable_korean_sentence_aggregation


@pytest.mark.asyncio
async def test_korean_terminal_punctuation_releases_sentence_without_lookahead():
    aggregator = KoreanSentenceAggregator(language="ko")

    assert [item.text async for item in aggregator.aggregate("예약이")] == []
    released = [item.text async for item in aggregator.aggregate(" 가능합니다.")]

    assert released == ["예약이 가능합니다."]


@pytest.mark.asyncio
async def test_non_korean_period_keeps_pipecat_lookahead_protection():
    aggregator = KoreanSentenceAggregator(language="ko")

    assert [item.text async for item in aggregator.aggregate("English.")] == []
    assert [item.text async for item in aggregator.aggregate(" ")] == []
    released = [item.text async for item in aggregator.aggregate(" Next sentence.")]

    assert released == ["English."]


def test_factory_only_replaces_sentence_aggregation_for_korean():
    korean = type("FakeTTS", (), {"_text_aggregator": SimpleTextAggregator()})()
    english = type("FakeTTS", (), {"_text_aggregator": SimpleTextAggregator()})()

    _enable_korean_sentence_aggregation(korean, "ko-KR")
    _enable_korean_sentence_aggregation(english, "en-US")

    assert isinstance(korean._text_aggregator, KoreanSentenceAggregator)
    assert type(english._text_aggregator) is SimpleTextAggregator
