import pytest
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from api.services.pipecat.korean_sentence_aggregator import KoreanSentenceAggregator
from api.services.pipecat.service_factory import _enable_korean_sentence_aggregation


@pytest.mark.asyncio
async def test_plain_korean_terminal_releases_immediately():
    aggregator = KoreanSentenceAggregator(language="ko")

    released = [item.text async for item in aggregator.aggregate("예약했어요.")]

    assert released == ["예약했어요."]


@pytest.mark.asyncio
async def test_ellipsis_releases_as_a_single_sentence_boundary():
    aggregator = KoreanSentenceAggregator(language="ko")

    released = [item.text async for item in aggregator.aggregate("가능해요... 다음")]

    assert released == ["가능해요..."]


@pytest.mark.asyncio
async def test_same_chunk_closing_quote_stays_with_sentence():
    aggregator = KoreanSentenceAggregator(language="ko")

    released = [item.text async for item in aggregator.aggregate('예약했어요." 네')]

    assert released == ['예약했어요."']


@pytest.mark.asyncio
async def test_mixed_marks_and_closer_release_at_buffer_end():
    aggregator = KoreanSentenceAggregator(language="ko")

    released = [item.text async for item in aggregator.aggregate("예약했어요?!”")]

    assert released == ["예약했어요?!”"]


@pytest.mark.asyncio
async def test_exactly_two_trailing_dots_wait_for_ellipsis_completion():
    aggregator = KoreanSentenceAggregator(language="ko")

    assert [item.text async for item in aggregator.aggregate("가능해요..")] == []
    released = [item.text async for item in aggregator.aggregate(".")]

    assert released == ["가능해요..."]


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
