import pytest
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from api.services.pipecat.korean_sentence_aggregator import KoreanSentenceAggregator
from api.services.pipecat.service_factory import _enable_korean_sentence_aggregation


@pytest.mark.asyncio
async def test_plain_korean_terminal_releases_immediately():
    aggregator = KoreanSentenceAggregator(language="ko")

    released = [item.text async for item in aggregator.aggregate("예약했어요.")]

    assert released == ["예약했어요."]


@pytest.mark.parametrize(
    "chunk, expected",
    [
        ("가능해요... 다음", "가능해요..."),
        ("맞나요?! 네", "맞나요?!"),
    ],
)
@pytest.mark.asyncio
async def test_punctuation_clusters_stay_intact_within_one_chunk(chunk, expected):
    aggregator = KoreanSentenceAggregator(language="ko")

    released = [item.text async for item in aggregator.aggregate(chunk)]

    assert released == [expected]


@pytest.mark.parametrize("closing", ['"', ")", "]"])
@pytest.mark.asyncio
async def test_closing_delimiters_stay_with_sentence_until_cluster_ends(closing):
    aggregator = KoreanSentenceAggregator(language="ko")

    released = [
        item.text async for item in aggregator.aggregate(f"예약했어요.{closing} 네")
    ]

    assert released == [f"예약했어요.{closing}"]


@pytest.mark.asyncio
async def test_incomplete_two_mark_suffix_waits_for_cluster_completion():
    aggregator = KoreanSentenceAggregator(language="ko")

    assert [item.text async for item in aggregator.aggregate("가능해요..")] == []
    released = [item.text async for item in aggregator.aggregate(". 다음")]

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
