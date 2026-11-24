"""Usage units exposed by voice providers outside Pipecat's token metrics."""

from api.services.pipecat.fallback_llm import FallbackLLMProcessor
from pipecat.metrics.metrics import MetricsData
from pipecat.services.llm_service import LLMService


def on_inference_usage(llm: LLMService, handler) -> None:
    """Call ``handler(service, LLMUsageMetricsData)`` for each run_inference reply.

    Missing OSS capability: upstream Pipecat's run_inference drops provider
    usage; the fork reports it through each service's ``on_inference_usage``,
    so a fallback chain counts usage under the target that answered.
    """
    for service in llm.processors if isinstance(llm, FallbackLLMProcessor) else [llm]:
        if isinstance(service, LLMService):
            service.add_event_handler("on_inference_usage", handler)


def merge_llm_usage(existing: dict | None, delta: dict) -> dict:
    """Add provider token counts without losing optional or unreported fields."""
    merged = dict(existing or {})
    for key, value in delta.items():
        current = dict(merged.get(key) or {})
        for field, count in value.items():
            if count is not None:
                current[field] = (current.get(field) or 0) + count
            else:
                current.setdefault(field, None)
        merged[key] = current
    return merged


class LiveUsageMetricsData(MetricsData):
    """Additional seconds reported by a duration-billed Live session."""

    seconds: float
