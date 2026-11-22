"""Pre-call HTTP data fetch for StartCall node.

Executes an HTTP request before a voice call starts to enrich the
call context with data from external systems (CRM, ERP, etc.).
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from api.db import db_client
from api.errors.failure import (
    DograhFailure,
    ErrorSource,
    ErrorType,
    classify_exception,
    classify_http_response,
    log_failure,
    redact_failure_message,
)
from api.services.workflow.dto import ReactFlowDTO
from api.services.workflow.initial_context import merge_external_initial_context
from api.services.workflow.run_creation import published_definition
from api.services.workflow.workflow_graph import WorkflowGraph
from api.utils.credential_auth import build_auth_header

PRE_CALL_FETCH_TIMEOUT_SECONDS = 10

# Fetches the inbound webhook started, by run, until the call's pipeline takes
# them over. One whose stream never reaches this process is dropped.
_STARTED_FETCH_TTL_SECONDS = 60
_started_fetches: dict[int, "asyncio.Task[PreCallFetchResult]"] = {}


@dataclass(frozen=True)
class PreCallFetchResult:
    initial_context: dict = field(default_factory=dict)
    # The hook's `model_overrides` exactly as returned, or None when absent.
    # It is validated where it is applied, after the caller is connected, so
    # a malformed value is rejected there rather than failing the fetch.
    model_overrides: Any = None
    outcome: str = "unavailable"


def _extract_model_overrides(response_data: dict) -> Any:
    container = response_data.get("call_inbound")
    if not isinstance(container, dict):
        container = response_data
    return container.get("model_overrides")


def _extract_initial_context(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the context variables out of a pre-call fetch response.

    The canonical key is ``initial_context``. The legacy ``dynamic_variables``
    key is still accepted for backward compatibility, so existing endpoints
    keep working; ``initial_context`` takes precedence when both are present.

    Either key may appear at the top level or nested under ``call_inbound``:
        {"call_inbound": {"initial_context": {...}}} | {"initial_context": {...}}
        {"call_inbound": {"dynamic_variables": {...}}} | {"dynamic_variables": {...}}
    """
    container = response_data.get("call_inbound")
    if not isinstance(container, dict):
        container = response_data

    for key in ("initial_context", "dynamic_variables"):
        value = container.get(key)
        if isinstance(value, dict):
            # Pre-call fetch may enrich or override call-level context, including
            # the greeting. Only reserved run-owned metadata is discarded.
            return merge_external_initial_context({}, value)

    return {}


async def execute_pre_call_fetch(
    *,
    url: str,
    credential_uuid: Optional[str],
    call_context_vars: Dict[str, Any],
    workflow_id: int,
    organization_id: int,
) -> Dict[str, Any]:
    """Compatibility wrapper for callers that only consume context enrichment."""
    result = await execute_pre_call_fetch_result(
        url=url,
        credential_uuid=credential_uuid,
        call_context_vars=call_context_vars,
        workflow_id=workflow_id,
        organization_id=organization_id,
    )
    return result.initial_context


async def execute_pre_call_fetch_result(
    *,
    url: str,
    credential_uuid: Optional[str],
    call_context_vars: Dict[str, Any],
    workflow_id: int,
    organization_id: int,
    workflow_run_id: int | None = None,
) -> PreCallFetchResult:
    """Execute a POST request to fetch data before a call starts.

    Sends a standardized payload with call metadata (agent_id, from/to numbers).
    The response JSON is returned as a dict to be merged into initial_context.

    Returns:
        Context and the raw model-override envelope on success, an unavailable
        outcome on network failure.
    """
    # Build standardized payload
    payload = {
        "event": "call_inbound",
        "call_inbound": {
            "agent_id": workflow_id,
            "from_number": call_context_vars.get("caller_number", ""),
            "to_number": call_context_vars.get("called_number", ""),
        },
    }

    # Build headers
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if workflow_run_id is not None:
        headers["Idempotency-Key"] = f"dograh-pre-call-{workflow_run_id}"
        payload["call_inbound"]["workflow_run_id"] = workflow_run_id

    if credential_uuid:
        try:
            credential = await db_client.get_credential_by_uuid(
                credential_uuid, organization_id
            )
            if credential:
                headers.update(build_auth_header(credential))
            else:
                log_failure(
                    DograhFailure(
                        source=ErrorSource.INTEGRATION,
                        type=ErrorType.CONFIG_ERROR,
                        code="pre-call-fetch-credential-not-found",
                        internal_message="Pre-call fetch credential was not found",
                        external_message="The credential configured for pre-call fetch no longer exists.",
                        provider="pre-call-fetch",
                        error_owner="user",
                        retryable=False,
                    ),
                    organization_id=organization_id,
                    workflow_id=workflow_id,
                )
        except Exception as e:
            log_failure(
                classify_exception(
                    e,
                    source=ErrorSource.INTEGRATION,
                    provider="pre-call-fetch",
                    error_owner="user",
                ),
                organization_id=organization_id,
                workflow_id=workflow_id,
            )

    logger.info(f"Pre-call fetch: POST {redact_failure_message(url)}")

    try:
        async with httpx.AsyncClient(timeout=PRE_CALL_FETCH_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)

            try:
                response_data = response.json()
            except Exception:
                response_data = {}

            if response.is_success:
                if not isinstance(response_data, dict):
                    log_failure(
                        DograhFailure(
                            source=ErrorSource.INTEGRATION,
                            type=ErrorType.CONFIG_ERROR,
                            code="pre-call-fetch-invalid-response",
                            internal_message="Pre-call fetch response was not a JSON object",
                            external_message="The pre-call fetch endpoint returned an invalid response.",
                            provider="pre-call-fetch",
                            error_owner="user",
                            retryable=False,
                        ),
                        organization_id=organization_id,
                        workflow_id=workflow_id,
                    )
                    return PreCallFetchResult()

                # Extract the variables to merge into initial_context. Prefers
                # the canonical `initial_context` key, falling back to the
                # legacy `dynamic_variables` key for backward compatibility.
                initial_context_vars = _extract_initial_context(response_data)

                logger.info(
                    f"Pre-call fetch: success ({response.status_code}), "
                    f"initial_context keys: {list(initial_context_vars.keys())}"
                )
                return PreCallFetchResult(
                    initial_context=initial_context_vars,
                    model_overrides=_extract_model_overrides(response_data),
                    outcome="completed",
                )
            else:
                log_failure(
                    classify_http_response(
                        response.status_code,
                        f"Pre-call fetch returned HTTP {response.status_code}: "
                        f"{response.text[:200]}",
                        source=ErrorSource.INTEGRATION,
                        provider="pre-call-fetch",
                        error_owner="user",
                    ),
                    organization_id=organization_id,
                    workflow_id=workflow_id,
                )
                return PreCallFetchResult()

    except httpx.TimeoutException as e:
        log_failure(
            classify_exception(
                e,
                source=ErrorSource.INTEGRATION,
                provider="pre-call-fetch",
                error_owner="user",
            ),
            organization_id=organization_id,
            workflow_id=workflow_id,
        )
        return PreCallFetchResult()
    except httpx.RequestError as e:
        log_failure(
            classify_exception(
                e,
                source=ErrorSource.INTEGRATION,
                provider="pre-call-fetch",
                error_owner="user",
            ),
            organization_id=organization_id,
            workflow_id=workflow_id,
        )
        return PreCallFetchResult()
    except Exception as e:
        log_failure(
            classify_exception(
                e,
                source=ErrorSource.INTEGRATION,
                provider="pre-call-fetch",
                error_owner="user",
            ),
            organization_id=organization_id,
            workflow_id=workflow_id,
        )
        return PreCallFetchResult()


def start_pre_call_fetch(
    start_node, direction: str | None, *, workflow_run_id: int, **request
) -> "asyncio.Task[PreCallFetchResult] | None":
    """Take the run's fetch the inbound webhook started, else start one."""
    task = _started_fetches.pop(workflow_run_id, None)
    if (
        task is None
        and start_node
        and start_node.should_run_pre_call_fetch(direction)
        and start_node.pre_call_fetch_url
    ):
        logger.info(
            f"Pre-call fetch enabled for {direction or 'unknown'} workflow run "
            f"{workflow_run_id}, firing request to {start_node.pre_call_fetch_url}"
        )
        task = asyncio.create_task(
            execute_pre_call_fetch_result(
                url=start_node.pre_call_fetch_url,
                credential_uuid=start_node.pre_call_fetch_credential_uuid,
                workflow_run_id=workflow_run_id,
                **request,
            )
        )
    return task


def start_inbound_pre_call_fetch(workflow, workflow_run_id: int, webhook) -> None:
    """Start an inbound run's fetch while the carrier connects its media stream."""
    try:
        graph = WorkflowGraph(
            ReactFlowDTO.model_validate(published_definition(workflow).workflow_json),
            skip_instance_constraints_for={"trigger"},
        )
    except Exception:  # noqa: BLE001 - the pipeline builds the graph and reports it
        logger.warning(f"Pre-call fetch not started early for run {workflow_run_id}")
        return
    task = start_pre_call_fetch(
        graph.nodes.get(graph.start_node_id),
        "inbound",
        workflow_run_id=workflow_run_id,
        call_context_vars={
            "caller_number": webhook.from_number,
            "called_number": webhook.to_number,
        },
        workflow_id=workflow.id,
        organization_id=workflow.organization_id,
    )
    if task is not None:
        _started_fetches[workflow_run_id] = task
        asyncio.get_running_loop().call_later(
            _STARTED_FETCH_TTL_SECONDS, _started_fetches.pop, workflow_run_id, None
        )
