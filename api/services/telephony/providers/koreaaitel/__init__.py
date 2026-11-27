"""한국AI통신 (Korea AI Telecom) telephony provider package.

Missing OSS capability: Dograh has no provider for 한국AI통신's VoiceML webhook
and Stream, which is Twilio-shaped but with other field names, an HMAC-SHA256
signature and a Bearer-key calls API. Inbound: the number's VoiceML webhook
points at ``/inbound/run``. Outbound: ``POST /calls``, ``/twiml`` and ``routes.py``.
"""

from typing import Any, Dict

from api.services.telephony.registry import (
    ProviderSpec,
    ProviderUIField,
    ProviderUIMetadata,
    register,
)

from .config import KoreaAITelConfigurationRequest
from .provider import KoreaAITelProvider
from .transport import create_transport


def _config_loader(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "provider": "koreaaitel",
        "account_id": value.get("account_id"),
        "signing_secret": value.get("signing_secret"),
        "api_key": value.get("api_key"),
    }


_UI_METADATA = ProviderUIMetadata(
    display_name="한국AI통신",
    docs_url="https://platform.koreaaitel.com/docs/build/voiceml",
    fields=[
        ProviderUIField(
            name="account_id",
            label="Account ID",
            type="text",
            sensitive=True,
            description="한국AI통신 account ID (starts with AC)",
        ),
        ProviderUIField(
            name="signing_secret",
            label="Webhook Signing Secret",
            type="password",
            sensitive=True,
            description="Dashboard > 개발자 > 웹훅 > Webhook Signing Secret",
        ),
        ProviderUIField(
            name="api_key",
            label="API Key",
            type="password",
            required=False,
            sensitive=True,
        ),
    ],
)


SPEC = ProviderSpec(
    name="koreaaitel",
    provider_cls=KoreaAITelProvider,
    config_loader=_config_loader,
    transport_factory=create_transport,
    transport_sample_rate=8000,
    config_request_cls=KoreaAITelConfigurationRequest,
    ui_metadata=_UI_METADATA,
    account_id_credential_field="account_id",
)


register(SPEC)
