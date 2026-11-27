"""한국AI통신 telephony configuration schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class KoreaAITelConfigurationRequest(BaseModel):
    """Request schema for 한국AI통신 configuration."""

    provider: Literal["koreaaitel"] = Field(default="koreaaitel")
    account_id: str = Field(..., description="한국AI통신 account ID")
    signing_secret: str = Field(..., description="Webhook Signing Secret")
    api_key: str | None = None  # needed to place calls, not to receive them
