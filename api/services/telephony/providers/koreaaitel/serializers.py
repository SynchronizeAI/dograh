"""한국AI통신 frame serializer.

The vendor's Stream speaks Twilio Media Streams' media/clear/dtmf messages, so
pipecat's Twilio serializer is re-exported as is, as ``twilio/serializers.py``.
"""

from pipecat.serializers.twilio import TwilioFrameSerializer

__all__ = ["TwilioFrameSerializer"]
