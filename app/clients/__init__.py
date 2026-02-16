"""
Philips AI Model Serving API Clients.
"""

from .stt_client import (
    STTClient,
    STTClientError,
    STTAuthenticationError,
    STTTranscriptionError,
    STTConfigurationError,
)

__all__ = [
    "STTClient",
    "STTClientError",
    "STTAuthenticationError",
    "STTTranscriptionError",
    "STTConfigurationError",
]
