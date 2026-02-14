from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ModelSize(str, Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class TranscriptionSegment(BaseModel):
    """A single transcription segment with timestamps."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")


class TranscriptionRequest(BaseModel):
    """Request schema for file-based transcription."""
    audio_path: str = Field(..., description="Path to the audio file")
    model_size: ModelSize = Field(default=ModelSize.BASE, description="Whisper model size")
    language: Optional[str] = Field(default=None, description="Language code (e.g., 'en', 'es'). Auto-detect if None")


class TranscriptionResponse(BaseModel):
    """Response schema for transcription results."""
    text: str = Field(..., description="Full transcribed text")
    language: str = Field(..., description="Detected or specified language")
    segments: List[TranscriptionSegment] = Field(default=[], description="Segments with timestamps")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")


# WebSocket schemas

class WebSocketMessage(BaseModel):
    """Base WebSocket message."""
    type: str = Field(..., description="Message type")


class WebSocketTranscription(WebSocketMessage):
    """WebSocket transcription result."""
    type: str = Field(default="transcription")
    text: str = Field(..., description="Transcribed text")
    is_final: bool = Field(default=True, description="Whether this is a final result")


class WebSocketError(WebSocketMessage):
    """WebSocket error message."""
    type: str = Field(default="error")
    message: str = Field(..., description="Error message")


class WebSocketStatus(WebSocketMessage):
    """WebSocket status message."""
    type: str = Field(default="status")
    status: str = Field(..., description="Connection status")
    message: Optional[str] = Field(default=None, description="Status message")


# Health check

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok")
    model_loaded: bool = Field(..., description="Whether Whisper model is loaded")
    model_size: str = Field(..., description="Loaded model size")