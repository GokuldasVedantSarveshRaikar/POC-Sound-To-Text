import whisper
import numpy as np
from typing import Optional, Callable
from pathlib import Path
import soundfile as sf
from app.schemas.transcribe_schema import (
    TranscriptionResponse,
    TranscriptionSegment,
    ModelSize
)


class TranscriptionService:
    """Service for handling audio transcription using Whisper."""
    
    _instance: Optional["TranscriptionService"] = None
    
    def __init__(self, model_size: ModelSize = ModelSize.BASE):
        self.model_size = model_size
        self.model = None
        self.sample_rate = 16000
        
    @classmethod
    def get_instance(cls, model_size: ModelSize = ModelSize.BASE) -> "TranscriptionService":
        """Get singleton instance of the service."""
        if cls._instance is None:
            cls._instance = cls(model_size)
        return cls._instance
    
    def load_model(self) -> None:
        """Load the Whisper model."""
        if self.model is None:
            print(f"Loading Whisper model '{self.model_size.value}'...")
            self.model = whisper.load_model(self.model_size.value)
            print("Model loaded successfully!")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def unload_model(self) -> None:
        """Unload the Whisper model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            print("Model unloaded.")
    
    def transcribe_audio_array(
        self,
        audio: np.ndarray,
        language: Optional[str] = None
    ) -> TranscriptionResponse:
        """
        Transcribe audio from numpy array.
        
        Args:
            audio: Audio data as numpy array (float32, 16kHz, mono)
            language: Language code or None for auto-detect
        
        Returns:
            TranscriptionResponse with text and segments
        """
        if not self.is_model_loaded():
            self.load_model()
        
        # Ensure correct dtype
        audio = audio.astype(np.float32)
        
        # Transcribe
        options = {"fp16": False}
        if language:
            options["language"] = language
            
        result = self.model.transcribe(audio, **options)
        
        # Build segments
        segments = [
            TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip()
            )
            for seg in result.get("segments", [])
        ]
        
        return TranscriptionResponse(
            text=result["text"].strip(),
            language=result.get("language", "unknown"),
            segments=segments,
            duration=len(audio) / self.sample_rate
        )
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None
    ) -> TranscriptionResponse:
        """
        Transcribe audio from raw bytes (16-bit PCM, 16kHz, mono).
        
        Args:
            audio_bytes: Raw audio bytes
            language: Language code or None for auto-detect
        
        Returns:
            TranscriptionResponse with text and segments
        """
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.transcribe_audio_array(audio, language)
    
    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResponse:
        """
        Transcribe audio from file.
        
        Args:
            file_path: Path to audio file (WAV, FLAC, OGG)
            language: Language code or None for auto-detect
        
        Returns:
            TranscriptionResponse with text and segments
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        audio = self._load_audio_file(file_path)
        return self.transcribe_audio_array(audio, language)
    
    def _load_audio_file(self, audio_path: str) -> np.ndarray:
        """Load audio file without FFmpeg."""
        audio, sample_rate = sf.read(audio_path, dtype='float32')
        
        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != self.sample_rate:
            duration = len(audio) / sample_rate
            target_length = int(duration * self.sample_rate)
            audio = np.interp(
                np.linspace(0, len(audio), target_length),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)
        
        return audio


class StreamingBuffer:
    """Buffer for handling streaming audio chunks."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_seconds: float = 3.0,
        overlap_seconds: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.overlap_seconds = overlap_seconds
        self.buffer = np.array([], dtype=np.float32)
        
    @property
    def samples_needed(self) -> int:
        """Number of samples needed before transcription."""
        return int(self.sample_rate * self.buffer_seconds)
    
    @property
    def overlap_samples(self) -> int:
        """Number of samples to keep for overlap."""
        return int(self.sample_rate * self.overlap_seconds)
    
    def add_chunk(self, chunk: bytes) -> None:
        """Add audio chunk (16-bit PCM bytes) to buffer."""
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.append(self.buffer, audio)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough audio for transcription."""
        return len(self.buffer) >= self.samples_needed
    
    def get_audio(self) -> np.ndarray:
        """Get audio for transcription and keep overlap."""
        audio = self.buffer[:self.samples_needed].copy()
        # Keep overlap for context continuity
        self.buffer = self.buffer[self.samples_needed - self.overlap_samples:]
        return audio
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = np.array([], dtype=np.float32)
    
    def get_remaining(self) -> Optional[np.ndarray]:
        """Get any remaining audio in buffer."""
        if len(self.buffer) > self.overlap_samples:
            audio = self.buffer.copy()
            self.clear()
            return audio
        return None