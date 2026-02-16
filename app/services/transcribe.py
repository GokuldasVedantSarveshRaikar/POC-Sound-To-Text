"""
Transcription service using Philips AI Model Serving STT API.

This module provides transcription services using the remote STT API
with automatic token management and caching via STTClient.
"""

import numpy as np
import tempfile
import os
import logging
import multiprocessing
from typing import Optional, Callable
from pathlib import Path
import soundfile as sf
from dotenv import load_dotenv
from reactivex import operators as ops
from reactivex.subject import Subject
from reactivex.scheduler import ThreadPoolScheduler
from reactivex.disposable import CompositeDisposable

from app.clients import STTClient, STTClientError
from app.schemas.transcribe_schema import (
    TranscriptionResponse,
    TranscriptionSegment,
)
from app.services.vad import VoiceActivityDetector

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Thread pool scheduler for non-blocking transcription
_optimal_thread_count = max(1, multiprocessing.cpu_count() - 1)
_pool_scheduler = ThreadPoolScheduler(_optimal_thread_count)


class TranscriptionService:
    """Service for handling audio transcription using Philips AI STT API."""
    
    _instance: Optional["TranscriptionService"] = None
    
    def __init__(self):
        self.sample_rate = 16000
        self._client: Optional[STTClient] = None
        self._initialized = False
        
    @classmethod
    def get_instance(cls) -> "TranscriptionService":
        """Get singleton instance of the service."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_model(self) -> None:
        """Initialize the STT client (replaces load_model for API compatibility)."""
        if self._client is None:
            logger.info("Initializing STT Client...")
            self._client = STTClient()
            self._initialized = True
            logger.info("STT Client initialized successfully!")
    
    def is_model_loaded(self) -> bool:
        """Check if client is initialized."""
        return self._initialized and self._client is not None
    
    def unload_model(self) -> None:
        """Close the STT client and release resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._initialized = False
            logger.info("STT Client closed.")
    
    def health_check(self) -> bool:
        """Check if the service can authenticate successfully."""
        if self._client is None:
            self.load_model()
        return self._client.health_check()
    
    def transcribe_audio_array(
        self,
        audio: np.ndarray,
        language: Optional[str] = None
    ) -> TranscriptionResponse:
        """
        Transcribe audio from numpy array by saving to temp file and calling API.
        
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
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio, self.sample_rate)
        
        try:
            # Call STT API
            result = self._client.transcribe(tmp_path, language=language)
            
            # Parse response - API returns {"text": "..."}
            text = result.get("text", "").strip()
            
            # Build response (API doesn't return segments, so we create a single segment)
            duration = len(audio) / self.sample_rate
            segments = []
            if text:
                segments.append(
                    TranscriptionSegment(
                        start=0.0,
                        end=duration,
                        text=text
                    )
                )
            
            return TranscriptionResponse(
                text=text,
                language=language or "auto",
                segments=segments,
                duration=duration
            )
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
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
            file_path: Path to audio file (WAV, MP3, FLAC, etc.)
            language: Language code or None for auto-detect
        
        Returns:
            TranscriptionResponse with text and segments
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if not self.is_model_loaded():
            self.load_model()
        
        # Call STT API directly with the file
        result = self._client.transcribe(file_path, language=language)
        
        # Parse response
        text = result.get("text", "").strip()
        
        # Load audio to get duration
        audio, sample_rate = sf.read(file_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / sample_rate
        
        # Build response
        segments = []
        if text:
            segments.append(
                TranscriptionSegment(
                    start=0.0,
                    end=duration,
                    text=text
                )
            )
        
        return TranscriptionResponse(
            text=text,
            language=language or "auto",
            segments=segments,
            duration=duration
        )


class StreamingBuffer:
    """
    Reactive buffer for handling streaming audio chunks using RxPY.
    
    This class uses reactive streams to handle audio data flow,
    providing non-blocking transcription with automatic buffering.
    Includes Voice Activity Detection (VAD) to reduce unnecessary API calls.
    """
    
    def __init__(
        self,
        transcription_service: Optional[TranscriptionService] = None,
        sample_rate: int = 16000,
        buffer_seconds: float = 3.0,
        overlap_seconds: float = 0.5,
        language: Optional[str] = None,
        vad_enabled: bool = True,
        vad_energy_threshold: float = 0.005,  # Lower threshold to catch quieter speech
        vad_speech_ratio: float = 0.1  # Only 10% of frames need speech (more permissive)
    ):
        """
        Initialize StreamingBuffer.
        
        Args:
            transcription_service: Service for transcription (default: singleton)
            sample_rate: Audio sample rate (default: 16000 Hz)
            buffer_seconds: Seconds of audio to buffer before transcription
            overlap_seconds: Seconds of overlap between chunks
            language: Language code for transcription
            vad_enabled: Enable Voice Activity Detection to skip silence
            vad_energy_threshold: VAD energy threshold (0.0 to 1.0)
            vad_speech_ratio: Minimum ratio of frames with speech (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.overlap_seconds = overlap_seconds
        self.language = language
        self._service = transcription_service or TranscriptionService.get_instance()
        
        # Voice Activity Detection
        self.vad_enabled = vad_enabled
        self._vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            energy_threshold=vad_energy_threshold,
            speech_ratio_threshold=vad_speech_ratio,
            adaptive=True
        ) if vad_enabled else None
        
        # Statistics for monitoring
        self._chunks_processed = 0
        self._chunks_skipped_vad = 0
        
        # Internal buffer for accumulating audio
        self._buffer = np.array([], dtype=np.float32)
        
        # Reactive subjects
        self._audio_input: Subject[bytes] = Subject()
        self._transcription_output: Subject[TranscriptionResponse] = Subject()
        self._error_output: Subject[Exception] = Subject()
        
        # Track subscriptions for cleanup
        self._disposables = CompositeDisposable()
        
        # Setup reactive pipeline
        self._setup_pipeline()
        
    @property
    def samples_needed(self) -> int:
        """Number of samples needed before transcription."""
        return int(self.sample_rate * self.buffer_seconds)
    
    @property
    def overlap_samples(self) -> int:
        """Number of samples to keep for overlap."""
        return int(self.sample_rate * self.overlap_seconds)
    
    @property
    def transcriptions(self) -> Subject[TranscriptionResponse]:
        """Observable stream of transcription results."""
        return self._transcription_output
    
    @property
    def errors(self) -> Subject[Exception]:
        """Observable stream of errors."""
        return self._error_output
    
    def _setup_pipeline(self) -> None:
        """Setup the reactive audio processing pipeline with VAD."""
        subscription = self._audio_input.pipe(
            # Convert bytes to numpy array
            ops.map(self._bytes_to_array),
            # Accumulate into buffer
            ops.do_action(self._accumulate_buffer),
            # Only proceed when buffer is ready
            ops.filter(lambda _: self._is_buffer_ready()),
            # Extract audio chunk for transcription
            ops.map(lambda _: self._extract_chunk()),
            # Filter out empty chunks
            ops.filter(lambda audio: audio is not None and len(audio) > 0),
            # VAD: Only transcribe if speech is detected
            ops.filter(self._check_vad),
            # Transcribe on background thread (non-blocking)
            ops.observe_on(_pool_scheduler),
            ops.map(self._transcribe_chunk),
            # Handle any errors gracefully
            ops.do_action(
                on_error=lambda e: self._error_output.on_next(e)
            )
        ).subscribe(
            on_next=lambda result: self._transcription_output.on_next(result),
            on_error=lambda e: self._error_output.on_next(e)
        )
        
        self._disposables.add(subscription)
    
    def _check_vad(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech using VAD."""
        self._chunks_processed += 1
        
        if not self.vad_enabled or self._vad is None:
            return True
        
        has_speech = self._vad.is_speech(audio)
        
        if not has_speech:
            self._chunks_skipped_vad += 1
            logger.debug(
                f"VAD: Skipping chunk (silence). "
                f"Skipped: {self._chunks_skipped_vad}/{self._chunks_processed} "
                f"({100 * self._chunks_skipped_vad / self._chunks_processed:.1f}%)"
            )
        
        return has_speech
    
    def get_vad_stats(self) -> dict:
        """Get VAD statistics."""
        return {
            "chunks_processed": self._chunks_processed,
            "chunks_skipped": self._chunks_skipped_vad,
            "skip_ratio": self._chunks_skipped_vad / self._chunks_processed 
                if self._chunks_processed > 0 else 0,
            "api_calls_saved_percent": 100 * self._chunks_skipped_vad / self._chunks_processed 
                if self._chunks_processed > 0 else 0
        }
    
    def _bytes_to_array(self, chunk: bytes) -> np.ndarray:
        """Convert raw bytes to float32 numpy array."""
        return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    
    def _accumulate_buffer(self, audio: np.ndarray) -> None:
        """Add audio to the internal buffer."""
        self._buffer = np.append(self._buffer, audio)
    
    def _is_buffer_ready(self) -> bool:
        """Check if buffer has enough samples for transcription."""
        return len(self._buffer) >= self.samples_needed
    
    def _extract_chunk(self) -> Optional[np.ndarray]:
        """Extract audio chunk and maintain overlap."""
        if len(self._buffer) < self.samples_needed:
            return None
        
        audio = self._buffer[:self.samples_needed].copy()
        # Keep overlap for context continuity
        self._buffer = self._buffer[self.samples_needed - self.overlap_samples:]
        return audio
    
    def _transcribe_chunk(self, audio: np.ndarray) -> TranscriptionResponse:
        """Transcribe audio chunk using the transcription service."""
        return self._service.transcribe_audio_array(audio, self.language)
    
    def add_chunk(self, chunk: bytes) -> None:
        """
        Add audio chunk to the reactive stream.
        
        Args:
            chunk: Raw audio bytes (16-bit PCM, 16kHz, mono)
        """
        self._audio_input.on_next(chunk)
    
    def subscribe(
        self,
        on_transcription: Callable[[TranscriptionResponse], None],
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """
        Subscribe to transcription results.
        
        Args:
            on_transcription: Callback for transcription results
            on_error: Optional callback for errors
        
        Returns:
            Disposable subscription
        """
        sub = self._transcription_output.subscribe(
            on_next=on_transcription,
            on_error=on_error or (lambda e: logger.error(f"Transcription error: {e}"))
        )
        self._disposables.add(sub)
        
        if on_error:
            error_sub = self._error_output.subscribe(on_next=on_error)
            self._disposables.add(error_sub)
        
        return sub
    
    def is_ready(self) -> bool:
        """Check if buffer has enough audio for transcription."""
        return len(self._buffer) >= self.samples_needed
    
    def get_audio(self) -> np.ndarray:
        """Get audio for transcription and keep overlap (legacy compatibility)."""
        audio = self._buffer[:self.samples_needed].copy()
        self._buffer = self._buffer[self.samples_needed - self.overlap_samples:]
        return audio
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer = np.array([], dtype=np.float32)
    
    def get_remaining(self) -> Optional[np.ndarray]:
        """Get any remaining audio in buffer."""
        if len(self._buffer) > self.overlap_samples:
            audio = self._buffer.copy()
            self.clear()
            return audio
        return None
    
    def flush(self) -> None:
        """
        Flush remaining audio in buffer and transcribe it.
        Call this when the stream ends to process any remaining audio.
        """
        remaining = self.get_remaining()
        if remaining is not None and len(remaining) > 0:
            try:
                result = self._service.transcribe_audio_array(remaining, self.language)
                self._transcription_output.on_next(result)
            except Exception as e:
                self._error_output.on_next(e)
    
    def complete(self) -> None:
        """
        Signal end of stream, flush remaining audio, and complete subjects.
        """
        self.flush()
        self._audio_input.on_completed()
        self._transcription_output.on_completed()
        self._error_output.on_completed()
    
    def dispose(self) -> None:
        """Clean up all subscriptions and resources."""
        # Log VAD statistics
        if self.vad_enabled and self._chunks_processed > 0:
            stats = self.get_vad_stats()
            logger.info(
                f"VAD Stats: {stats['chunks_skipped']}/{stats['chunks_processed']} "
                f"chunks skipped ({stats['api_calls_saved_percent']:.1f}% API calls saved)"
            )
        
        self._disposables.dispose()
        self.clear()
        
        # Reset VAD if present
        if self._vad:
            self._vad.reset()
        self._chunks_processed = 0
        self._chunks_skipped_vad = 0