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
from langdetect import detect, LangDetectException

from app.clients import STTClient
from app.schemas.transcribe_schema import (
    TranscriptionResponse,
    TranscriptionSegment,
)
from app.services.silerovad import SileroVADService
from app.services.noisereduce import SpectralNoiseReducer
import torch


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Thread pool scheduler for non-blocking transcription
_optimal_thread_count = max(1, multiprocessing.cpu_count() - 1)
_pool_scheduler = ThreadPoolScheduler(_optimal_thread_count)


class TranscriptionService:
    """Service for handling audio transcription using Philips AI STT API."""

    _instance: Optional["TranscriptionService"] = None
    DEFAULT_LANGUAGE = "en"

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

    def _filter_english_text(self, text: str) -> str:
        """
        Filter text to only include English content.

        Args:
            text: The transcribed text

        Returns:
            The text if it's detected as English, otherwise empty string
        """
        if not text.strip():
            return text

        try:
            detected_lang = detect(text)
            if detected_lang == "en":
                return text
            else:
                logger.info(
                    f"Detected non-English language '{detected_lang}', ignoring transcription: {text[:50]}..."
                )
                return ""
        except LangDetectException:
            # If language detection fails, assume it's not English and ignore
            logger.warning(
                f"Could not detect language for text, ignoring: {text[:50]}..."
            )
            return ""

    def transcribe_audio_array(
        self, audio: np.ndarray, language: Optional[str] = DEFAULT_LANGUAGE
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

        effective_language = language or self.DEFAULT_LANGUAGE

        # Ensure correct dtype
        audio = audio.astype(np.float32)

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio, self.sample_rate)

        try:
            # Call STT API
            result = self._client.transcribe(tmp_path, language=effective_language)

            # Parse response - API returns {"text": "..."}
            text = result.get("text", "").strip()

            # Filter to English only
            text = self._filter_english_text(text)

            # Build response (API doesn't return segments, so we create a single segment)
            duration = len(audio) / self.sample_rate
            segments = []
            if text:
                segments.append(
                    TranscriptionSegment(start=0.0, end=duration, text=text)
                )

            return TranscriptionResponse(
                text=text,
                language=effective_language,
                segments=segments,
                duration=duration,
            )
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def transcribe_bytes(
        self, audio_bytes: bytes, language: Optional[str] = DEFAULT_LANGUAGE
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
        self, file_path: str, language: Optional[str] = DEFAULT_LANGUAGE
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

        effective_language = language or self.DEFAULT_LANGUAGE

        # Call STT API directly with the file
        result = self._client.transcribe(file_path, language=effective_language)

        # Parse response
        text = result.get("text", "").strip()

        # Filter to English only
        text = self._filter_english_text(text)

        # Load audio to get duration
        audio, sample_rate = sf.read(file_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / sample_rate

        # Build response
        segments = []
        if text:
            segments.append(TranscriptionSegment(start=0.0, end=duration, text=text))

        return TranscriptionResponse(
            text=text, language=effective_language, segments=segments, duration=duration
        )


class StreamingBuffer:
    """
    Reactive buffer for handling streaming audio chunks using RxPY.

    This class uses reactive streams to handle audio data flow,
    providing non-blocking transcription with automatic buffering.
    """

    def __init__(
        self,
        transcription_service: Optional[TranscriptionService] = None,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        overlap_ms: int = 0,
        language: Optional[str] = TranscriptionService.DEFAULT_LANGUAGE,
        vad_enabled: bool = True,
        vad_threshold: float = 0.3,
        noise_reduce_enabled: bool = True,
        noise_reduce_strength: float = 0.5,
    ):
        """
        Initialize StreamingBuffer for real-time STT with 100 ms chunks.

        Args:
            transcription_service: STT service instance
            sample_rate: Audio sample rate in Hz (default 16000)
            chunk_duration_ms: Duration of each audio chunk in milliseconds (default 100)
            overlap_ms: Overlap between chunks in milliseconds (default 0)
            language: Language code for transcription
            vad_enabled: Enable or disable Voice Activity Detection
            vad_threshold: VAD probability threshold (0-1, default 0.3 = 30% confidence)
            noise_reduce_enabled: Enable or disable noise reduction
            noise_reduce_strength: Noise reduction strength (0-1, default 0.5 = moderate)
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_ms = overlap_ms
        self.language = language or TranscriptionService.DEFAULT_LANGUAGE
        self.vad_enabled = vad_enabled
        self.noise_reduce_enabled = noise_reduce_enabled
        self._service = transcription_service or TranscriptionService.get_instance()

        # Initialize Silero VAD
        self._vad = SileroVADService(threshold=vad_threshold) if vad_enabled else None
        self._noise_reducer = (
            SpectralNoiseReducer(
                sample_rate=sample_rate, prop_decrease=noise_reduce_strength
            )
            if noise_reduce_enabled
            else None
        )

        # Internal buffer for accumulating audio
        self._buffer = np.array([], dtype=np.float32)

        # Reactive subjects
        self._audio_input: Subject[bytes] = Subject()
        self._transcription_output: Subject[TranscriptionResponse] = Subject()
        self._error_output: Subject[Exception] = Subject()

        # Track subscriptions for cleanup
        self._disposables = CompositeDisposable()

        logger.info(
            f"StreamingBuffer initialized for real-time STT: "
            f"chunk_duration={chunk_duration_ms}ms ({self.samples_needed} samples), "
            f"sample_rate={sample_rate}Hz, overlap={overlap_ms}ms"
        )

        # Setup reactive pipeline
        self._setup_pipeline()

    @property
    def samples_needed(self) -> int:
        """Number of samples needed before transcription (100 ms chunks)."""
        return int((self.chunk_duration_ms / 1000.0) * self.sample_rate)

    @property
    def overlap_samples(self) -> int:
        """Number of samples to keep for overlap."""
        return int((self.overlap_ms / 1000.0) * self.sample_rate)

    @property
    def transcriptions(self) -> Subject[TranscriptionResponse]:
        """Observable stream of transcription results."""
        return self._transcription_output

    @property
    def errors(self) -> Subject[Exception]:
        """Observable stream of errors."""
        return self._error_output

    def _setup_pipeline(self) -> None:
        """Setup the reactive audio processing pipeline."""
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
            # Apply noise reduction on full chunks (more stable than per-packet)
            ops.map(self._apply_noise_reduction),
            # Apply VAD filtering if enabled
            ops.filter(self._check_vad if self.vad_enabled else lambda _: True),
            # Transcribe on background thread (non-blocking)
            ops.observe_on(_pool_scheduler),
            ops.map(self._transcribe_chunk),
            # Handle any errors gracefully
            ops.do_action(on_error=lambda e: self._error_output.on_next(e)),
        ).subscribe(
            on_next=lambda result: self._transcription_output.on_next(result),
            on_error=lambda e: self._error_output.on_next(e),
        )

        self._disposables.add(subscription)

    def _bytes_to_array(self, chunk: bytes) -> np.ndarray:
        """Convert raw bytes to float32 numpy array."""
        return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

    def _accumulate_buffer(self, audio: np.ndarray) -> None:
        """Add audio to the internal buffer."""
        self._buffer = np.append(self._buffer, audio)

    def _apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral noise reduction to audio chunks when enabled."""
        if not self.noise_reduce_enabled or self._noise_reducer is None:
            return audio
        try:
            enhanced = self._noise_reducer.denoise_array(audio)

            # Guardrail: if denoising suppresses too much signal energy, keep original
            # Use absolute threshold instead of ratio to avoid issues with quiet audio
            in_rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-12)
            out_rms = float(np.sqrt(np.mean(np.square(enhanced))) + 1e-12)

            # Only reject if BOTH conditions are true:
            # 1. Output is much weaker than input (< 30% signal retention)
            # 2. Input signal was actually strong enough to matter
            if in_rms > 0.01 and out_rms < 0.3 * in_rms:
                logger.warning(
                    f"Noise reduction too aggressive (in_rms={in_rms:.4f}, out_rms={out_rms:.4f}); using raw audio"
                )
                return audio

            return enhanced
        except Exception as e:
            logger.warning(f"Noise reduction failed, using raw audio: {e}")
            return audio

    def _is_buffer_ready(self) -> bool:
        """Check if buffer has enough samples for transcription."""
        return len(self._buffer) >= self.samples_needed

    def _extract_chunk(self) -> Optional[np.ndarray]:
        """Extract audio chunk and maintain overlap."""
        if len(self._buffer) < self.samples_needed:
            return None

        audio = self._buffer[: self.samples_needed].copy()
        # Keep overlap for context continuity
        self._buffer = self._buffer[self.samples_needed - self.overlap_samples :]
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
        on_error: Optional[Callable[[Exception], None]] = None,
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
            on_error=on_error or (lambda e: logger.error(f"Transcription error: {e}")),
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
        audio = self._buffer[: self.samples_needed].copy()
        self._buffer = self._buffer[self.samples_needed - self.overlap_samples :]
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
        self._disposables.dispose()
        self.clear()

    def _check_vad(self, audio: np.ndarray) -> bool:
        """
        Check if the audio contains speech using VAD.

        Args:
            audio: Audio data as numpy array (float32, mono).

        Returns:
            True if speech is detected, False otherwise.
        """

        if self._vad is None:
            return True  # If VAD is disabled, allow all audio
        audio_tensor = torch.tensor(audio, dtype=torch.float32).to(self._vad.device)
        return self._vad.is_speech(audio_tensor, self.sample_rate)
