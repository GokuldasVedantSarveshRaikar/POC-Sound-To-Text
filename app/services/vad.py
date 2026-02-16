"""
Voice Activity Detection (VAD) service.

Provides energy-based speech detection to reduce unnecessary API calls
during silence periods in audio streams.
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Energy-based Voice Activity Detection (VAD).
    
    Detects speech presence based on audio energy levels.
    Simple but effective for reducing API calls during silence.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01,
        speech_ratio_threshold: float = 0.3,
        adaptive: bool = True,
        adaptive_factor: float = 0.95
    ):
        """
        Initialize VAD.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration for analysis (10, 20, or 30 ms)
            energy_threshold: Initial energy threshold (0.0 to 1.0)
            speech_ratio_threshold: Ratio of frames that must have speech (0.0 to 1.0)
            adaptive: Whether to adapt threshold based on noise floor
            adaptive_factor: How quickly to adapt (0.9 to 0.99, higher = slower)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.energy_threshold = energy_threshold
        self.speech_ratio_threshold = speech_ratio_threshold
        self.adaptive = adaptive
        self.adaptive_factor = adaptive_factor
        
        # Adaptive noise floor tracking
        self._noise_floor = energy_threshold * 0.5
        self._is_initialized = False
        
    def _calculate_rms_energy(self, frame: np.ndarray) -> float:
        """Calculate RMS energy of a frame."""
        return np.sqrt(np.mean(frame ** 2))
    
    def _calculate_zero_crossing_rate(self, frame: np.ndarray) -> float:
        """Calculate zero crossing rate (helps distinguish speech from noise)."""
        signs = np.sign(frame)
        signs[signs == 0] = 1
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return crossings / len(frame)
    
    def _update_noise_floor(self, energy: float, is_speech: bool) -> None:
        """Update adaptive noise floor estimate."""
        if not is_speech and self.adaptive:
            # Update noise floor when no speech detected
            self._noise_floor = (
                self.adaptive_factor * self._noise_floor +
                (1 - self.adaptive_factor) * energy
            )
            # Ensure threshold stays above noise floor
            self.energy_threshold = max(
                self.energy_threshold,
                self._noise_floor * 2.0
            )
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech.
        
        Args:
            audio: Audio data as float32 numpy array (-1.0 to 1.0)
            
        Returns:
            True if speech is detected, False otherwise
        """
        if len(audio) < self.frame_samples:
            # Too short - assume it might be speech to avoid missing anything
            return True
        
        # Analyze frames
        num_frames = len(audio) // self.frame_samples
        speech_frames = 0
        max_energy = 0.0
        
        for i in range(num_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame = audio[start:end]
            
            energy = self._calculate_rms_energy(frame)
            max_energy = max(max_energy, energy)
            
            # Simple energy-based detection only
            # Removed ZCR filter as it was too restrictive for fricatives (s, f, sh)
            # and sustained vowels
            is_frame_speech = energy > self.energy_threshold
            
            if is_frame_speech:
                speech_frames += 1
            
            self._update_noise_floor(energy, is_frame_speech)
        
        # Check if enough frames contain speech
        speech_ratio = speech_frames / num_frames if num_frames > 0 else 0
        has_speech = speech_ratio >= self.speech_ratio_threshold
        
        if has_speech:
            logger.debug(f"VAD: Speech detected (ratio: {speech_ratio:.2f}, max_energy: {max_energy:.4f})")
        else:
            logger.debug(f"VAD: Silence (ratio: {speech_ratio:.2f}, max_energy: {max_energy:.4f})")
        
        return has_speech
    
    def get_speech_segments(
        self,
        audio: np.ndarray
    ) -> list[Tuple[int, int]]:
        """
        Get start/end sample indices of speech segments.
        
        Args:
            audio: Audio data as float32 numpy array
            
        Returns:
            List of (start, end) sample index tuples
        """
        segments = []
        num_frames = len(audio) // self.frame_samples
        
        in_speech = False
        speech_start = 0
        
        for i in range(num_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame = audio[start:end]
            
            energy = self._calculate_rms_energy(frame)
            is_frame_speech = energy > self.energy_threshold
            
            if is_frame_speech and not in_speech:
                # Speech started
                in_speech = True
                speech_start = start
            elif not is_frame_speech and in_speech:
                # Speech ended
                in_speech = False
                segments.append((speech_start, start))
        
        # Handle case where speech continues to end
        if in_speech:
            segments.append((speech_start, len(audio)))
        
        return segments
    
    def reset(self) -> None:
        """Reset adaptive parameters."""
        self._noise_floor = self.energy_threshold * 0.5
        self._is_initialized = False
