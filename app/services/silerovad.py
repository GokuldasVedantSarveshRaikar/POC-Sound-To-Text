import torch
from silero_vad import get_speech_timestamps
import threading




class SileroVADService:
    _shared_model = None
    _shared_device = None
    _load_lock = threading.Lock()

    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250):
        self.model, self.device = self._get_shared_model_and_device()
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms

    @classmethod
    def _get_shared_model_and_device(cls):
        if cls._shared_model is not None and cls._shared_device is not None:
            return cls._shared_model, cls._shared_device

        with cls._load_lock:
            if cls._shared_model is None or cls._shared_device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    skip_validation=True
                )[0]
                model = model.to(device)
                model.eval()

                cls._shared_model = model
                cls._shared_device = device

        return cls._shared_model, cls._shared_device

    def is_speech(self, audio: torch.Tensor, sample_rate: int) -> bool:
        """Robust speech decision for noisy chunks."""
        try:
            audio = audio.to(self.device).float().flatten()

            timestamps = get_speech_timestamps(
                audio,
                self.model,
                sampling_rate=sample_rate,
                threshold=self.threshold,                 # try 0.8-0.9 in noisy rooms
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=120,
                speech_pad_ms=30,
                return_seconds=False                      # start/end in samples
            )

            if not timestamps:
                return False

            total_samples = int(audio.numel())
            min_samples = int(sample_rate * self.min_speech_duration_ms / 1000)

            speech_samples = sum(seg["end"] - seg["start"] for seg in timestamps)
            longest_segment = max(seg["end"] - seg["start"] for seg in timestamps)
            speech_ratio = speech_samples / max(total_samples, 1)

            # Tune these on your environment
            return longest_segment >= min_samples and speech_ratio >= 0.08

        except Exception as e:
            print(f"Error during VAD processing: {e}")
            return False



