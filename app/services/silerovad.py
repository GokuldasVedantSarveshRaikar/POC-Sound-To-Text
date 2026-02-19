import torch
from silero_vad import get_speech_timestamps, load_silero_vad
import soundfile as sf
import numpy as np

class SileroVADService:

    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model without passing the 'device' argument
        self.model = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            skip_validation=True
        )[0]
        
        # Move the model to the appropriate device
        self.model = self.model.to(self.device)
        
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms

    def is_speech(self, audio: torch.Tensor, sample_rate: int) -> bool:
        """Detect if the given audio contains speech."""
        try:
            # Move audio to the same device as the model
            audio = audio.to(self.device)
            
            # Get speech timestamps
            timestamps = get_speech_timestamps(
                audio,
                self.model,
                sampling_rate=sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms
            )
            
            # Check if speech is detected
            return len(timestamps) > 0 and (timestamps[-1]['end'] - timestamps[0]['start']) >= self.min_speech_duration_ms
        except Exception as e:
            print(f"Error during VAD processing: {e}")
            return False


def main():
    # Initialize the Silero VAD service
    vad_service = SileroVADService(threshold=0.5, min_speech_duration_ms=250)

    # Path to the audio file (ensure it's 16kHz mono)
    audio_file_path = "recording.wav"

    # Read the audio file
    audio, sample_rate = sf.read(audio_file_path, dtype="float32")
    
    # Convert to mono if the audio is stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Convert the audio to a PyTorch tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32)

    # Check if the audio contains speech
    is_speech_detected = vad_service.is_speech(audio_tensor, sample_rate)

    # Print the result
    if is_speech_detected:
        print("Speech detected in the audio.")
    else:
        print("No speech detected in the audio.")



if __name__ == "__main__":
    main()