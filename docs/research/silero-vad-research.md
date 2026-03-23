# Silero VAD Research

## Source
- Repository: https://github.com/snakers4/silero-vad

## Why it was selected
- Lightweight speech activity detection suitable for real-time pipelines.
- Works well on noisy audio with threshold tuning.
- Easy integration into PyTorch-based Python services.

## How it maps to this project
- Used as a gate before STT API calls to avoid transcribing silence/non-speech.
- Loaded once and shared to reduce repeated initialization overhead.
- Speech decision considers segment duration and speech ratio to improve robustness.

## Practical tuning guidance
- Increase threshold to reduce false positives in noisy environments.
- Increase minimum speech duration to reject short transient noises.
- Monitor rejection rate to avoid dropping valid short utterances.

## Risks and caveats
- Aggressive thresholding can suppress soft-spoken speech.
- Acoustic domains vary; threshold values should be validated with representative audio.
