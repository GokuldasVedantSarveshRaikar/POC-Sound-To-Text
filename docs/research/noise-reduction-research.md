# Noise Reduction Research

## Source
- Paper link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1164453

## Why it was selected
- Spectral-domain noise suppression is suitable for speech enhancement in streaming pipelines.
- Balances quality and computational cost for real-time use.

## How it maps to this project
- Input chunks are transformed with STFT.
- Noise power is estimated and used to construct a suppression gain mask.
- Enhanced spectrum is reconstructed with ISTFT.
- Guardrails prevent over-suppression by reverting to raw audio when needed.

## Practical tuning guidance
- `prop_decrease` controls suppression strength.
- `gain_floor` protects low-energy speech from being fully removed.
- Window and hop sizes should match speech bandwidth and latency targets.

## Risks and caveats
- Over-suppression can reduce intelligibility.
- Under-suppression leaves noise that may degrade transcription quality.
- Tuning should be validated against SNR and transcription word error rate (WER).
