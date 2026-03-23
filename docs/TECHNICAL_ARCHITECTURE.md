# STT POC Technical Architecture

## 1. Purpose and Scope

This document explains the runtime architecture, processing pipeline, concurrency model, and operational considerations of the STT POC service.

The system provides:
- Real-time speech-to-text over WebSocket.
- Audio pre-processing using optional Voice Activity Detection (VAD) and spectral noise reduction.
- Remote transcription using Philips AI Model Serving STT API.
- Health and service lifecycle management with FastAPI.

## 2. High-Level Architecture

```mermaid
flowchart LR
    Client[Web Client or App\nPCM 16-bit mono 16kHz] -->|WebSocket frames| Router[FastAPI WebSocket Router]
    Router --> Buffer[StreamingBuffer RxPY Pipeline]
    Buffer --> NR[Noise Reduction\nSpectralNoiseReducer]
    Buffer --> VAD[Silero VAD Gate]
    NR --> STT[TranscriptionService]
    VAD --> STT
    STT --> API[Philips STT API]
    API --> STT
    STT --> Router
    Router -->|JSON transcription events| Client
```

### Main Responsibilities
- `app/main.py`: app bootstrapping, lifespan startup/shutdown hooks.
- `app/routers/transcribe_router.py`: WebSocket endpoint and health endpoint.
- `app/services/transcribe.py`: singleton service, stream buffer, reactive pipeline, async bridge.
- `app/clients/stt_client.py`: token acquisition/caching, resilient HTTP client, API calls.
- `app/services/silerovad.py`: speech detection.
- `app/services/noisereduce.py`: noise suppression.
- `app/schemas/transcribe_schema.py`: request/response contract.

## 3. Runtime Lifecycle

```mermaid
sequenceDiagram
    participant U as Uvicorn
    participant A as FastAPI Lifespan
    participant S as TranscriptionService
    participant C as STTClient

    U->>A: Startup event
    A->>S: init_service()
    S->>C: load_model() / initialize client
    C-->>S: ready
    A-->>U: app ready

    U->>A: Shutdown event
    A->>S: cleanup_service()
    S->>C: close session
    C-->>S: closed
    A-->>U: shutdown complete
```

## 4. Real-Time WebSocket Processing

### Endpoint behavior
1. Accept WebSocket connection.
2. Send connected status payload.
3. Instantiate `StreamingBuffer` with feature toggles and thresholds.
4. Receive binary audio chunks continuously.
5. Emit transcription events asynchronously while receiving continues.
6. On disconnect, flush buffered audio and dispose subscriptions.

```mermaid
sequenceDiagram
    participant CL as Client
    participant WS as WebSocket Router
    participant SB as StreamingBuffer
    participant TP as Thread Pool
    participant TS as TranscriptionService
    participant API as Philips STT API

    CL->>WS: Connect ws://.../api/v1/ws/transcribe
    WS-->>CL: status=connected

    loop Streaming audio
        CL->>WS: send_bytes(audio_chunk)
        WS->>SB: add_chunk(chunk)
        SB->>SB: Rx pipeline stages
        SB->>TP: schedule transcription task
        TP->>TS: transcribe_audio_array(...)
        TS->>API: HTTP transcription request
        API-->>TS: text result
        TS-->>TP: TranscriptionResponse
        TP-->>SB: on_next(result)
        SB-->>WS: enqueue result
        WS-->>CL: {type:"transcription", text:"..."}
    end

    CL--xWS: disconnect
    WS->>SB: complete() + dispose()
```

## 5. Reactive Pipeline Design (RxPY)

`StreamingBuffer` converts the inbound byte stream into a transformation pipeline.

```mermaid
flowchart TD
    A[Audio bytes input] --> B[bytes to float32 array]
    B --> C[append to internal buffer]
    C --> D{buffer >= samples_needed}
    D -- no --> A
    D -- yes --> E[extract chunk with overlap]
    E --> F{noise reduction enabled}
    F -- yes --> G[denoise chunk]
    F -- no --> H[skip]
    G --> I{VAD enabled}
    H --> I
    I -- speech --> J[observe_on thread pool]
    I -- no speech --> A
    J --> K[transcribe chunk via service]
    K --> L[emit TranscriptionResponse]
```

### Why this model was chosen
- Keeps WebSocket receive loop responsive.
- Encapsulates streaming transformations as composable operators.
- Supports non-blocking fan-out of transcription events.

## 6. Concurrency and Threading Model

The design uses two cooperating concurrency domains:

1. AsyncIO event loop:
- Handles WebSocket I/O and result emission.
- Should remain non-blocking for low latency.

2. Thread pool workers:
- Perform CPU and network-heavy transcription steps.
- Triggered with Rx `observe_on(ThreadPoolScheduler)`.

Bridge pattern:
- Worker threads push completed results back into an `asyncio.Queue` using `loop.call_soon_threadsafe`.
- A dedicated async sender task consumes queue items and sends JSON to the WebSocket.

This prevents STT network latency from blocking inbound audio ingestion.

## 7. STT Client and Authentication Strategy

The STT client is production-oriented and includes:
- Azure AD client credentials flow.
- Token caching with expiry tracking from JWT payload.
- Thread-safe token refresh via lock.
- Retry strategy for transient HTTP failures (429/5xx).
- Connection pooling via a shared `requests.Session`.
- One-time retry on 401 by clearing token cache.

```mermaid
flowchart LR
    A[Need token] --> B{Cached token valid?}
    B -- yes --> C[Use cached token]
    B -- no --> D[Acquire new token]
    D --> E[Decode JWT exp]
    E --> F[Cache token and expiry]
    C --> G[Call STT API]
    F --> G
    G --> H{401?}
    H -- yes --> I[Clear cache and retry once]
    H -- no --> J[Return response]
    I --> G
```

## 8. Audio Processing Details

### 8.1 Noise Reduction
- Implements a spectral-domain noise suppression approach.
- Uses STFT, noise power estimation, gain masking, and ISTFT reconstruction.
- Includes guardrails to avoid over-suppression by reverting to raw audio if energy drop is excessive.

### 8.2 Voice Activity Detection (VAD)
- Uses Silero VAD model loaded once and shared across instances.
- Applies timestamp-based speech detection with configurable threshold and minimum speech duration.
- Filters non-speech chunks to reduce unnecessary STT API calls.

## 9. Data Contracts

WebSocket messages are strongly typed using Pydantic models:
- Status events (`type=status`).
- Transcription events (`type=transcription`, `text`, `is_final`).
- Error events (`type=error`).

Health endpoint returns service status and model/client load state.

## 10. Configuration and Dependencies

Primary dependencies include:
- FastAPI and Uvicorn for service runtime.
- reactivex for streaming pipeline composition.
- requests and PyJWT for API transport and token handling.
- torch and silero-vad for speech gating.
- scipy and numpy for DSP and denoising primitives.

Environment variables:
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `AZURE_TENANT_ID` (optional)
- `STT_API_HOST_URL` (optional)
- `STT_API_SCOPE` (optional)

## 11. Operational Considerations

### Latency drivers
- Chunk size configuration.
- VAD gate pass rate.
- Network RTT to STT API.
- API inference time.

### Reliability drivers
- Token refresh correctness.
- Retry policy tuning.
- Backpressure handling when producer rate exceeds consumer throughput.

### Security considerations
- Never commit credentials.
- Use least-privilege service principal scopes.
- Rotate secrets and monitor token acquisition errors.

## 12. Known Tuning Hotspots

- `chunk_duration_ms`: directly impacts latency and context size.
- `vad_threshold` and `min_speech_duration_ms`: controls false accept/reject behavior.
- `noise_reduce_strength` and gain floor: controls denoise aggressiveness.
- thread pool size: controls parallelism and CPU usage.

## 13. Suggested Future Improvements

- Add per-stage latency metrics and structured tracing.
- Add bounded queue policy and backpressure telemetry.
- Add integration tests for WebSocket streaming and disconnect-flush behavior.
- Add dynamic runtime tuning endpoint for VAD/noise/chunk configs.
