# Agent Guidelines for STT POC

Real-time Speech-to-Text service using Philips AI Model Serving STT API with WebSocket-based transcription, VAD, and noise reduction.

## Project Structure
```
app/
├── main.py                 # FastAPI entry point
├── clients/stt_client.py   # Philips AI STT API client with JWT caching
├── routers/transcribe_router.py # WebSocket/HTTP endpoints
├── schemas/transcribe_schema.py  # Pydantic models
├── services/
│   ├── transcribe.py       # Transcription with RxPY streaming
│   ├── silerovad.py        # Silero VAD integration
│   └── noisereduce.py      # Spectral noise reduction
└── utils/logger.py
```

## Commands

### Running Server
```bash
python -m app.main --port 9000           # Start on port 9000
python -m app.main --port 9000 --reload  # Auto-reload
python -m app.main --host 0.0.0.0 --port 9000
```

### Linting (ruff)
```bash
ruff check .           # Lint all
ruff check --fix .     # Auto-fix
ruff check app/main.py # Lint specific file
```

### Testing (pytest)
```bash
pytest                         # All tests
pytest tests/test_client.py   # Single file
pytest -v                     # Verbose
```

### Virtual Environment
```bash
.venv\Scripts\activate  # Windows
uv run python -m app.main --port 9000
```

## Code Style

### Imports (grouped: stdlib, third-party, local)
```python
import os
import logging
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel, Field
from reactivex import operators as ops

from app.clients import STTClient
```

### Formatting
- Max line length: 88 chars, 4 spaces indentation, use f-strings

### Types
- Type hints for all params/returns, use `Optional[T]` not `T | None`

### Naming
- Classes: `PascalCase`, functions: `snake_case`, constants: `SCREAMING_SNAKE_CASE`

### Error Handling
```python
class STTClientError(Exception): pass
class STTAuthenticationError(STTClientError): pass

try:
    response = self._session.post(url, data=data)
except requests.exceptions.RequestException as e:
    logger.error("Failed: %s", str(e))
    raise STTAuthenticationError(f"Failed: {e}") from e
```

### Logging
```python
logger = logging.getLogger(__name__)
logger.info("STT initialized: %s", self.host_url)
```

### Service Patterns
Singleton pattern, dependency injection via FastAPI `Depends`.

```python
class TranscriptionService:
    _instance: Optional["TranscriptionService"] = None

    @classmethod
    def get_instance(cls) -> "TranscriptionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

### Reactive Programming (RxPY)
```python
from reactivex import operators as ops
from reactivex.subject import Subject

self._audio_input: Subject[bytes] = Subject()
```

## Environment Variables
```
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=optional
STT_API_HOST_URL=dev.api.it.philips.com
```

## Using STT Client
```python
from dotenv import load_dotenv
from app.clients import STTClient

load_dotenv()

with STTClient() as client:
    result = client.transcribe("audio.wav")
    print(result["text"])
```

### Error Handling
```python
from app.clients import (
    STTClient,
    STTConfigurationError,
    STTAuthenticationError,
    STTTranscriptionError,
)

try:
    with STTClient() as client:
        result = client.transcribe("audio.wav")
except STTConfigurationError as e:
    print(f"Config error: {e}")
except STTAuthenticationError as e:
    print(f"Auth failed: {e}")
except STTTranscriptionError as e:
    print(f"Transcription failed: {e}")
```

## WebSocket API

Connect to `ws://localhost:9000/api/v1/ws/transcribe`, send raw PCM audio (16-bit, 16kHz, mono):

```python
await websocket.send_bytes(audio_bytes)
# Receive: {"type": "transcription", "text": "Hello", "is_final": true}
```
