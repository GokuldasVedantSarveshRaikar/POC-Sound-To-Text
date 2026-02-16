# STT POC

This project contains a real-time STT service using Whisper and a client for the Philips AI Model Serving STT API.

## Local STT Service

The local service provides WebSocket-based real-time speech-to-text transcription.

### Running the Service

```bash
python -m app.main --port 9000
```

### API Endpoints

- WebSocket: `ws://localhost:9000/api/v1/ws/transcribe`
- Health: `http://localhost:9000/api/v1/health`
- Docs: `http://localhost:9000/docs`

## Philips AI STT Client

The client provides access to the Philips AI Model Serving STT API with automatic JWT token caching.

### Setup

1. Copy the `.env` file and fill in your Azure AD credentials:
   ```bash
   cp .env .env.local  # Optional: use a local env file
   ```

2. Edit the `.env` file with your credentials:
   ```
   AZURE_CLIENT_ID=your-actual-client-id
   AZURE_CLIENT_SECRET=your-actual-client-secret
   ```

### Usage

```python
from dotenv import load_dotenv
from app.clients import STTClient, STTClientError

# Load environment variables
load_dotenv()

# Use as context manager (recommended)
with STTClient() as client:
    # Health check
    if client.health_check():
        print("Authentication successful")
    
    # Transcribe audio file
    result = client.transcribe("path/to/audio.mp3")
    print(result)
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
        result = client.transcribe("audio.mp3")
except STTConfigurationError as e:
    print(f"Missing credentials: {e}")
except STTAuthenticationError as e:
    print(f"Auth failed: {e}")
except STTTranscriptionError as e:
    print(f"Transcription failed: {e}")
```

### Configuration

All configuration is loaded from the `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_CLIENT_ID` | Yes | - | Azure AD application client ID |
| `AZURE_CLIENT_SECRET` | Yes | - | Azure AD application client secret |
| `AZURE_TENANT_ID` | No | Philips tenant | Azure AD tenant ID |
| `STT_API_HOST_URL` | No | dev.api.it.philips.com | API host URL |
| `STT_API_SCOPE` | No | Philips STT scope | API scope |

### Advanced Configuration

```python
client = STTClient(
    timeout=180,           # Request timeout (seconds)
    max_retries=5,          # Retry count for transient failures
    token_buffer_seconds=600  # Refresh token 10 mins before expiry
)
```

### Features

- **Token Caching**: JWT tokens are cached and automatically refreshed before expiry
- **Thread Safety**: Token refresh is thread-safe for multi-threaded applications
- **Retry Logic**: Automatic retries with exponential backoff for transient failures
- **Connection Pooling**: HTTP connections are reused for better performance
- **Context Manager**: Proper resource cleanup with `with` statement
