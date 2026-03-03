from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import asyncio
from app.services.transcribe import TranscriptionService, StreamingBuffer
from app.schemas.transcribe_schema import (
    WebSocketTranscription,
    WebSocketError,
    WebSocketStatus,
    HealthResponse,
)


router = APIRouter(prefix="/api/v1", tags=["transcription"])

# Service instance
_transcription_service: TranscriptionService = None


def get_transcription_service() -> TranscriptionService:
    """Dependency to get transcription service."""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService.get_instance()
    return _transcription_service


def init_service() -> None:
    """Initialize the transcription service (connects to STT API)."""
    global _transcription_service
    _transcription_service = TranscriptionService.get_instance()
    _transcription_service.load_model()


def cleanup_service() -> None:
    """Cleanup the transcription service."""
    global _transcription_service
    if _transcription_service is not None:
        _transcription_service.unload_model()
        _transcription_service = None


@router.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    service: TranscriptionService = Depends(get_transcription_service)
):
    await websocket.accept()
    await websocket.send_json(
        WebSocketStatus(type="status", status="connected", message="Ready to receive audio").model_dump()
    )
    
    # Create buffer with service injection
    buffer = StreamingBuffer(
        transcription_service=service,
        buffer_seconds=3.0,
        overlap_seconds=0.5,
        language="en",
        vad_enabled=True, 
        vad_threshold=0.5,
        noise_reduce_enabled=True,
        noise_reduce_strength=1.0,
    )
    
    # Queue for async communication between RxPY and asyncio
    result_queue = asyncio.Queue()
    
    # Capture the running event loop BEFORE subscribing (important!)
    loop = asyncio.get_running_loop()
    
    # Subscribe to transcription results
    buffer.subscribe(
        on_transcription=lambda r: loop.call_soon_threadsafe(result_queue.put_nowait, r),
        on_error=lambda e: print(f"Error: {e}")
    )
    
    async def send_results():
        while True:
            result = await result_queue.get()
            if result.text:
                await websocket.send_json(
                    WebSocketTranscription(text=result.text, is_final=True).model_dump()
                )
    
    # Start result sender task
    sender_task = asyncio.create_task(send_results())
    
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.add_chunk(data)  # Non-blocking, RxPY handles the rest
            
    except WebSocketDisconnect:
        buffer.complete()  # Flush remaining audio
        
    except Exception as e:
        await websocket.send_json(WebSocketError(type="error", message=str(e)).model_dump())
        
    finally:
        sender_task.cancel()
        buffer.dispose()


@router.get("/health", response_model=HealthResponse)
async def health(
    service: TranscriptionService = Depends(get_transcription_service)
):
    return HealthResponse(
        status="ok" if service.is_model_loaded() else "not_initialized",
        model_loaded=service.is_model_loaded(),
        model_size="api"  # Using remote STT API
    )