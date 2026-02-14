from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import asyncio
from app.services.transcribe import TranscriptionService, StreamingBuffer
from app.schemas.transcribe_schema import (
    WebSocketTranscription,
    WebSocketError,
    WebSocketStatus,
    HealthResponse,
    ModelSize
)

router = APIRouter(prefix="/api/v1", tags=["transcription"])

# Service instance
_transcription_service: TranscriptionService = None


def get_transcription_service() -> TranscriptionService:
    """Dependency to get transcription service."""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService.get_instance(ModelSize.BASE)
    return _transcription_service


def init_service(model_size: ModelSize = ModelSize.BASE) -> None:
    """Initialize and load the transcription service."""
    global _transcription_service
    _transcription_service = TranscriptionService.get_instance(model_size)
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
    
    # Send connected status
    await websocket.send_json(
        WebSocketStatus(
            type="status",
            status="connected",
            message="Ready to receive audio"
        ).model_dump()
    )
    
    buffer = StreamingBuffer(buffer_seconds=3.0, overlap_seconds=0.5)
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            buffer.add_chunk(data)
            
            # Transcribe when buffer is ready
            if buffer.is_ready():
                audio = buffer.get_audio()
                
                # Run transcription in thread pool
                result = await asyncio.to_thread(
                    service.transcribe_audio_array,
                    audio,
                    "en"
                )
                
                # Send result
                if result.text:
                    await websocket.send_json(
                        WebSocketTranscription(
                            text=result.text,
                            is_final=True
                        ).model_dump()
                    )
                    
    except WebSocketDisconnect:
        # Process remaining audio
        remaining = buffer.get_remaining()
        if remaining is not None and len(remaining) > 8000:
            result = await asyncio.to_thread(
                service.transcribe_audio_array,
                remaining,
                "en"
            )
            if result.text:
                try:
                    await websocket.send_json(
                        WebSocketTranscription(
                            text=result.text,
                            is_final=True
                        ).model_dump()
                    )
                except:
                    pass
                    
    except Exception as e:
        await websocket.send_json(
            WebSocketError(type="error", message=str(e)).model_dump()
        )


@router.get("/health", response_model=HealthResponse)
async def health(
    service: TranscriptionService = Depends(get_transcription_service)
):
    return HealthResponse(
        status="ok",
        model_loaded=service.is_model_loaded(),
        model_size=service.model_size.value
    )