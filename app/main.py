from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers.transcribe_router import (
    router as transcribe_router,
    init_service,
    cleanup_service,
)
import uvicorn
import argparse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize STT API client on startup, cleanup on shutdown."""
    print("Starting up...")
    init_service()
    print("STT API Client initialized!")
    yield
    # Cleanup
    print("Shutting down...")
    cleanup_service()
    print("Cleanup complete!")


app = FastAPI(
    title="Real-time STT Service",
    version="1.0.0",
    description="WebSocket-based real-time speech-to-text service using Philips AI STT API",
    lifespan=lifespan,
)

# Include routers
app.include_router(transcribe_router)


@app.get("/")
async def root():
    return {
        "message": "Real-time STT Service",
        "docs": "/docs",
        "websocket": "/api/v1/ws/transcribe",
        "health": "/api/v1/health",
    }


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        description="Real-time STT WebSocket Server (Philips AI API)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=9000, help="Port to bind (default: 9000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers (default: 1)"
    )

    args = parser.parse_args()

    print(f"Starting Real-time STT Service (Philips AI API) on {args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print(f"WebSocket: ws://{args.host}:{args.port}/api/v1/ws/transcribe")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
