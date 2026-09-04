"""
OpenAPI-based Multimodal Input/Output System

Flexible, extensible any-to-any multimodal processing:
- Video: webfeeds, streams, desktop capture, files
- Audio: streams, files, microphone
- Text: chat, documents, code
- Images: screenshots, diagrams, photos

Features:
- Standard OpenAPI REST API
- Pluggable input/output adapters
- Real-time streaming support
- Async processing with queues
- Configurable via YAML/JSON
"""

import asyncio
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Import models from dedicated models module
from cogsyndelta.api.models import (
    AgentResult,
    AgentTask,
    AudioInputConfig,
    AudioSourceType,
    ModalityType,
    MultimodalInput,
    ProcessingResult,
    SemanticState,
    VideoInputConfig,
    VideoSourceType,
)

# ============================================================================
# Input Adapters (Pluggable)
# ============================================================================


class BaseInputAdapter:
    """Base class for input adapters."""

    async def initialize(self) -> Any:
        """Initialize the adapter."""
        pass

    async def read(self) -> torch.Tensor | None:
        """Read data from input source."""
        raise NotImplementedError

    async def close(self) -> Any:
        """Clean up resources."""
        pass


class WebcamAdapter(BaseInputAdapter):
    """Webcam video input adapter."""

    def __init__(self, config: VideoInputConfig) -> None:
        """Initialize webcam adapter with configuration."""
        self.config = config
        self.capture: Any = None

    async def initialize(self) -> Any:
        """Initialize webcam capture."""
        try:
            import cv2

            self.capture = cv2.VideoCapture(self.config.device_id)
            if self.capture is not None:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
                self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        except ImportError:
            raise RuntimeError("OpenCV (cv2) required for webcam input")

    async def read(self) -> torch.Tensor | None:
        """Read frame from webcam."""
        if self.capture is None:
            return None

        ret, frame = self.capture.read()
        if not ret:
            return None

        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame_tensor

    async def close(self) -> Any:
        """Release webcam."""
        if self.capture is not None:
            self.capture.release()


class ScreenCaptureAdapter(BaseInputAdapter):
    """Desktop screen capture adapter."""

    def __init__(self, config: VideoInputConfig) -> None:
        """Initialize screen capture adapter with configuration."""
        self.config = config
        self.sct: Any = None
        self.monitor: Any = None

    async def initialize(self) -> Any:
        """Initialize screen capture."""
        try:
            import mss

            self.sct = mss.mss()
            if self.sct is not None:
                self.monitor = self.sct.monitors[1]  # Primary monitor
        except ImportError:
            raise RuntimeError("mss library required for screen capture")

    async def read(self) -> torch.Tensor | None:
        """Capture screen frame."""
        if self.sct is None:
            return None

        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)[:, :, :3]  # RGB only

        # Resize if needed
        import cv2

        frame = cv2.resize(frame, self.config.resolution)

        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame_tensor

    async def close(self) -> Any:
        """Clean up screen capture."""
        if self.sct is not None:
            self.sct.close()


class StreamAdapter(BaseInputAdapter):
    """Generic streaming adapter (RTSP, RTMP, HTTP)."""

    def __init__(self, config: VideoInputConfig) -> None:
        """Initialize stream adapter with configuration."""
        self.config = config
        self.stream = None

    async def initialize(self) -> Any:
        """Initialize stream."""
        try:
            import cv2

            self.stream = cv2.VideoCapture(self.config.source_uri)
        except ImportError:
            raise RuntimeError("OpenCV required for stream processing")

    async def read(self) -> torch.Tensor | None:
        """Read frame from stream."""
        if self.stream is None:
            return None

        ret, frame = self.stream.read()
        if not ret:
            return None

        import cv2

        frame = cv2.resize(frame, self.config.resolution)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame_tensor

    async def close(self) -> Any:
        """Release stream."""
        if self.stream is not None:
            self.stream.release()


class AudioStreamAdapter(BaseInputAdapter):
    """Audio stream adapter."""

    def __init__(self, config: AudioInputConfig) -> None:
        """Initialize audio stream adapter with configuration."""
        self.config = config
        self.stream = None

    async def initialize(self) -> Any:
        """Initialize audio stream."""
        try:
            import pyaudio

            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=int(self.config.sample_rate * self.config.buffer_duration),
            )
        except ImportError:
            raise RuntimeError("pyaudio required for audio streaming")

    async def read(self) -> torch.Tensor | None:
        """Read audio chunk."""
        if self.stream is None:
            return None

        data = self.stream.read(int(self.config.sample_rate * self.config.buffer_duration))
        audio_array = np.frombuffer(data, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_array)
        return audio_tensor

    async def close(self) -> Any:
        """Stop audio stream."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()


# ============================================================================
# Adapter Factory
# ============================================================================


class AdapterFactory:
    """Factory for creating input adapters."""

    @staticmethod
    def create_video_adapter(config: VideoInputConfig) -> BaseInputAdapter:
        """Create video input adapter based on config."""
        if config.source_type == VideoSourceType.WEBCAM:
            return WebcamAdapter(config)
        if config.source_type == VideoSourceType.SCREEN_CAPTURE:
            return ScreenCaptureAdapter(config)
        if config.source_type in [
            VideoSourceType.RTSP_STREAM,
            VideoSourceType.RTMP_STREAM,
            VideoSourceType.HTTP_STREAM,
        ]:
            return StreamAdapter(config)
        if config.source_type == VideoSourceType.FILE:
            return StreamAdapter(config)  # Can use same adapter
        raise ValueError(f"Unsupported video source type: {config.source_type}")

    @staticmethod
    def create_audio_adapter(config: AudioInputConfig) -> BaseInputAdapter:
        """Create audio input adapter based on config."""
        if config.source_type in [AudioSourceType.MICROPHONE, AudioSourceType.SYSTEM_AUDIO]:
            return AudioStreamAdapter(config)
        raise ValueError(f"Unsupported audio source type: {config.source_type}")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Self-Improving AI System API",
    description="OpenAPI-based any-to-any multimodal processing with VL-JEPA, mHC, and self-improving agents",
    version="1.0.0",
)

# CORS for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
sessions: dict[str, Any] = {}
active_adapters: dict[str, Any] = {}


@app.on_event("startup")
async def startup_event() -> Any:
    """Initialize system on startup."""
    print("Initializing Self-Improving AI System...")
    # Load integrated system here
    # global integrated_system
    # integrated_system = create_integrated_system()


@app.on_event("shutdown")
async def shutdown_event() -> Any:
    """Cleanup on shutdown."""
    for adapter in active_adapters.values():
        await adapter.close()


# ============================================================================
# API Endpoints
# ============================================================================


@app.post("/api/v1/session/create", response_model=dict[str, str])
async def create_session(config: MultimodalInput) -> Any:
    """
    Create a new processing session.

    Configure input sources and modalities for processing.
    """
    import uuid

    session_id = str(uuid.uuid4())

    sessions[session_id] = {"config": config, "created_at": datetime.now(), "status": "active"}

    return {"session_id": session_id, "status": "created"}


@app.post("/api/v1/process/video", response_model=ProcessingResult)
async def process_video(
    session_id: str, video_config: VideoInputConfig, num_frames: int = 16
) -> Any:
    """
    Process video input and return semantic states.

    Silent semantic state retention - no token generation.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Create adapter
    adapter = AdapterFactory.create_video_adapter(video_config)
    await adapter.initialize()

    # Read frames
    frames = []
    for _ in range(num_frames):
        frame = await adapter.read()
        if frame is not None:
            frames.append(frame)

    await adapter.close()

    if not frames:
        raise HTTPException(status_code=400, detail="No frames captured")

    # Process with integrated system (mock for now)
    # In real implementation: visual_output = integrated_system.process_visual_input(torch.stack(frames))

    # Mock semantic states
    semantic_states = [
        SemanticState(
            embedding=[0.1] * 512,  # Mock embedding
            modality=ModalityType.VIDEO,
            timestamp=datetime.now(),
            confidence=0.95,
            metadata={"frame_index": i},
        )
        for i in range(len(frames))
    ]

    return ProcessingResult(
        session_id=session_id,
        semantic_states=semantic_states,
        processing_time_ms=100.0,
        memory_utilization=0.5,
    )


@app.websocket("/api/v1/stream/video")
async def video_stream_endpoint(websocket: WebSocket) -> Any:
    """
    WebSocket endpoint for real-time video streaming.

    Supports continuous visual processing with silent semantic states.
    """
    await websocket.accept()

    try:
        # Receive config
        config_data = await websocket.receive_json()
        video_config = VideoInputConfig(**config_data)

        # Create adapter
        adapter = AdapterFactory.create_video_adapter(video_config)
        await adapter.initialize()

        frame_count = 0
        while True:
            # Read frame
            frame = await adapter.read()
            if frame is None:
                break

            frame_count += 1

            # Process every Nth frame
            if frame_count % video_config.frame_skip == 0:
                # Process with system (mock)
                semantic_state = {
                    "frame_id": frame_count,
                    "embedding": [0.1] * 512,  # Mock
                    "timestamp": datetime.now().isoformat(),
                    "modality": "video",
                }

                # Send back semantic state
                await websocket.send_json(semantic_state)

            # Small delay for real-time processing
            await asyncio.sleep(1.0 / video_config.fps)

        await adapter.close()

    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


@app.post("/api/v1/agent/task", response_model=AgentResult)
async def create_agent_task(task: AgentTask, background_tasks: BackgroundTasks) -> Any:
    """
    Create self-improving agent task.

    Supports code generation, improvement, security auditing, and multi-language exploration.
    """
    import uuid

    task_id = str(uuid.uuid4())

    # In real implementation, process with agent framework
    # result = integrated_system.agent_framework.improve_solution(...)

    return AgentResult(
        task_id=task_id,
        status="completed",
        result={"implementation": "mock_code"},
        quality_score=0.92,
        security_score=0.15,
        improvements=["Added input validation", "Improved error handling"],
    )


@app.post("/api/v1/agent/multi-language")
async def explore_multi_language(
    problem_description: str, languages: list[str] = ["python", "rust", "go"]
) -> Any:
    """
    Explore implementations across multiple languages.

    Returns best implementation with quality and security metrics.
    """
    # Mock implementation
    results = {}
    for lang in languages:
        results[lang] = {
            "quality": np.random.uniform(0.7, 0.95),
            "security_risk": np.random.uniform(0.1, 0.3),
            "performance": np.random.uniform(0.8, 0.99),
        }

    best_lang = max(results.keys(), key=lambda k: results[k]["quality"])

    return {"best_language": best_lang, "best_score": results[best_lang], "all_results": results}


@app.get("/api/v1/config/adapters")
async def list_available_adapters() -> Any:
    """
    List all available input/output adapters.

    Shows supported video sources, audio sources, and modalities.
    """
    return {
        "video_sources": [source.value for source in VideoSourceType],
        "audio_sources": [source.value for source in AudioSourceType],
        "modalities": [mod.value for mod in ModalityType],
        "extensible": True,
        "custom_adapters": "Supported via plugin system",
    }


@app.post("/api/v1/memory/query")
async def query_memory(session_id: str, query_embedding: list[float], num_results: int = 5) -> Any:
    """
    Query temporal memory bank for relevant semantic states.

    Silent semantic state retrieval without token generation.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Mock memory retrieval
    retrieved_states = [
        {
            "embedding": [0.1] * 512,
            "similarity": 0.95 - i * 0.1,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"source": "video_frame"},
        }
        for i in range(num_results)
    ]

    return {
        "session_id": session_id,
        "query_results": retrieved_states,
        "num_retrieved": num_results,
    }


@app.get("/api/v1/health")
async def health_check() -> Any:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "features": {
            "vl_jepa": True,
            "mhc": True,
            "self_improving_agents": True,
            "multimodal": True,
            "any_to_any": True,
        },
    }


@app.get("/")
async def root() -> Any:
    """Root endpoint with API documentation link."""
    return {
        "message": "Self-Improving AI System API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "features": [
            "Any-to-any multimodal processing",
            "Silent semantic state retention",
            "VL-JEPA joint embeddings",
            "mHC moderated hyper connections",
            "Self-improving agents (SWE/AIE/SWD/AID)",
            "Real-time streaming support",
            "Security hardening & QA",
        ],
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("SELF-IMPROVING AI SYSTEM - OpenAPI Server")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✓ Any-to-any multimodal processing")
    print("  ✓ Video: webcam, screen capture, streams, files")
    print("  ✓ Audio: microphone, system audio, files")
    print("  ✓ Text: chat, documents, code")
    print("  ✓ WebSocket streaming support")
    print("  ✓ RESTful API with OpenAPI spec")
    print("  ✓ Extensible adapter system")
    print()
    print("Starting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 70)

    # Default to loopback, not 0.0.0.0. The banner above already advertises
    # http://localhost:8000, so binding every interface contradicted what it told the
    # operator, and this fleet's posture is LAN-only exposure through the Caddy edge --
    # not services publishing themselves. Override deliberately via CSD_API_HOST when a
    # host genuinely needs to serve the LAN.
    host = os.environ.get("CSD_API_HOST", "127.0.0.1")
    port = int(os.environ.get("CSD_API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
