"""
Pydantic models for the API.

These models are separated from server.py to avoid
FastAPI import issues with Python 3.14 beta.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    """Supported input/output modalities."""

    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    MULTIMODAL = "multimodal"


class VideoSourceType(str, Enum):
    """Video input source types."""

    WEBCAM = "webcam"
    SCREEN_CAPTURE = "screen_capture"
    FILE = "file"
    URL = "url"
    RTSP_STREAM = "rtsp_stream"
    RTMP_STREAM = "rtmp_stream"
    WEBRTC = "webrtc"
    HTTP_STREAM = "http_stream"
    WEBSOCKET = "websocket"


class AudioSourceType(str, Enum):
    """Audio input source types."""

    MICROPHONE = "microphone"
    FILE = "file"
    STREAM = "stream"
    SYSTEM_AUDIO = "system_audio"


class ProcessingMode(str, Enum):
    """Processing mode for inputs."""

    REALTIME = "realtime"  # Low latency, streaming
    BATCH = "batch"  # Higher quality, batched
    ASYNC = "async"  # Background processing


# Request/Response Models


class VideoInputConfig(BaseModel):
    """Configuration for video input."""

    source_type: VideoSourceType
    source_uri: str | None = Field(None, description="URI for file, URL, or stream")
    device_id: int | None = Field(0, description="Device ID for webcam")
    resolution: tuple[int, int] | None = Field(
        (224, 224), description="Target resolution (width, height)"
    )
    fps: int | None = Field(30, description="Target frames per second")
    buffer_size: int | None = Field(16, description="Frame buffer size")
    frame_skip: int | None = Field(1, description="Process every Nth frame")
    enable_preprocessing: bool = Field(True, description="Enable frame preprocessing")


class AudioInputConfig(BaseModel):
    """Configuration for audio input."""

    source_type: AudioSourceType
    source_uri: str | None = None
    sample_rate: int = Field(16000, description="Audio sample rate")
    channels: int = Field(1, description="Number of audio channels")
    buffer_duration: float = Field(1.0, description="Buffer duration in seconds")


class TextInputConfig(BaseModel):
    """Configuration for text/code input."""

    language: str | None = Field("python", description="Programming language for code")
    max_length: int = Field(2048, description="Maximum sequence length")
    tokenizer: str = Field("default", description="Tokenizer to use")


class MultimodalInput(BaseModel):
    """Multimodal input request."""

    modalities: list[ModalityType]
    video_config: VideoInputConfig | None = None
    audio_config: AudioInputConfig | None = None
    text_config: TextInputConfig | None = None
    processing_mode: ProcessingMode = ProcessingMode.REALTIME
    session_id: str | None = None


class SemanticState(BaseModel):
    """Silent semantic state representation."""

    embedding: list[float]
    modality: ModalityType
    timestamp: datetime
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Result from processing multimodal input."""

    session_id: str
    semantic_states: list[SemanticState]
    predictions: dict[str, Any] | None = None
    quality_metrics: dict[str, float] | None = None
    processing_time_ms: float
    memory_utilization: float


class AgentTask(BaseModel):
    """Task for self-improving agent."""

    task_type: Literal[
        "code_generation",
        "code_improvement",
        "security_audit",
        "multi_language_exploration",
    ]
    input_data: dict[str, Any]
    languages: list[str] | None = Field(["python"], description="Target languages")
    quality_threshold: float = Field(0.85, description="Minimum quality threshold")
    security_threshold: float = Field(0.2, description="Maximum security risk threshold")
    num_iterations: int = Field(3, description="Self-improvement iterations")


class AgentResult(BaseModel):
    """Result from agent task."""

    task_id: str
    status: Literal["completed", "failed", "in_progress"]
    result: dict[str, Any] | None = None
    quality_score: float | None = None
    security_score: float | None = None
    improvements: list[str] | None = None
    error: str | None = None
