# API Reference

Complete reference for the CogSynDelta OpenAPI.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently no authentication required. Future versions may add API key authentication.

## REST Endpoints

### Session Management

#### Create Session

```http
POST /session/create
```

**Request Body:**

```json
{
  "modalities": ["video", "audio", "text"],
  "video_config": {
    "source_type": "webcam",
    "device_id": 0,
    "resolution": [224, 224],
    "fps": 30
  },
  "processing_mode": "realtime"
}
```

**Response:**

```json
{
  "session_id": "uuid-string",
  "status": "created"
}
```

### Video Processing

#### Process Video

```http
POST /process/video
```

**Request Body:**

```json
{
  "session_id": "uuid-string",
  "video_config": {
    "source_type": "webcam",
    "device_id": 0,
    "resolution": [224, 224]
  },
  "num_frames": 16
}
```

**Response:**

```json
{
  "session_id": "uuid-string",
  "semantic_states": [
    {
      "embedding": [0.1, 0.2, ...],
      "modality": "video",
      "timestamp": "2024-01-18T12:00:00",
      "confidence": 0.95
    }
  ],
  "processing_time_ms": 100.0,
  "memory_utilization": 0.5
}
```

### Agent Tasks

#### Create Agent Task

```http
POST /agent/task
```

**Request Body:**

```json
{
  "task_type": "code_generation",
  "input_data": {
    "problem_description": "Sort an array in Python",
    "requirements": ["efficient", "in-place"]
  },
  "languages": ["python"],
  "quality_threshold": 0.85,
  "num_iterations": 3
}
```

**Response:**

```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "result": {
    "code": "def sort_array(arr):...",
    "language": "python"
  },
  "quality_score": 0.92,
  "security_score": 0.15,
  "improvements": [
    "Added input validation",
    "Improved error handling"
  ]
}
```

#### Multi-Language Exploration

```http
POST /agent/multi-language
```

**Request Body:**

```json
{
  "problem_description": "Implement binary search",
  "languages": ["python", "rust", "go"]
}
```

**Response:**

```json
{
  "best_language": "rust",
  "best_score": {
    "quality": 0.95,
    "security_risk": 0.10,
    "performance": 0.98
  },
  "all_results": {
    "python": {...},
    "rust": {...},
    "go": {...}
  }
}
```

### Memory Queries

#### Query Memory

```http
POST /memory/query
```

**Request Body:**

```json
{
  "session_id": "uuid-string",
  "query_embedding": [0.1, 0.2, ...],
  "num_results": 5
}
```

**Response:**

```json
{
  "session_id": "uuid-string",
  "query_results": [
    {
      "embedding": [0.1, 0.2, ...],
      "similarity": 0.95,
      "timestamp": "2024-01-18T12:00:00",
      "metadata": {}
    }
  ],
  "num_retrieved": 5
}
```

### Configuration

#### List Available Adapters

```http
GET /config/adapters
```

**Response:**

```json
{
  "video_sources": ["webcam", "screen_capture", "rtsp_stream", ...],
  "audio_sources": ["microphone", "system_audio", ...],
  "modalities": ["video", "audio", "text", "code", "image"],
  "extensible": true,
  "custom_adapters": "Supported via plugin system"
}
```

### Health Check

#### Health

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "features": {
    "vl_jepa": true,
    "mhc": true,
    "self_improving_agents": true,
    "multimodal": true,
    "any_to_any": true
  }
}
```

## WebSocket Endpoints

### Video Streaming

```
ws://localhost:8000/api/v1/stream/video
```

**Send configuration:**

```json
{
  "source_type": "webcam",
  "device_id": 0,
  "resolution": [224, 224],
  "fps": 30,
  "frame_skip": 1
}
```

**Receive semantic states:**

```json
{
  "frame_id": 1,
  "embedding": [0.1, 0.2, ...],
  "timestamp": "2024-01-18T12:00:00.123",
  "modality": "video"
}
```

## Error Responses

All endpoints return errors in consistent format:

```json
{
  "error": "Error description",
  "details": "Additional details",
  "status_code": 400
}
```

### Common Status Codes

- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error
- `503`: Service Unavailable

## Rate Limiting

Currently no rate limiting. Future versions may implement:
- 100 requests/minute per IP
- 10 concurrent WebSocket connections

## SDK Examples

### Python

```python
import requests

class CogSynDeltaClient:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
    
    def create_session(self, modalities):
        response = requests.post(
            f"{self.base_url}/session/create",
            json={"modalities": modalities}
        )
        return response.json()
    
    def process_video(self, session_id, config):
        response = requests.post(
            f"{self.base_url}/process/video",
            json={
                "session_id": session_id,
                "video_config": config
            }
        )
        return response.json()

# Usage
client = CogSynDeltaClient()
session = client.create_session(["video", "text"])
result = client.process_video(session['session_id'], {
    "source_type": "webcam",
    "device_id": 0
})
```

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:
```
http://localhost:8000/openapi.json
```

Interactive documentation:
```
http://localhost:8000/docs
```
