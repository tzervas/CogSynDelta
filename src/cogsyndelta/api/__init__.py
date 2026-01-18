"""API modules for REST and WebSocket interfaces."""

import sys

from cogsyndelta.api.google_adk_adapter import (
    A2AMessage,
    A2AProtocolAdapter,
    ADKAdapter,
    ADKAgent,
    AgentCapability,
    AgentRole,
    CogSynDeltaADKAgent,
    Message,
    MessageType,
)

# Pydantic models are incompatible with Python 3.14 beta due to
# typing._eval_type changes. Only import when not on 3.14 beta.
_PYTHON_314_BETA = sys.version_info[:2] == (3, 14)

if not _PYTHON_314_BETA:
    from cogsyndelta.api.models import (
        AgentResult,
        AgentTask,
        AudioInputConfig,
        AudioSourceType,
        ModalityType,
        MultimodalInput,
        ProcessingMode,
        ProcessingResult,
        SemanticState,
        TextInputConfig,
        VideoInputConfig,
        VideoSourceType,
    )

    __all__ = [
        "A2AMessage",
        "A2AProtocolAdapter",
        "ADKAdapter",
        "ADKAgent",
        "AgentCapability",
        "AgentResult",
        "AgentRole",
        "AgentTask",
        "AudioInputConfig",
        "AudioSourceType",
        "CogSynDeltaADKAgent",
        "Message",
        "MessageType",
        "ModalityType",
        "MultimodalInput",
        "ProcessingMode",
        "ProcessingResult",
        "SemanticState",
        "TextInputConfig",
        "VideoInputConfig",
        "VideoSourceType",
    ]
else:
    __all__ = [
        "A2AMessage",
        "A2AProtocolAdapter",
        "ADKAdapter",
        "ADKAgent",
        "AgentCapability",
        "AgentRole",
        "CogSynDeltaADKAgent",
        "Message",
        "MessageType",
    ]
