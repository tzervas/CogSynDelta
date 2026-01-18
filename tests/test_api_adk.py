"""
Unit and Integration Tests for Google ADK Adapter

Tests cover:
1. Message and A2AMessage dataclass validation
2. AgentRole and MessageType enums
3. ADKAgent tool registration and function execution
4. A2AProtocolAdapter message routing (mocked)
5. CogSynDeltaADKAgent integration
6. ADKAdapter factory methods

Based on Google ADK specification:
- Function tools with parameters and return types
- A2A protocol message format
- Agent lifecycle management

All tests are CPU-compatible for CI.

NOTE: Tests are skipped on Python 3.14 beta due to Pydantic compatibility
issues with typing._eval_type changes. This is a known upstream issue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Check if we're on Python 3.14 beta - Pydantic has compatibility issues
PYTHON_314_BETA = sys.version_info[:2] == (3, 14)

# Skip entire module on Python 3.14 beta
if PYTHON_314_BETA:
    pytest.skip(
        "Pydantic incompatible with Python 3.14 beta typing._eval_type changes",
        allow_module_level=True,
    )

# Add src to path (E402 is expected here due to conditional skip)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cogsyndelta.api import (  # noqa: E402
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


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_agent_role_values(self) -> None:
        """Test all agent role values per ADK spec."""
        assert AgentRole.USER == "user"
        assert AgentRole.ASSISTANT == "assistant"
        assert AgentRole.SYSTEM == "system"
        assert AgentRole.FUNCTION == "function"

    def test_agent_role_count(self) -> None:
        """Test expected number of roles."""
        assert len(AgentRole) == 4


class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_type_values(self) -> None:
        """Test all A2A message type values."""
        assert MessageType.REQUEST == "request"
        assert MessageType.RESPONSE == "response"
        assert MessageType.FUNCTION_CALL == "function_call"
        assert MessageType.FUNCTION_RESPONSE == "function_response"
        assert MessageType.ERROR == "error"
        assert MessageType.STREAM == "stream"

    def test_message_type_count(self) -> None:
        """Test expected number of message types."""
        assert len(MessageType) == 6


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation_minimal(self) -> None:
        """Test creating message with minimal fields."""
        msg = Message(role=AgentRole.USER, content="Hello, agent!")
        assert msg.role == AgentRole.USER
        assert msg.content == "Hello, agent!"
        assert msg.name is None
        assert msg.function_call is None
        assert msg.timestamp is not None
        assert msg.metadata == {}

    def test_message_with_function_call(self) -> None:
        """Test creating message with function call per ADK spec."""
        msg = Message(
            role=AgentRole.ASSISTANT,
            content="",
            function_call={"name": "get_weather", "arguments": {"city": "NYC"}},
        )
        assert msg.function_call is not None
        assert msg.function_call["name"] == "get_weather"
        assert msg.function_call["arguments"]["city"] == "NYC"

    def test_message_with_metadata(self) -> None:
        """Test message with custom metadata."""
        msg = Message(
            role=AgentRole.SYSTEM,
            content="System prompt",
            metadata={"version": "1.0", "context_id": "ctx-123"},
        )
        assert msg.metadata["version"] == "1.0"


class TestAgentCapability:
    """Tests for AgentCapability dataclass."""

    def test_capability_creation(self) -> None:
        """Test creating agent capability per ADK spec."""
        cap = AgentCapability(
            name="search_code",
            description="Search for code snippets in repository",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "language": {"type": "string"},
                },
            },
            required=["query"],
        )
        assert cap.name == "search_code"
        assert "query" in cap.required
        assert cap.parameters["type"] == "object"


class TestA2AMessage:
    """Tests for A2AMessage dataclass."""

    def test_a2a_message_creation(self) -> None:
        """Test creating A2A protocol message."""
        msg = A2AMessage(
            message_id="msg-001",
            message_type=MessageType.REQUEST,
            sender_agent_id="agent-a",
            receiver_agent_id="agent-b",
            payload={"task": "analyze_code", "code": "def foo(): pass"},
        )
        assert msg.message_id == "msg-001"
        assert msg.message_type == MessageType.REQUEST
        assert msg.sender_agent_id == "agent-a"
        assert msg.receiver_agent_id == "agent-b"
        assert msg.payload["task"] == "analyze_code"

    def test_a2a_message_to_dict(self) -> None:
        """Test A2A message serialization to dict."""
        msg = A2AMessage(
            message_id="msg-002",
            message_type=MessageType.RESPONSE,
            sender_agent_id="agent-b",
            receiver_agent_id="agent-a",
            payload={"result": "success"},
        )
        data = msg.to_dict()

        assert data["message_id"] == "msg-002"
        assert data["type"] == "response"  # MessageType value
        assert data["sender"] == "agent-b"
        assert data["receiver"] == "agent-a"
        assert data["payload"]["result"] == "success"
        assert "timestamp" in data

    def test_a2a_message_from_dict(self) -> None:
        """Test A2A message deserialization from dict."""
        data = {
            "message_id": "msg-003",
            "type": "error",
            "sender": "system",
            "receiver": "agent-a",
            "payload": {"error": "Not found"},
            "timestamp": "2026-01-18T12:00:00",
        }
        msg = A2AMessage.from_dict(data)

        assert msg.message_id == "msg-003"
        assert msg.message_type == MessageType.ERROR
        assert msg.sender_agent_id == "system"
        assert msg.payload["error"] == "Not found"

    def test_a2a_message_roundtrip(self) -> None:
        """Test A2A message serialization/deserialization roundtrip."""
        original = A2AMessage(
            message_id="msg-rt",
            message_type=MessageType.FUNCTION_CALL,
            sender_agent_id="caller",
            receiver_agent_id="executor",
            payload={"function": "process", "args": [1, 2, 3]},
        )
        data = original.to_dict()
        restored = A2AMessage.from_dict(data)

        assert restored.message_id == original.message_id
        assert restored.message_type == original.message_type
        assert restored.payload == original.payload


class TestADKAgent:
    """Tests for ADKAgent base class."""

    @pytest.fixture
    def agent(self) -> ADKAgent:
        """Create a concrete ADKAgent for testing."""

        class TestAgent(ADKAgent):
            def _generate_response(self, prompt: str) -> str:
                return f"Echo: {prompt}"

        return TestAgent(
            agent_id="test-agent-001",
            name="Test Agent",
            description="An agent for testing",
        )

    def test_agent_initialization(self, agent: ADKAgent) -> None:
        """Test agent initializes with correct identity."""
        assert agent.agent_id == "test-agent-001"
        assert agent.name == "Test Agent"
        assert agent.description == "An agent for testing"
        assert agent.conversation_history == []
        assert agent.tools == {}
        assert agent.capabilities == []

    def test_register_tool(self, agent: ADKAgent) -> None:
        """Test tool registration per ADK spec."""

        def sample_tool(query: str, limit: int = 10) -> dict:
            return {"results": [], "count": 0}

        agent.register_tool(
            name="search",
            func=sample_tool,
            description="Search for items",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            required=["query"],
        )

        assert "search" in agent.tools
        assert len(agent.capabilities) == 1
        assert agent.capabilities[0].name == "search"

    def test_process_message(self, agent: ADKAgent) -> None:
        """Test message processing."""
        msg = Message(role=AgentRole.USER, content="Hello!")
        response = agent.process_message(msg)

        assert response.role == AgentRole.ASSISTANT
        assert "Echo: Hello!" in response.content
        assert len(agent.conversation_history) == 2  # Input + response

    def test_process_function_call(self, agent: ADKAgent) -> None:
        """Test function call execution per ADK spec."""

        def add_numbers(a: int, b: int) -> int:
            return a + b

        agent.register_tool(
            name="add",
            func=add_numbers,
            description="Add two numbers",
            parameters={"type": "object", "properties": {"a": {}, "b": {}}},
            required=["a", "b"],
        )

        msg = Message(
            role=AgentRole.USER,
            content="",
            function_call={"name": "add", "arguments": {"a": 5, "b": 3}},
        )
        response = agent.process_message(msg)

        assert response.role == AgentRole.FUNCTION
        assert response.name == "add"
        result = json.loads(response.content)
        assert result["result"] == 8

    def test_process_unknown_function(self, agent: ADKAgent) -> None:
        """Test handling of unknown function call."""
        msg = Message(
            role=AgentRole.USER,
            content="",
            function_call={"name": "unknown_func", "arguments": {}},
        )
        response = agent.process_message(msg)

        assert response.role == AgentRole.FUNCTION
        result = json.loads(response.content)
        assert "error" in result
        assert "not found" in result["error"]

    def test_get_capabilities(self, agent: ADKAgent) -> None:
        """Test capabilities export in ADK format."""

        def tool_func() -> str:
            return "result"

        agent.register_tool(
            name="my_tool",
            func=tool_func,
            description="A tool",
            parameters={"type": "object"},
        )

        caps = agent.get_capabilities()
        assert len(caps) == 1
        assert caps[0]["type"] == "function"
        assert caps[0]["function"]["name"] == "my_tool"

    def test_reset_conversation(self, agent: ADKAgent) -> None:
        """Test conversation reset."""
        msg = Message(role=AgentRole.USER, content="Test")
        agent.process_message(msg)
        assert len(agent.conversation_history) > 0

        agent.reset_conversation()
        assert agent.conversation_history == []


class TestA2AProtocolAdapter:
    """Tests for A2AProtocolAdapter."""

    @pytest.fixture
    def adapter(self) -> A2AProtocolAdapter:
        """Create A2A protocol adapter."""
        return A2AProtocolAdapter()

    @pytest.fixture
    def test_agents(self) -> tuple[ADKAgent, ADKAgent]:
        """Create two test agents for A2A communication."""

        class EchoAgent(ADKAgent):
            def _generate_response(self, prompt: str) -> str:
                return f"Agent {self.name} received: {prompt}"

        agent_a = EchoAgent("agent-a", "Agent A", "First agent")
        agent_b = EchoAgent("agent-b", "Agent B", "Second agent")
        return agent_a, agent_b

    def test_register_agent(
        self, adapter: A2AProtocolAdapter, test_agents: tuple[ADKAgent, ADKAgent]
    ) -> None:
        """Test agent registration with A2A adapter."""
        agent_a, agent_b = test_agents
        adapter.register_agent(agent_a)
        adapter.register_agent(agent_b)

        assert "agent-a" in adapter.agents
        assert "agent-b" in adapter.agents

    @pytest.mark.asyncio
    async def test_send_message(
        self, adapter: A2AProtocolAdapter, test_agents: tuple[ADKAgent, ADKAgent]
    ) -> None:
        """Test A2A message sending."""
        agent_a, agent_b = test_agents
        adapter.register_agent(agent_a)
        adapter.register_agent(agent_b)

        msg = A2AMessage(
            message_id="test-msg",
            message_type=MessageType.REQUEST,
            sender_agent_id="agent-a",
            receiver_agent_id="agent-b",
            payload={"message": "Hello from A"},
        )

        response = await adapter.send_message(msg)

        assert response.message_type == MessageType.RESPONSE
        assert response.sender_agent_id == "agent-b"
        assert response.receiver_agent_id == "agent-a"
        assert "content" in response.payload

    @pytest.mark.asyncio
    async def test_send_message_unknown_receiver(
        self, adapter: A2AProtocolAdapter, test_agents: tuple[ADKAgent, ADKAgent]
    ) -> None:
        """Test A2A message to unknown receiver returns error."""
        agent_a, _ = test_agents
        adapter.register_agent(agent_a)

        msg = A2AMessage(
            message_id="test-msg",
            message_type=MessageType.REQUEST,
            sender_agent_id="agent-a",
            receiver_agent_id="unknown-agent",
            payload={"message": "Hello"},
        )

        response = await adapter.send_message(msg)

        assert response.message_type == MessageType.ERROR
        assert "not found" in response.payload["error"]


class TestCogSynDeltaADKAgent:
    """Tests for CogSynDeltaADKAgent integration."""

    @pytest.fixture
    def agent(self, mock_integrated_system: MagicMock) -> CogSynDeltaADKAgent:
        """Create CogSynDelta ADK agent."""
        return CogSynDeltaADKAgent(
            agent_id="cogsyn-001", integrated_system=mock_integrated_system
        )

    def test_agent_identity(self, agent: CogSynDeltaADKAgent) -> None:
        """Test agent has correct identity."""
        assert agent.agent_id == "cogsyn-001"
        assert "CogSynDelta" in agent.name
        assert "self-improving" in agent.description.lower()

    def test_standard_tools_registered(self, agent: CogSynDeltaADKAgent) -> None:
        """Test standard tools are registered."""
        assert "process_visual_input" in agent.tools
        assert "generate_code" in agent.tools
        assert "query_memory" in agent.tools

    def test_capabilities_format(self, agent: CogSynDeltaADKAgent) -> None:
        """Test capabilities are in ADK format."""
        caps = agent.get_capabilities()
        assert len(caps) >= 3

        for cap in caps:
            assert cap["type"] == "function"
            assert "name" in cap["function"]
            assert "description" in cap["function"]
            assert "parameters" in cap["function"]

    def test_visual_processing_tool(self, agent: CogSynDeltaADKAgent) -> None:
        """Test visual processing tool execution."""
        msg = Message(
            role=AgentRole.USER,
            content="",
            function_call={
                "name": "process_visual_input",
                "arguments": {"image_data": "base64_encoded_image"},
            },
        )
        response = agent.process_message(msg)
        result = json.loads(response.content)
        assert result["status"] == "success"

    def test_code_generation_tool(self, agent: CogSynDeltaADKAgent) -> None:
        """Test code generation tool execution."""
        msg = Message(
            role=AgentRole.USER,
            content="",
            function_call={
                "name": "generate_code",
                "arguments": {
                    "problem_description": "Sort a list",
                    "language": "python",
                    "optimize": True,
                },
            },
        )
        response = agent.process_message(msg)
        result = json.loads(response.content)
        assert result["status"] == "success"
        assert result["language"] == "python"

    def test_memory_query_tool(self, agent: CogSynDeltaADKAgent) -> None:
        """Test memory query tool execution."""
        msg = Message(
            role=AgentRole.USER,
            content="",
            function_call={
                "name": "query_memory",
                "arguments": {"query": "previous solutions", "num_results": 3},
            },
        )
        response = agent.process_message(msg)
        result = json.loads(response.content)
        assert result["status"] == "success"

    def test_agent_without_integrated_system(self) -> None:
        """Test agent handles missing integrated system gracefully."""
        agent = CogSynDeltaADKAgent(agent_id="no-system", integrated_system=None)

        msg = Message(
            role=AgentRole.USER,
            content="",
            function_call={
                "name": "generate_code",
                "arguments": {"problem_description": "Test", "language": "python"},
            },
        )
        response = agent.process_message(msg)
        result = json.loads(response.content)
        assert "error" in result or result.get("status") == "success"


class TestADKAdapter:
    """Tests for ADKAdapter factory."""

    @pytest.fixture
    def adapter(self) -> ADKAdapter:
        """Create ADK adapter."""
        return ADKAdapter()

    def test_create_agent(
        self, adapter: ADKAdapter, mock_integrated_system: MagicMock
    ) -> None:
        """Test creating agent through adapter."""
        agent = adapter.create_agent("new-agent", mock_integrated_system)

        assert agent.agent_id == "new-agent"
        assert isinstance(agent, CogSynDeltaADKAgent)
        assert "new-agent" in adapter.agents

    def test_get_agent(
        self, adapter: ADKAdapter, mock_integrated_system: MagicMock
    ) -> None:
        """Test retrieving agent by ID."""
        adapter.create_agent("find-me", mock_integrated_system)

        found = adapter.get_agent("find-me")
        not_found = adapter.get_agent("nonexistent")

        assert found is not None
        assert found.agent_id == "find-me"
        assert not_found is None

    @pytest.mark.asyncio
    async def test_agent_to_agent_call(
        self, adapter: ADKAdapter, mock_integrated_system: MagicMock
    ) -> None:
        """Test A2A communication through adapter."""
        adapter.create_agent("sender", mock_integrated_system)
        adapter.create_agent("receiver", mock_integrated_system)

        result = await adapter.agent_to_agent_call(
            sender_id="sender", receiver_id="receiver", message="Hello!"
        )

        # Should return response content
        assert isinstance(result, str)

    def test_export_agent_spec(
        self, adapter: ADKAdapter, mock_integrated_system: MagicMock
    ) -> None:
        """Test exporting agent specification."""
        adapter.create_agent("export-me", mock_integrated_system)

        spec = adapter.export_agent_spec("export-me")

        assert spec["agent_id"] == "export-me"
        assert "name" in spec
        assert "description" in spec
        assert "capabilities" in spec

    def test_export_nonexistent_agent(self, adapter: ADKAdapter) -> None:
        """Test exporting spec for nonexistent agent returns empty."""
        spec = adapter.export_agent_spec("ghost")
        assert spec == {}


# Regression tests
class TestADKRegressions:
    """Regression tests for ADK adapter."""

    def test_message_role_string_values(self) -> None:
        """Regression: AgentRole must have lowercase string values per ADK spec."""
        for role in AgentRole:
            assert role.value == role.value.lower()

    def test_a2a_message_type_values(self) -> None:
        """Regression: MessageType must have snake_case values per A2A spec."""
        for msg_type in MessageType:
            assert msg_type.value == msg_type.value.lower()
            assert " " not in msg_type.value

    def test_capability_format_unchanged(self) -> None:
        """Regression: Capability format must match ADK function tool spec."""

        class TestAgent(ADKAgent):
            def _generate_response(self, prompt: str) -> str:
                return ""

        agent = TestAgent("test", "Test", "Test")
        agent.register_tool(
            name="test_tool",
            func=lambda: None,
            description="Test",
            parameters={"type": "object"},
        )

        caps = agent.get_capabilities()
        cap = caps[0]

        # Must have this exact structure per ADK spec
        assert cap["type"] == "function"
        assert "function" in cap
        assert "name" in cap["function"]
        assert "description" in cap["function"]
        assert "parameters" in cap["function"]
