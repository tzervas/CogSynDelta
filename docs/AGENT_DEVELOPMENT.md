# Agent Development Guide

Guide for developing custom agents compatible with CogSynDelta.

## Google ADK Compliance

All agents must comply with Google Agent Development Kit standards.

### Basic Agent Structure

```python
from google_adk_adapter import ADKAgent, AgentRole, Message

class MyCustomAgent(ADKAgent):
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            name="My Custom Agent",
            description="Description of what this agent does"
        )
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        self.register_tool(
            name="my_tool",
            func=self._my_tool_impl,
            description="Tool description",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": ["param1"]
            }
        )
    
    def _my_tool_impl(self, param1: str) -> dict:
        # Tool implementation
        return {"result": f"Processed: {param1}"}
    
    def _generate_response(self, prompt: str) -> str:
        # Response generation logic
        return f"Response to: {prompt}"
```

### Registering with CogSynDelta

```python
from google_adk_adapter import ADKAdapter
from integrated_system import create_integrated_system

# Create integrated system
system = create_integrated_system()

# Create adapter
adapter = ADKAdapter()

# Create and register agent
agent = MyCustomAgent("my_agent_1")
adapter.a2a_adapter.register_agent(agent)
```

## Tool Development

### Tool Specification

Tools must follow JSON Schema format:

```python
{
    "type": "object",
    "properties": {
        "parameter_name": {
            "type": "string|number|boolean|array|object",
            "description": "Parameter description",
            "enum": ["value1", "value2"],  # Optional
            "default": "default_value"  # Optional
        }
    },
    "required": ["parameter1", "parameter2"]
}
```

### Tool Implementation

```python
def my_tool(param1: str, param2: int = 10) -> dict:
    """
    Tool description.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
        
    Returns:
        Dictionary with result
    """
    # Implementation
    result = process(param1, param2)
    
    return {
        "status": "success",
        "result": result,
        "metadata": {}
    }
```

## Agent-to-Agent Communication

### Sending Messages

```python
from google_adk_adapter import A2AMessage, MessageType
import uuid

# Create A2A message
message = A2AMessage(
    message_id=str(uuid.uuid4()),
    message_type=MessageType.REQUEST,
    sender_agent_id="agent_1",
    receiver_agent_id="agent_2",
    payload={"query": "Process this data"}
)

# Send via adapter
response = await adapter.a2a_adapter.send_message(message)
```

### Broadcasting

```python
# Broadcast to all agents
responses = await adapter.a2a_adapter.broadcast_message(
    message,
    exclude=["agent_1"]  # Exclude sender
)
```

## Integration with CogSynDelta Features

### Using Memory System

```python
def my_agent_with_memory(self):
    # Access persistent memory
    if hasattr(self, 'integrated_system'):
        memory_bank = self.integrated_system.memory_bank
        
        # Write to memory
        embedding = torch.randn(1, 512)
        memory_bank.write(embedding, importance=0.9)
        
        # Query memory
        query = torch.randn(1, 512)
        retrieved, metadata = memory_bank.read(query, num_reads=5)
```

### Using Model Sections

```python
def process_with_sections(self, input_data):
    # Access sectioned model
    sectioned_model = self.integrated_system.sectioned_model
    
    # Process through specific sections
    outputs = sectioned_model.forward(
        input_data={"visual": input_data},
        required_sections=["visual", "prefrontal"]
    )
    
    return outputs
```

### Using Quantum Backend

```python
from quantum_compute import ComputeJob, ComputeBackendType
import uuid

def quantum_processing(self, data):
    # Create quantum job
    job = ComputeJob(
        job_id=str(uuid.uuid4()),
        backend_type=ComputeBackendType.QUANTUM_GATE,
        operation="quantum_forward",
        input_data=data,
        parameters={'shots': 1024}
    )
    
    # Submit to orchestrator
    result = await orchestrator.submit_job(job)
    return result
```

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
def my_tool(self, param):
    try:
        result = process(param)
        return {"status": "success", "result": result}
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"Unexpected error: {str(e)}"}
```

### 2. Input Validation

Validate inputs before processing:

```python
def my_tool(self, param: str):
    if not param or not isinstance(param, str):
        return {"status": "error", "error": "Invalid parameter"}
    
    if len(param) > 1000:
        return {"status": "error", "error": "Parameter too long"}
    
    # Process...
```

### 3. Documentation

Document all tools and methods:

```python
def my_tool(self, param1: str, param2: int) -> dict:
    """
    Brief description.
    
    Detailed description of what this tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - result: Processed result
        - metadata: Additional information
        
    Examples:
        >>> result = my_tool("test", 10)
        >>> print(result['status'])
        success
    """
```

### 4. Testing

Test agents thoroughly:

```python
def test_my_agent():
    agent = MyCustomAgent("test_agent")
    
    # Test tool calling
    message = Message(
        role=AgentRole.USER,
        content="Test",
        function_call={
            "name": "my_tool",
            "arguments": {"param1": "test"}
        }
    )
    
    response = agent.process_message(message)
    assert response.role == AgentRole.FUNCTION
    
    # Test response generation
    message = Message(
        role=AgentRole.USER,
        content="Hello"
    )
    
    response = agent.process_message(message)
    assert response.role == AgentRole.ASSISTANT
```

### 5. Safeguards

Implement safeguards in agents:

```python
from memory_persistence import InfiniteLoopSafeguard

class SafeAgent(ADKAgent):
    def __init__(self, agent_id):
        super().__init__(agent_id, "Safe Agent", "Description")
        self.safeguard = InfiniteLoopSafeguard(
            max_iterations=100,
            max_repetitions=3
        )
    
    def _generate_response(self, prompt):
        # Check safeguards
        state = torch.tensor([hash(prompt)])
        is_safe, message = self.safeguard.check_state(state)
        
        if not is_safe:
            return f"Safeguard triggered: {message}"
        
        # Generate response...
```

## Publishing Agents

### Package Structure

```
my_agent/
├── __init__.py
├── agent.py
├── tools/
│   ├── __init__.py
│   └── my_tools.py
├── tests/
│   └── test_agent.py
├── README.md
└── requirements.txt
```

### Installation

```bash
pip install -e .
```

### Usage

```python
from my_agent import MyCustomAgent
from google_adk_adapter import ADKAdapter

adapter = ADKAdapter()
agent = adapter.create_agent("my_agent_1", MyCustomAgent)
```

## Examples

See `examples/agents/` directory for complete agent implementations.
