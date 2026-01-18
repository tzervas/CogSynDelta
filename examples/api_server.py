"""
API Server Example

This example demonstrates:
1. Starting the FastAPI server
2. Processing multimodal inputs
3. Using the agent system

To run this example:
    python examples/api_server.py

Then test with:
    curl http://localhost:8000/health
"""

import uvicorn

from cogsyndelta.api.server import app


def main() -> None:
    """Start the CogSynDelta API server."""
    print("=" * 60)
    print("CogSynDelta API Server")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    print("\nAvailable endpoints:")
    print("  GET  /health          - Health check")
    print("  GET  /                - API documentation")
    print("  POST /sessions        - Create multimodal session")
    print("  POST /agent/task      - Submit agent task")
    print("  GET  /adapters        - List available adapters")
    print("\n" + "=" * 60)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
