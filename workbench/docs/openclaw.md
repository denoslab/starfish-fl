# OpenClaw Integration

OpenClaw is an LLM-powered agent framework integrated into the Starfish workbench. It provides agentic capabilities for intelligent orchestration and autonomous federated learning workflows.

## Setup

OpenClaw is automatically included when you run the Starfish workbench with Docker Compose:

```bash
cd workbench
make up
```

The compose stack will:
1. Build the OpenClaw Docker image from `Dockerfile.openclaw`
2. Start the OpenClaw gateway service on port 18789
3. Mount the Starfish CLI workspace for autonomous orchestration capabilities

## Accessing OpenClaw

Once the stack is running, access the OpenClaw UI:
- **URL**: http://localhost:18789

## Important: UI Responsiveness

The OpenClaw UI may take a moment to become fully responsive after startup. 

**Wait for the first message to appear in the chat section from OpenClaw's side before proceeding.** Once you see this initial message, the agent is ready and fully operational.

This initial delay is normal and occurs during OpenClaw's plugin initialization and canvas setup.

## Architecture

OpenClaw is configured with:
- **Starfish CLI**: Starfish CLI is mounted at `/home/node/.openclaw/workspace/starfish-cli`. The OpenClaw image includes Starfish CLI agent dependencies
- **API access**: Available via http://127.0.0.1:18789 (internal) and http://localhost:18789 (external)

## Configuration

OpenClaw configuration is managed in `config/openclaw.env`:
- `OPENCLAW_GATEWAY_BIND`: Network binding (default: lan)
- `OPENCLAW_GATEWAY_TOKEN`: Authentication token for gateway API
- `OPENAI_API_KEY`: LLM provider credentials

The compose file also sets:
- `OPENCLAW_CONTROL_UI_ALLOWED_ORIGINS`: WebSocket origins for UI connections
