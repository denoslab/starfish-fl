"""
Main agent loop using the Anthropic Python SDK with tool use.
"""

import json
from anthropic import Anthropic

from starfish_cli.agent.tools import get_tool_schemas, execute_tool
from starfish_cli.agent.prompts import SYSTEM_PROMPT

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TURNS = 50


def create_client(api_key: str | None = None) -> Anthropic:
    """Create an Anthropic client. Uses ANTHROPIC_API_KEY env var if no key provided."""
    if api_key:
        return Anthropic(api_key=api_key)
    return Anthropic()


def run_agent_loop(
    goal: str,
    client: Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_turns: int = MAX_TURNS,
    verbose: bool = False,
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
) -> list[dict]:
    """
    Run the agent loop to accomplish a user-specified goal.

    Parameters
    ----------
    goal : str
        Natural language description of what the agent should do.
    client : Anthropic | None
        Pre-configured Anthropic client. If None, creates one.
    model : str
        Model to use for the agent.
    api_key : str | None
        Anthropic API key. Uses env var if not provided.
    max_turns : int
        Maximum number of LLM turns before stopping.
    verbose : bool
        If True, print each tool call and result.
    system_prompt : str | None
        Custom system prompt. If None, uses the default SYSTEM_PROMPT.
    tools : list[dict] | None
        Custom tool schemas. If None, uses get_tool_schemas() (CLI tools only).

    Returns
    -------
    list[dict]
        Full conversation history.
    """
    if client is None:
        client = create_client(api_key)

    effective_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    effective_tools = tools if tools is not None else get_tool_schemas()
    messages = [{"role": "user", "content": goal}]

    for turn in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=effective_prompt,
            tools=effective_tools,
            messages=messages,
        )

        # Collect assistant content blocks
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Check if the model wants to use tools
        tool_uses = [block for block in assistant_content if block.type == "tool_use"]

        if not tool_uses:
            # No tool calls — agent is done, extract final text
            break

        # Execute each tool call and collect results
        tool_results = []
        for tool_use in tool_uses:
            if verbose:
                print(f"\n[Tool Call] {tool_use.name}")
                print(f"  Input: {json.dumps(tool_use.input, indent=2)}")

            result = execute_tool(tool_use.name, tool_use.input)

            if verbose:
                print(f"  Result: {result[:500]}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return messages


def extract_final_response(messages: list[dict]) -> str:
    """Extract the final text response from the agent conversation."""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg["content"]
            text_parts = []
            for block in content:
                if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                    text_parts.append(block.text)
            if text_parts:
                return "\n".join(text_parts)
    return ""
