"""
CLI entry point for the agent subcommand: `starfish agent run`
"""

import json
import typer
from typing import Optional

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    goal: str = typer.Argument(..., help="Natural language goal for the agent"),
    model: str = typer.Option(
        "claude-sonnet-4-6",
        "--model", "-m",
        help="Anthropic model to use"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key", "-k",
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
        envvar="ANTHROPIC_API_KEY",
    ),
    max_turns: int = typer.Option(
        50,
        "--max-turns",
        help="Maximum number of agent turns"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show tool calls and results"
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output full conversation as JSON"
    ),
):
    """Run the AI agent to orchestrate FL experiments using natural language."""
    try:
        from anthropic import Anthropic
    except ImportError:
        typer.echo(
            "Error: anthropic package not installed. "
            "Install it with: poetry install --extras agent"
        )
        raise typer.Exit(code=1)

    from starfish_cli.agent.agent import run_agent_loop, extract_final_response

    if not api_key:
        typer.echo("Error: ANTHROPIC_API_KEY not set. Provide via --api-key or environment variable.")
        raise typer.Exit(code=1)

    if verbose:
        typer.echo(f"Agent goal: {goal}")
        typer.echo(f"Model: {model}")
        typer.echo("---")

    messages = run_agent_loop(
        goal=goal,
        model=model,
        api_key=api_key,
        max_turns=max_turns,
        verbose=verbose,
    )

    if json_output:
        # Serialize the conversation — convert API objects to dicts
        serializable = []
        for msg in messages:
            if msg["role"] == "assistant":
                content = []
                for block in msg["content"]:
                    if hasattr(block, "text"):
                        content.append({"type": "text", "text": block.text})
                    elif hasattr(block, "name"):
                        content.append({
                            "type": "tool_use",
                            "name": block.name,
                            "input": block.input,
                        })
                serializable.append({"role": "assistant", "content": content})
            else:
                serializable.append(msg)
        typer.echo(json.dumps(serializable, indent=2))
    else:
        final = extract_final_response(messages)
        if final:
            typer.echo(final)
        else:
            typer.echo("Agent completed without a final text response.")


@app.command()
def experiment(
    goal: str = typer.Argument(..., help="Describe the experiment (include dataset paths and research goal)"),
    model: str = typer.Option(
        "claude-sonnet-4-6",
        "--model", "-m",
        help="Anthropic model to use"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key", "-k",
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
        envvar="ANTHROPIC_API_KEY",
    ),
    max_turns: int = typer.Option(
        100,
        "--max-turns",
        help="Maximum number of agent turns (default: 100)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show tool calls and results"
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output full conversation as JSON"
    ),
):
    """Run an autonomous FL experiment: analyze data, select model, train, and interpret results."""
    try:
        from anthropic import Anthropic
    except ImportError:
        typer.echo(
            "Error: anthropic package not installed. "
            "Install it with: poetry install --extras agent"
        )
        raise typer.Exit(code=1)

    from starfish_cli.agent.agent import run_agent_loop, extract_final_response
    from starfish_cli.agent.prompts import EXPERIMENT_SYSTEM_PROMPT
    from starfish_cli.agent.tools import get_experiment_tool_schemas

    if not api_key:
        typer.echo("Error: ANTHROPIC_API_KEY not set. Provide via --api-key or environment variable.")
        raise typer.Exit(code=1)

    if verbose:
        typer.echo(f"Experiment goal: {goal}")
        typer.echo(f"Model: {model}")
        typer.echo("Mode: autonomous experiment")
        typer.echo("---")

    messages = run_agent_loop(
        goal=goal,
        model=model,
        api_key=api_key,
        max_turns=max_turns,
        verbose=verbose,
        system_prompt=EXPERIMENT_SYSTEM_PROMPT,
        tools=get_experiment_tool_schemas(),
    )

    if json_output:
        serializable = []
        for msg in messages:
            if msg["role"] == "assistant":
                content = []
                for block in msg["content"]:
                    if hasattr(block, "text"):
                        content.append({"type": "text", "text": block.text})
                    elif hasattr(block, "name"):
                        content.append({
                            "type": "tool_use",
                            "name": block.name,
                            "input": block.input,
                        })
                serializable.append({"role": "assistant", "content": content})
            else:
                serializable.append(msg)
        typer.echo(json.dumps(serializable, indent=2))
    else:
        final = extract_final_response(messages)
        if final:
            typer.echo(final)
        else:
            typer.echo("Experiment completed without a final text response.")


@app.command()
def tools(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    all_tools: bool = typer.Option(False, "--all", help="Include experiment-specific tools"),
):
    """List all available agent tools and their schemas."""
    from starfish_cli.agent.tools import get_tool_schemas, get_experiment_tool_schemas

    schemas = get_experiment_tool_schemas() if all_tools else get_tool_schemas()

    if json_output:
        typer.echo(json.dumps(schemas, indent=2))
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    title = "Starfish Agent Tools (all)" if all_tools else "Starfish Agent Tools"
    table = Table(title=title)
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description")
    table.add_column("Required Params", style="green")

    for tool in schemas:
        required = tool["input_schema"].get("required", [])
        table.add_row(
            tool["name"],
            tool["description"][:80],
            ", ".join(required) if required else "none",
        )
    console.print(table)
