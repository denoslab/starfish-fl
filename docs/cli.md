# CLI Reference

The Starfish-FL CLI (`starfish` command) replicates the web portal functionality for both human and AI agent use.

For the full CLI README with example workflows, see the [CLI README](https://github.com/denoslab/starfish-fl/blob/main/cli/README.md).

## Setup

```bash
cd cli
cp .env.example .env   # fill in your values
poetry install
poetry run starfish --help
```

## Configuration

| Variable | Description |
|----------|-------------|
| `SITE_UID` | Unique UUID for this site |
| `ROUTER_URL` | URL to the Starfish Router API |
| `ROUTER_USERNAME` | Router superuser username |
| `ROUTER_PASSWORD` | Router superuser password |
| `CONTROLLER_URL` | URL to this site's Controller (default: `http://localhost:8001`) |

!!! tip
    Use `STARFISH_ENV` environment variable to point to an alternate `.env` file (e.g., `STARFISH_ENV=.env.site2 poetry run starfish site info`).

## Commands

### site -- Manage site registration

| Command | Description |
|---------|-------------|
| `starfish site info` | Show current site info |
| `starfish site register` | Register this site with the router |
| `starfish site update` | Update site name and description |
| `starfish site deregister` | Deregister this site from the router |

### project -- Create, join, and manage projects

| Command | Description |
|---------|-------------|
| `starfish project list` | List all projects this site is involved in |
| `starfish project new` | Create a new project (this site becomes coordinator) |
| `starfish project join` | Join an existing project as a participant |
| `starfish project leave` | Leave a project using your participant ID |
| `starfish project detail` | Show detailed info for a project |

### run -- Start and monitor FL runs

| Command | Description |
|---------|-------------|
| `starfish run start` | Start a new FL run (coordinator only) |
| `starfish run status` | Show all runs and their statuses for a project |
| `starfish run detail` | Show detailed info for a specific run batch |
| `starfish run logs` | Fetch logs for a specific run |

### dataset -- Upload datasets

| Command | Description |
|---------|-------------|
| `starfish dataset upload` | Upload a dataset for a run |

### artifact -- Download results

| Command | Description |
|---------|-------------|
| `starfish artifact download` | Download artifacts or logs for a run |

## Example Workflow

```bash
# 1. Register sites
poetry run starfish site register --name "Hospital A" --desc "Site 1"
STARFISH_ENV=.env.site2 poetry run starfish site register --name "Hospital B" --desc "Site 2"

# 2. Create project (this site becomes coordinator)
poetry run starfish project new \
  --name "Breast Cancer Study" \
  --tasks '[{"seq":1,"model":"LogisticRegression","config":{"total_round":2,"current_round":1}}]'

# 3. Second site joins the project
STARFISH_ENV=.env.site2 poetry run starfish project join --name "Breast Cancer Study"

# 4. Upload datasets
poetry run starfish dataset upload --run-id 1 --file data/site1.csv
STARFISH_ENV=.env.site2 poetry run starfish dataset upload --run-id 2 --file data/site2.csv

# 5. Monitor progress
poetry run starfish run status --project-id 1

# 6. Download results
poetry run starfish artifact download --run-id 1 --type artifacts
```

## AI Agent

The CLI includes an AI agent that can autonomously orchestrate FL experiments using natural language. It wraps all CLI commands as LLM tools and uses Claude to plan and execute multi-step workflows.

### Setup

```bash
# Install with agent dependencies
poetry install --extras agent

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your-key-here
```

### agent -- AI-driven FL orchestration

| Command | Description |
|---------|-------------|
| `starfish agent run` | Run the AI agent with a natural language goal |
| `starfish agent tools` | List all available agent tools and their schemas |

### Agent Examples

```bash
# Run the agent with a natural language goal
poetry run starfish agent run "Register two sites and create a logistic regression project"

# Verbose mode shows tool calls and results
poetry run starfish agent run "Check status of project 1" --verbose

# List available tools in JSON format (useful for other LLM frameworks)
poetry run starfish agent tools --json
```

### Agent Options

| Option | Description |
|--------|-------------|
| `--model` / `-m` | Anthropic model to use (default: `claude-sonnet-4-6`) |
| `--api-key` / `-k` | Anthropic API key (default: `ANTHROPIC_API_KEY` env var) |
| `--max-turns` | Maximum agent turns (default: 50) |
| `--verbose` / `-v` | Show tool calls and results |
| `--json` | Output full conversation as JSON |

## JSON Output

All commands support `--json` for machine-readable output, making it easy for AI agents to parse responses.
