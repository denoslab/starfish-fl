# Starfish-FL CLI

A command-line interface for [Starfish-FL](https://github.com/denoslab/starfish-fl) that replicates the web portal functionality. Designed to be used by both humans and AI agents (e.g. OpenClaw).

## Requirements

- Python 3.10+
- Poetry
- Starfish-FL stack running (see [workbench setup](../workbench/README.md))

## Setup
```bash
cd cli
cp .env.example .env        # fill in your values
poetry install
poetry run starfish --help
```

## Configuration

Edit `.env` with your values:

| Variable | Description |
|---|---|
| `SITE_UID` | Unique UUID for this site |
| `ROUTER_URL` | URL to the Starfish Router API |
| `ROUTER_USERNAME` | Router superuser username |
| `ROUTER_PASSWORD` | Router superuser password |
| `CONTROLLER_URL` | URL to this site's Controller (default: `http://localhost:8001`) |

Generate a UUID with:
```bash
python3 -c "import uuid; print(uuid.uuid4())"
```

## Commands
```bash
starfish site      info / register / update / deregister
starfish project   list / new / join / leave / detail
starfish run       start / status / detail / logs
starfish dataset   upload
starfish artifact  download
starfish agent     run / experiment / tools
```

## AI Agent

The CLI includes an AI agent that can autonomously orchestrate FL experiments using natural language. It wraps all CLI commands as LLM tools and uses Claude to plan and execute multi-step workflows.

### Agent Setup
```bash
# Install with agent dependencies
poetry install --extras agent

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your-key-here
```

### Agent Commands
```bash
# Run the agent with a natural language goal
poetry run starfish agent run "Register two sites and create a logistic regression project with 3 rounds"

# Run with verbose output to see tool calls
poetry run starfish agent run "Check the status of project 1" --verbose

# List all available agent tools
poetry run starfish agent tools
poetry run starfish agent tools --json

# Use a different model
poetry run starfish agent run "Monitor my runs" --model claude-haiku-4-5-20251001
```

### Agent Options
| Option | Description |
|--------|-------------|
| `--model` / `-m` | Anthropic model to use (default: `claude-sonnet-4-6`) |
| `--api-key` / `-k` | Anthropic API key (default: `ANTHROPIC_API_KEY` env var) |
| `--max-turns` | Maximum agent turns (default: 50 for `run`, 100 for `experiment`) |
| `--verbose` / `-v` | Show tool calls and results |
| `--json` | Output full conversation as JSON |

### Autonomous Experiment Mode

The `experiment` command provides a fully autonomous FL experiment workflow. Given dataset paths and a research goal, the agent will analyze the data, select the appropriate model, configure the project, run the experiment end-to-end, and interpret results — iterating to find the best approach.

```bash
# Run an autonomous experiment
poetry run starfish agent experiment \
  "Analyze the dataset at /data/site1.csv and /data/site2.csv, \
   choose the best model, run a federated learning experiment, \
   and interpret the results" --verbose

# Survival analysis experiment
poetry run starfish agent experiment \
  "Run a survival analysis on /data/veteran_site1.csv and /data/veteran_site2.csv \
   using two sites. The data has time and event columns." --verbose
```

#### Experiment Tools

In addition to all CLI tools, the experiment agent has access to local analysis tools:

| Tool | Description |
|------|-------------|
| `analyze_dataset` | Analyze CSV structure, column types, and detected patterns |
| `recommend_task` | Recommend FL task types based on data analysis |
| `generate_config` | Generate task config JSON for a model with correct defaults |
| `interpret_results` | Parse and interpret experiment artifact files |
| `compare_experiments` | Compare results across multiple experiments and rank models |

List all tools (including experiment tools) with:
```bash
poetry run starfish agent tools --all
```

## Example workflow would look something like
```bash
# 1. Register sites
poetry run starfish site register --name "Hospital A" --desc "Site 1"
STARFISH_ENV=.env.site2 poetry run starfish site register --name "Hospital B" --desc "Site 2"

# 2. Create project (this site becomes coordinator)
poetry run starfish project new \
  --name "Breast Cancer Study" \
  --tasks '[{"seq":1,"model":"LogisticRegression","config":{"total_round":2,"current_round":1}}]'

