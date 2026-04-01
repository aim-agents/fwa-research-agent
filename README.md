# FWA Research Agent

A2A Purple Agent for the [FieldWorkArena](https://github.com/FujitsuResearch/FieldWorkArena) benchmark - competing in AgentX AgentBeats Sprint 2 (Research Agent track).

## Overview

FieldWorkArena evaluates AI agents on real-world field work tasks in industrial environments (factories, warehouses, retail). The agent must handle three types of tasks:

1. **Planning** - Extract work procedures from documents and videos
2. **Perception** - Detect safety violations, classify incidents, spatial reasoning
3. **Action** - Execute plans, analyze observations, report incidents

### Available Tasks

- **Factory**: 79 tasks (safety, equipment, workflow)
- **Warehouse**: 155 tasks (inventory, logistics, incidents)
- **Retail**: 5 tasks (customer service, compliance)

## Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/aim-agents/fwa-research-agent.git
cd fwa-research-agent

# Quick setup
./setup.sh

# Or manual setup
pip install -e .
cp .env.sample .env
# Edit .env with your API keys

# Run the agent
python -m fwa_agent.server
```

### Docker

```bash
# Build
docker build -t fwa-research-agent .

# Run
docker run -p 9019:9019 --env-file .env fwa-research-agent
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `HF_TOKEN` | HuggingFace token for dataset access | Recommended |
| `FWA_HOST` | Server host | `0.0.0.0` |
| `FWA_PORT` | Server port | `9019` |
| `FWA_MODEL` | OpenAI model to use | `gpt-4o` |

## Architecture

```
src/fwa_agent/
├── server.py          # A2A server (FastAPI + uvicorn)
├── task_processor.py  # Task classification & routing
├── config.py          # Configuration management
├── retrieval.py       # arXiv/Semantic Scholar integration
├── experiment.py      # Experiment design module
├── video_processor.py # Video frame extraction
└── client.py          # Green Agent test client
```

## Competition Details

- **Track**: Research Agent (FieldWorkArena)
- **Deadline**: April 12, 2026
- **Platform**: [AgentBeats](https://agentbeats.dev)
- **Leaderboard**: [FieldWorkArena Leaderboard](https://agentbeats.dev/agentbeater/fieldworkarena)
- **Green Agent**: [FieldWorkArena-GreenAgent](https://github.com/ast-fri/FieldWorkArena-GreenAgent)

## Judging Criteria

1. **Leaderboard Performance** - Scores from green agent evaluations
2. **Generality** - Performance across factory, warehouse, and retail tasks
3. **Cost Efficiency** - Resource usage (API calls, tokens)
4. **Technical Quality** - Code maintainability and documentation
5. **Innovation** - Novel approaches and architectures

## License

Apache-2.0
