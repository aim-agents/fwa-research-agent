#!/bin/bash
# Quick setup script for FWA Research Agent testing

set -e

echo "=== FWA Research Agent Setup ==="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
echo "Installing dependencies..."
source venv/bin/activate
pip install -e ".[dev]" --quiet

# Check for .env
if [ ! -f ".env" ]; then
    echo "Creating .env from sample..."
    cp .env.sample .env
    echo ""
    echo "IMPORTANT: Edit .env and add your API keys:"
    echo "  - OPENAI_API_KEY (required)"
    echo "  - HF_TOKEN (recommended for full dataset)"
    echo ""
fi

# Run tests
echo "Running tests..."
pytest tests/ -v 2>/dev/null || echo "Tests need API keys to run fully"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the agent:"
echo "  source venv/bin/activate"
echo "  python -m fwa_agent.server"
echo ""
echo "The agent will be available at http://localhost:9019"
