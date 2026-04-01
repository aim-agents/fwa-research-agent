FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir .

# Install Playwright browsers (if needed for browser tasks)
RUN playwright install chromium || true

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9019/health || exit 1

# Expose the agent port
EXPOSE 9019

# Environment variables (override at runtime)
ENV FWA_HOST=0.0.0.0
ENV FWA_PORT=9019
ENV FWA_MODEL=gpt-4o
ENV PYTHONUNBUFFERED=1

# Run the agent
CMD ["python", "-m", "fwa_agent.server"]
