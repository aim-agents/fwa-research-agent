FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir .

# Install Playwright browsers
RUN playwright install chromium

# Expose the agent port
EXPOSE 9019

# Environment variables
ENV FWA_HOST=0.0.0.0
ENV FWA_PORT=9019
ENV FWA_MODEL=gpt-4o

# Run the agent
CMD ["python", "-m", "fwa_agent.server"]
