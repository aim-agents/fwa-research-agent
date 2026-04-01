"""Configuration management for FWA Research Agent."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for the FWA Research Agent."""

    # Server settings
    host: str = field(default_factory=lambda: os.getenv("FWA_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("FWA_PORT", "9019")))

    # OpenAI settings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("FWA_MODEL", "gpt-4o"))
    max_tokens: int = 2048
    temperature: float = 0.1

    # HuggingFace settings
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    # Green Agent settings
    green_agent_url: str = field(
        default_factory=lambda: os.getenv("GREEN_AGENT_URL", "http://127.0.0.1:9009")
    )

    # Task settings
    max_images_per_task: int = 4
    image_detail: str = "low"  # "low" or "high" - "low" saves tokens
    perception_image_detail: str = "high"  # Higher detail for safety tasks

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Cost optimization
    cache_vision_results: bool = True

    @property
    def is_configured(self) -> bool:
        """Check if minimum required config is present."""
        return bool(self.openai_api_key)

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.openai_api_key:
            issues.append("OPENAI_API_KEY is required")

        if not self.hf_token:
            issues.append("HF_TOKEN is recommended for full dataset access")

        if self.port < 1 or self.port > 65535:
            issues.append(f"Invalid port: {self.port}")

        return issues
