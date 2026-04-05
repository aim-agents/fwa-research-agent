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

    # LLM Provider settings
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))  # openai, openrouter, groq
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    
    # Model settings
    model: str = field(default_factory=lambda: os.getenv("FWA_MODEL", "gpt-4o"))
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    
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
        if self.llm_provider == "openrouter":
            return bool(self.openrouter_api_key)
        elif self.llm_provider == "groq":
            return bool(self.groq_api_key)
        return bool(self.openai_api_key)

    def get_model_name(self) -> str:
        """Get the actual model name based on provider."""
        if self.llm_provider == "openrouter":
            return self.openrouter_model
        elif self.llm_provider == "groq":
            return self.groq_model
        return self.model

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if self.llm_provider == "openrouter":
            if not self.openrouter_api_key:
                issues.append("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        elif self.llm_provider == "groq":
            if not self.groq_api_key:
                issues.append("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        else:
            if not self.openai_api_key:
                issues.append("OPENAI_API_KEY is required")

        if not self.hf_token:
            issues.append("HF_TOKEN is recommended for full dataset access")

        if self.port < 1 or self.port > 65535:
            issues.append(f"Invalid port: {self.port}")

        return issues
