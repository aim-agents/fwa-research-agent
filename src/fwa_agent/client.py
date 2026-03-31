"""A2A Client for testing against FieldWorkArena Green Agent."""

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GreenAgentClient:
    """Client to communicate with FieldWorkArena Green Agent via A2A protocol."""

    def __init__(self, green_agent_url: str | None = None):
        self.green_agent_url = green_agent_url or os.getenv(
            "GREEN_AGENT_URL", "http://127.0.0.1:9009"
        )
        self.client = httpx.AsyncClient(timeout=120.0)

    async def send_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Send a task to the Green Agent and get response.

        Args:
            task_data: Task data including prompt, images, domain

        Returns:
            Response from Green Agent
        """
        url = f"{self.green_agent_url}/a2a/tasks"

        try:
            response = await self.client.post(url, json=task_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error sending task: {e.response.status_code}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error sending task: {e}")
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check if Green Agent is available.

        Returns:
            True if agent is healthy
        """
        try:
            response = await self.client.get(f"{self.green_agent_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class LocalTestRunner:
    """Run local tests against the FieldWorkArena Green Agent."""

    def __init__(self, purple_agent_url: str = "http://127.0.0.1:9019"):
        self.purple_agent_url = purple_agent_url
        self.green_agent_url = os.getenv(
            "GREEN_AGENT_URL", "http://127.0.0.1:9009"
        )

    async def run_scenario(self, scenario_path: str) -> dict[str, Any]:
        """Run a local test scenario.

        Args:
            scenario_path: Path to scenario.toml file

        Returns:
            Test results
        """
        import tomllib

        with open(scenario_path, "rb") as f:
            scenario = tomllib.load(f)

        target = scenario.get("config", {}).get("target", "factory")

        logger.info(f"Running scenario for target: {target}")

        # This would integrate with the Green Agent's assessment flow
        # For now, return placeholder
        return {
            "status": "ready",
            "target": target,
            "purple_agent": self.purple_agent_url,
            "green_agent": self.green_agent_url,
        }

    async def test_task_classification(self) -> list[dict]:
        """Test task classification with sample tasks.

        Returns:
            List of test results
        """
        from fwa_agent.task_processor import TaskClassifier, TaskStage

        test_cases = [
            ("Extract the safety procedure from this video", TaskStage.PLANNING),
            ("Detect any PPE violations in this image", TaskStage.PERCEPTION),
            ("Report the incident findings", TaskStage.ACTION),
            ("Analyze the warehouse layout for hazards", TaskStage.PERCEPTION),
            ("Create a workflow plan for the shift", TaskStage.PLANNING),
        ]

        results = []
        for text, expected in test_cases:
            classified = TaskClassifier.classify(text)
            results.append({
                "input": text,
                "expected": expected.value,
                "actual": classified.value,
                "correct": classified == expected,
            })

        return results
