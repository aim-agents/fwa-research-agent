"""FWA Research Agent - A2A Purple Agent for FieldWorkArena Benchmark.

This agent handles three types of field work tasks:
1. Planning - Extract work procedures from documents and videos
2. Perception - Detect violations, classify incidents, spatial reasoning
3. Action - Execute plans, report incidents
"""

import asyncio
import base64
import json
import logging
import os
import sys
from io import BytesIO
from typing import Any

import httpx
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    DataPart,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from fwa_agent.config import AgentConfig

load_dotenv()

logger = logging.getLogger(__name__)

# Global config
config = AgentConfig()


class OpenAIClient:
    """Wrapper for OpenAI client with retry logic."""

    def __init__(self, config: AgentConfig):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.config = config
        self._cache: dict[str, str] = {}

    def _cache_key(self, messages: list, model: str) -> str:
        """Generate cache key from messages."""
        content = json.dumps(messages, sort_keys=True)
        return f"{model}:{hash(content)}"

    def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Create chat completion with retry logic."""
        model = model or self.config.model
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        # Check cache
        if self.config.cache_vision_results:
            cache_key = self._cache_key(messages, model)
            if cache_key in self._cache:
                logger.info("Cache hit for completion")
                return self._cache[cache_key]

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result = response.choices[0].message.content

                # Cache result
                if self.config.cache_vision_results and result:
                    self._cache[cache_key] = result

                return result or ""

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    import time
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise last_error or Exception("Failed after retries")


# Initialize client
openai_client: OpenAIClient | None = None


def get_openai_client() -> OpenAIClient:
    """Get or create OpenAI client."""
    global openai_client
    if openai_client is None:
        openai_client = OpenAIClient(config)
    return openai_client


class FWATaskHandler:
    """Handles FieldWorkArena tasks across Planning, Perception, and Action stages."""

    PLANNING_SYSTEM = """You are a field work planning expert for industrial environments (factories, warehouses, retail).

Analyze documents, images, and videos to extract work procedures. Focus on:
1. Key steps in work processes
2. Safety requirements and PPE needs
3. Required tools and equipment
4. Dependencies between steps
5. Time estimates

Output structured JSON with: procedure_name, steps (array with order, action, safety_notes, equipment_needed), estimated_duration_minutes, safety_requirements, dependencies."""

    PERCEPTION_SYSTEM = """You are a field work safety and compliance expert for {domain} environments.

Analyze images and reports to detect violations. Focus on:
1. Safety violations or hazards
2. PPE compliance (helmets, gloves, vests, etc.)
3. Spatial anomalies or unsafe arrangements
4. Incident classification by type and severity
5. Environmental conditions

Output structured JSON with: violations (type, severity, description), ppe_compliance (required, observed), incidents (type, severity, description), spatial_notes, confidence_score (0-1)."""

    ACTION_SYSTEM = """You are a field work operations agent executing work plans.

When executing tasks:
1. Follow procedures precisely
2. Document each action taken
3. Note deviations from expected outcomes
4. Generate clear, structured reports
5. Flag issues requiring attention

Output structured JSON with: actions_taken (step, result), observations, issues, report, status (completed/requires_attention), recommendations."""

    def __init__(self):
        self.client = get_openai_client()

    async def process_planning_task(self, task_data: dict) -> str:
        """Extract work procedures and understand workflows from documents/videos."""
        prompt = task_data.get("prompt", "")
        images = task_data.get("images", [])

        messages = [{"role": "system", "content": self.PLANNING_SYSTEM}]

        content = [{"type": "text", "text": prompt}]

        for img_b64 in images[:config.max_images_per_task]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": config.image_detail},
            })

        messages.append({"role": "user", "content": content})

        return self.client.chat_completion(messages)

    async def process_perception_task(self, task_data: dict) -> str:
        """Detect safety violations, classify incidents, perform spatial reasoning."""
        prompt = task_data.get("prompt", "")
        images = task_data.get("images", [])
        domain = task_data.get("domain", "factory")

        system = self.PERCEPTION_SYSTEM.format(domain=domain)
        messages = [{"role": "system", "content": system}]

        content = [{"type": "text", "text": prompt}]

        for img_b64 in images[:config.max_images_per_task]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": config.perception_image_detail},
            })

        messages.append({"role": "user", "content": content})

        return self.client.chat_completion(messages)

    async def process_action_task(self, task_data: dict) -> str:
        """Execute plans and decisions, analyze observations and report incidents."""
        prompt = task_data.get("prompt", "")
        images = task_data.get("images", [])
        context = task_data.get("context", "")

        messages = [{"role": "system", "content": self.ACTION_SYSTEM}]

        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})

        content = [{"type": "text", "text": prompt}]

        for img_b64 in images[:config.max_images_per_task]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": config.image_detail},
            })

        messages.append({"role": "user", "content": content})

        return self.client.chat_completion(messages)

    def classify_task(self, task_text: str) -> str:
        """Classify task into Planning, Perception, or Action stage."""
        task_lower = task_text.lower()

        planning_keywords = ["plan", "procedure", "workflow", "extract", "document", "video", "protocol", "steps"]
        perception_keywords = ["detect", "identify", "classify", "violation", "safety", "ppe", "incident", "spatial", "hazard"]
        action_keywords = ["execute", "report", "perform", "action", "submit", "complete", "carry out"]

        scores = {
            "planning": sum(1 for kw in planning_keywords if kw in task_lower),
            "perception": sum(1 for kw in perception_keywords if kw in task_lower),
            "action": sum(1 for kw in action_keywords if kw in task_lower),
        }

        # Default to perception if no clear match
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "perception"


class FWAPurpleAgentExecution(AgentExecutor):
    """A2A Agent Execution handler for FieldWorkArena purple agent."""

    def __init__(self):
        self.task_handler = FWATaskHandler()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a FieldWorkArena task."""
        task_id = context.task_id
        logger.info(f"Executing FWA task: {task_id}")

        message = context.message
        if not message or not message.parts:
            await event_queue.enqueue_event(
                new_agent_text_message("No task provided.", context_id=context.context_id, task_id=task_id)
            )
            return

        # Extract task from last message
        last_message = message
        task_text = ""
        images = []

        for part in last_message.parts:
            if isinstance(part.root, TextPart):
                task_text = part.root.text
            elif isinstance(part.root, DataPart):
                data = part.root.data
                if "images" in data:
                    images = data["images"]
                if "prompt" in data:
                    task_text = data["prompt"]

        if not task_text:
            await event_queue.enqueue_event(
                new_agent_text_message("Could not extract task from message.", context_id=context.context_id, task_id=task_id)
            )
            return

        # Classify and process
        task_type = self.task_handler.classify_task(task_text)
        task_data = {"prompt": task_text, "images": images, "domain": "factory"}

        try:
            if task_type == "planning":
                result = await self.task_handler.process_planning_task(task_data)
            elif task_type == "perception":
                result = await self.task_handler.process_perception_task(task_data)
            else:
                result = await self.task_handler.process_action_task(task_data)

            await event_queue.enqueue_event(
                new_agent_text_message(result, context_id=context.context_id, task_id=task_id)
            )

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {str(e)}", context_id=context.context_id, task_id=task_id)
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel is not supported."""
        raise Exception("Cancel not supported")


def create_agent_card(host: str, port: int) -> AgentCard:
    """Create the agent card for A2A server."""
    return AgentCard(
        name="FWA Research Agent",
        description=(
            "A2A Purple Agent for FieldWorkArena benchmark. "
            "Handles Planning, Perception, and Action tasks in factory, warehouse, and retail environments."
        ),
        url=f"http://{host}:{port}",
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="planning",
                name="Task Planning",
                description="Extract work procedures and workflows from documents and videos",
                tags=["planning", "procedures", "workflow"],
            ),
            AgentSkill(
                id="perception",
                name="Safety Perception",
                description="Detect safety violations, classify incidents, spatial reasoning",
                tags=["safety", "perception", "incidents"],
            ),
            AgentSkill(
                id="action",
                name="Field Action",
                description="Execute plans, analyze observations, report incidents",
                tags=["action", "reporting", "execution"],
            ),
        ],
    )


def main():
    """Start the FWA Research Agent A2A server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Validate config
    issues = config.validate()
    if issues:
        for issue in issues:
            logger.warning(f"Config issue: {issue}")
        if not config.is_configured:
            logger.error("Cannot start: OPENAI_API_KEY is required")
            sys.exit(1)

    from a2a.server.tasks import InMemoryTaskStore

    logger.info(f"Starting FWA Research Agent on {config.host}:{config.port}")
    logger.info(f"Using model: {config.model}")

    agent_card = create_agent_card(config.host, config.port)
    execution = FWAPurpleAgentExecution()
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(agent_executor=execution, task_store=task_store)

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

    uvicorn.run(server.build(), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
