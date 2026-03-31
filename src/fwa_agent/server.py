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
from a2a.server.agent_execution import AgentExecution, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handler import DefaultRequestHandler
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

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class FWATaskHandler:
    """Handles FieldWorkArena tasks across Planning, Perception, and Action stages."""

    def __init__(self):
        self.model = os.getenv("FWA_MODEL", "gpt-4o")

    async def process_planning_task(self, task_data: dict) -> str:
        """Extract work procedures and understand workflows from documents/videos."""
        prompt = task_data.get("prompt", "")
        images = task_data.get("images", [])
        videos = task_data.get("videos", [])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a field work planning expert. Analyze documents, images, and videos "
                    "to extract work procedures, safety protocols, and operational workflows. "
                    "Provide structured, actionable plans."
                ),
            }
        ]

        content = [{"type": "text", "text": prompt}]

        # Add images if present
        for img_b64 in images[:4]:  # Limit to 4 images for cost efficiency
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"},
                }
            )

        messages.append({"role": "user", "content": content})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )

        return response.choices[0].message.content

    async def process_perception_task(self, task_data: dict) -> str:
        """Detect safety violations, classify incidents, perform spatial reasoning."""
        prompt = task_data.get("prompt", "")
        images = task_data.get("images", [])
        domain = task_data.get("domain", "factory")

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a field work safety and compliance expert specializing in {domain} environments. "
                    "Analyze images and reports to detect safety violations, PPE compliance, "
                    "incidents, and spatial anomalies. Provide structured findings with confidence scores."
                ),
            }
        ]

        content = [{"type": "text", "text": prompt}]

        for img_b64 in images[:4]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"},
                }
            )

        messages.append({"role": "user", "content": content})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )

        return response.choices[0].message.content

    async def process_action_task(self, task_data: dict) -> str:
        """Execute plans and decisions, analyze observations and report incidents."""
        prompt = task_data.get("prompt", "")
        images = task_data.get("images", [])
        context = task_data.get("context", "")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a field work operations agent. Execute work plans, analyze "
                    "observations from the field, and generate incident reports. "
                    "Be precise and follow standard operational procedures."
                ),
            }
        ]

        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})

        content = [{"type": "text", "text": prompt}]

        for img_b64 in images[:4]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"},
                }
            )

        messages.append({"role": "user", "content": content})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )

        return response.choices[0].message.content

    def classify_task(self, task_text: str) -> str:
        """Classify task into Planning, Perception, or Action stage."""
        task_lower = task_text.lower()

        planning_keywords = ["plan", "procedure", "workflow", "extract", "document", "video", "protocol"]
        perception_keywords = ["detect", "identify", "classify", "violation", "safety", "ppe", "incident", "spatial"]
        action_keywords = ["execute", "report", "perform", "action", "submit", "complete"]

        scores = {
            "planning": sum(1 for kw in planning_keywords if kw in task_lower),
            "perception": sum(1 for kw in perception_keywords if kw in task_lower),
            "action": sum(1 for kw in action_keywords if kw in task_lower),
        }

        return max(scores, key=scores.get)


class FWAPurpleAgentExecution(AgentExecution):
    """A2A Agent Execution handler for FieldWorkArena purple agent."""

    def __init__(self):
        self.task_handler = FWATaskHandler()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a FieldWorkArena task."""
        task_id = context.task_id
        logger.info(f"Executing FWA task: {task_id}")

        # Extract task data from context
        messages = context.messages
        if not messages:
            await event_queue.enqueue_event(
                new_agent_text_message("No task provided.", context_id=context.context_id, task_id=task_id)
            )
            return

        # Get the last message content
        last_message = messages[-1]
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

        # Classify and process task
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
                new_agent_text_message(
                    f"Error processing task: {str(e)}", context_id=context.context_id, task_id=task_id
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel is not supported for FWA tasks."""
        raise UnsupportedOperationError("Cancel not supported")


def create_agent_card(host: str, port: int) -> AgentCard:
    """Create the agent card for the A2A server."""
    return AgentCard(
        name="FWA Research Agent",
        description=(
            "A2A Purple Agent for FieldWorkArena benchmark. "
            "Handles Planning, Perception, and Action tasks in factory, warehouse, and retail environments."
        ),
        url=f"http://{host}:{port}",
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        defaultInputModes=["text/plain", "application/json"],
        defaultOutputModes=["text/plain"],
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
    host = os.getenv("FWA_HOST", "0.0.0.0")
    port = int(os.getenv("FWA_PORT", "9019"))

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting FWA Research Agent on {host}:{port}")

    agent_card = create_agent_card(host, port)
    execution = FWAPurpleAgentExecution()
    request_handler = DefaultRequestHandler(agent_executor=execution, agent_card=agent_card)

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
