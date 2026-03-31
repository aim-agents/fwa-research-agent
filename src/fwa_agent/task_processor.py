"""Task classification and processing for FieldWorkArena tasks."""

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class TaskStage(Enum):
    """FieldWorkArena task stages."""
    PLANNING = "planning"
    PERCEPTION = "perception"
    ACTION = "action"


@dataclass
class FWATask:
    """Represents a FieldWorkArena task."""
    prompt: str
    stage: TaskStage
    domain: str = "factory"
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    context: str = ""
    task_id: str = ""


class TaskClassifier:
    """Classifies FieldWorkArena tasks into stages."""

    PLANNING_KEYWORDS = [
        "plan", "procedure", "workflow", "extract", "document", "video",
        "protocol", "steps", "sequence", "schedule", "organize", "structure"
    ]
    PERCEPTION_KEYWORDS = [
        "detect", "identify", "classify", "violation", "safety", "ppe",
        "incident", "spatial", "observe", "recognize", "find", "locate",
        "anomaly", "compliance", "hazard"
    ]
    ACTION_KEYWORDS = [
        "execute", "report", "perform", "action", "submit", "complete",
        "carry out", "implement", "do", "run", "process"
    ]

    @classmethod
    def classify(cls, task_text: str) -> TaskStage:
        """Classify a task into Planning, Perception, or Action stage."""
        task_lower = task_text.lower()

        scores = {
            TaskStage.PLANNING: sum(1 for kw in cls.PLANNING_KEYWORDS if kw in task_lower),
            TaskStage.PERCEPTION: sum(1 for kw in cls.PERCEPTION_KEYWORDS if kw in task_lower),
            TaskStage.ACTION: sum(1 for kw in cls.ACTION_KEYWORDS if kw in task_lower),
        }

        # Default to perception if no clear match
        best_stage = max(scores, key=scores.get)
        if scores[best_stage] == 0:
            return TaskStage.PERCEPTION

        return best_stage


class PlanningProcessor:
    """Processes planning-stage tasks: extract procedures from docs/videos."""

    SYSTEM_PROMPT = """You are a field work planning expert. Your job is to analyze documents, images, and videos from industrial environments (factories, warehouses, retail) and extract structured work procedures.

When analyzing materials:
1. Identify the key steps in any work process
2. Note safety requirements and PPE needs
3. List required tools and equipment
4. Highlight dependencies between steps
5. Estimate time for each step

Output format: Provide a structured JSON response with:
- "procedure_name": Name of the procedure
- "steps": Array of step objects with "order", "action", "safety_notes", "equipment_needed"
- "estimated_duration_minutes": Total time estimate
- "safety_requirements": List of safety requirements
- "dependencies": Any step dependencies"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    async def process(self, task: FWATask) -> str:
        """Process a planning task."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        content = [{"type": "text", "text": task.prompt}]

        for img_b64 in task.images[:4]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "low"
                }
            })

        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
        )

        return response.choices[0].message.content


class PerceptionProcessor:
    """Processes perception-stage tasks: detect violations, classify incidents."""

    SYSTEM_PROMPT = """You are a field work safety and compliance expert. Your job is to analyze images and reports from industrial environments to detect safety violations, classify incidents, and perform spatial reasoning.

When analyzing field materials:
1. Identify any safety violations or hazards
2. Check for PPE compliance (helmets, gloves, vests, etc.)
3. Detect spatial anomalies or unsafe arrangements
4. Classify any incidents by type and severity
5. Note environmental conditions

Output format: Provide a structured JSON response with:
- "violations": Array of detected violations with "type", "severity" (low/medium/high), "description"
- "ppe_compliance": Object with "required" list and "observed" list
- "incidents": Array of incidents with "type", "severity", "description"
- "spatial_notes": Any spatial observations
- "confidence_score": 0-1 confidence in findings"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    async def process(self, task: FWATask) -> str:
        """Process a perception task."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        content = [{"type": "text", "text": task.prompt}]

        for img_b64 in task.images[:4]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high"
                }
            })

        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
        )

        return response.choices[0].message.content


class ActionProcessor:
    """Processes action-stage tasks: execute plans, report incidents."""

    SYSTEM_PROMPT = """You are a field work operations agent. Your job is to execute work plans, analyze observations from the field, and generate incident reports.

When executing tasks:
1. Follow the provided procedures precisely
2. Document each action taken
3. Note any deviations from expected outcomes
4. Generate clear, structured reports
5. Flag any issues requiring attention

Output format: Provide a structured JSON response with:
- "actions_taken": Array of actions with "step", "result", "timestamp_note"
- "observations": Any notable observations during execution
- "issues": Array of any issues encountered
- "report": Summary report text
- "status": "completed" or "requires_attention"
- "recommendations": Array of recommendations for next steps"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    async def process(self, task: FWATask) -> str:
        """Process an action task."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        if task.context:
            messages.append({"role": "user", "content": f"Context: {task.context}"})

        content = [{"type": "text", "text": task.prompt}]

        for img_b64 in task.images[:4]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "low"
                }
            })

        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
        )

        return response.choices[0].message.content


class FWATaskProcessor:
    """Main task processor that routes tasks to appropriate handlers."""

    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        self.planning = PlanningProcessor(client, model)
        self.perception = PerceptionProcessor(client, model)
        self.action = ActionProcessor(client, model)

    async def process_task(self, task: FWATask) -> str:
        """Route and process a task based on its stage."""
        logger.info(f"Processing {task.stage.value} task for {task.domain}")

        if task.stage == TaskStage.PLANNING:
            return await self.planning.process(task)
        elif task.stage == TaskStage.PERCEPTION:
            return await self.perception.process(task)
        else:
            return await self.action.process(task)
