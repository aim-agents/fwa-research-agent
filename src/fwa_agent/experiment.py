"""Experiment design module for FieldWorkArena research tasks.

Generates hypotheses, designs validation approaches, and plans experiments.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """Represents a research hypothesis."""
    statement: str
    variables: list[str]
    expected_outcome: str
    confidence: float = 0.5
    testable: bool = True


@dataclass
class ExperimentDesign:
    """Represents an experiment design."""
    title: str
    hypothesis: Hypothesis
    methodology: str
    variables: dict[str, list[str]]  # independent, dependent, controlled
    sample_size: int
    duration_estimate: str
    success_criteria: list[str]
    risks: list[str] = field(default_factory=list)


@dataclass
class ValidationPlan:
    """Plan for validating experiment results."""
    metrics: list[str]
    methods: list[str]
    thresholds: dict[str, float]
    analysis_approach: str


class ExperimentDesigner:
    """Designs experiments based on research questions."""

    SYSTEM_PROMPT = """You are an expert research scientist specializing in experimental design.

Given a research question or observation, generate:
1. A testable hypothesis
2. Independent, dependent, and controlled variables
3. A methodology for testing the hypothesis
4. Success criteria and validation metrics
5. Potential risks and mitigations

Output structured JSON with:
- hypothesis: {statement, variables, expected_outcome, confidence}
- methodology: Description of experimental approach
- variables: {independent, dependent, controlled}
- sample_size: Recommended sample size
- duration_estimate: Estimated time
- success_criteria: Array of success conditions
- risks: Array of potential risks
- validation_plan: {metrics, methods, thresholds}"""

    def __init__(self, client=None, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def design_from_research_question(self, question: str) -> ExperimentDesign:
        """Design an experiment from a research question.

        Args:
            question: The research question to investigate

        Returns:
            ExperimentDesign object
        """
        # If no client, return a template design
        if not self.client:
            return self._template_design(question)

        # Use LLM to generate design
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Research question: {question}"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )

            # Parse response
            data = json.loads(response.choices[0].message.content)
            return self._parse_design(data, question)

        except Exception as e:
            logger.error(f"Error designing experiment: {e}")
            return self._template_design(question)

    def _template_design(self, question: str) -> ExperimentDesign:
        """Create a template experiment design."""
        return ExperimentDesign(
            title=f"Investigation: {question[:100]}",
            hypothesis=Hypothesis(
                statement="To be determined based on initial analysis",
                variables=[],
                expected_outcome="",
                confidence=0.5,
            ),
            methodology="Controlled experiment with A/B testing",
            variables={
                "independent": ["treatment_condition"],
                "dependent": ["outcome_measure"],
                "controlled": ["environment", "resources"],
            },
            sample_size=30,
            duration_estimate="1-2 weeks",
            success_criteria=["Statistical significance p < 0.05"],
            risks=["Sample bias", "Confounding variables"],
        )

    def _parse_design(self, data: dict, question: str) -> ExperimentDesign:
        """Parse LLM response into ExperimentDesign."""
        hyp_data = data.get("hypothesis", {})

        return ExperimentDesign(
            title=data.get("title", f"Investigation: {question[:100]}"),
            hypothesis=Hypothesis(
                statement=hyp_data.get("statement", ""),
                variables=hyp_data.get("variables", []),
                expected_outcome=hyp_data.get("expected_outcome", ""),
                confidence=hyp_data.get("confidence", 0.5),
            ),
            methodology=data.get("methodology", ""),
            variables=data.get("variables", {}),
            sample_size=data.get("sample_size", 30),
            duration_estimate=data.get("duration_estimate", ""),
            success_criteria=data.get("success_criteria", []),
            risks=data.get("risks", []),
        )

    def validate_design(self, design: ExperimentDesign) -> list[str]:
        """Validate an experiment design for completeness.

        Args:
            design: The experiment design to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not design.hypothesis.statement:
            issues.append("Missing hypothesis statement")

        if not design.hypothesis.variables:
            issues.append("No variables identified")

        if not design.methodology:
            issues.append("Missing methodology")

        if design.sample_size < 1:
            issues.append("Invalid sample size")

        if not design.success_criteria:
            issues.append("No success criteria defined")

        return issues
