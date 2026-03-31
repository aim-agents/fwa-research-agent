"""Tests for FWA Research Agent."""

import pytest

from fwa_agent.task_processor import TaskClassifier, TaskStage


def test_classify_planning_task():
    """Test classification of planning tasks."""
    task = "Extract the work procedure from this video documentation"
    assert TaskClassifier.classify(task) == TaskStage.PLANNING


def test_classify_perception_task():
    """Test classification of perception tasks."""
    task = "Detect any safety violations in this factory image"
    assert TaskClassifier.classify(task) == TaskStage.PERCEPTION


def test_classify_action_task():
    """Test classification of action tasks."""
    task = "Execute the plan and report the results"
    assert TaskClassifier.classify(task) == TaskStage.ACTION


def test_classify_default_perception():
    """Test default classification to perception."""
    task = "Analyze the data and provide findings"
    # Should default to perception when no clear match
    result = TaskClassifier.classify(task)
    assert result in [TaskStage.PERCEPTION, TaskStage.ACTION]
