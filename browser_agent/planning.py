"""Planning primitives for Manus-style autonomous task execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StepStatus(StrEnum):
    """Lifecycle status for an autonomous plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class PlanStep:
    """One actionable step in a task plan."""

    description: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: str = ""


@dataclass(slots=True)
class TaskPlan:
    """A mutable autonomous task plan."""

    objective: str
    steps: list[PlanStep]

    def next_step(self) -> PlanStep | None:
        """Return the next pending step."""

        return next((step for step in self.steps if step.status == StepStatus.PENDING), None)

    def progress(self) -> dict[str, int]:
        """Return counts by step status."""

        counts = {status.value: 0 for status in StepStatus}
        for step in self.steps:
            counts[step.status.value] += 1
        return counts


class Planner:
    """Creates a practical browser-first plan from a natural-language objective."""

    def create_plan(self, objective: str) -> TaskPlan:
        """Build a default autonomous plan.

        The planner is intentionally deterministic so the package works without
        requiring a hosted model. Applications can still insert, remove, or edit
        plan steps before running them.
        """

        normalized = objective.strip()
        steps: list[PlanStep] = [
            PlanStep(
                "Recall related long-term memories before acting.",
                "recall_memory",
                {"query": normalized, "limit": 5},
            )
        ]

        if self._looks_like_url(normalized):
            steps.extend(
                [
                    PlanStep("Open the requested page.", "open_url", {"url": normalized}),
                    PlanStep("Summarize the current browser page.", "summarize_page", {"max_chars": 3000}),
                ]
            )
        else:
            steps.extend(
                [
                    PlanStep("Search the web for useful starting points.", "search_web", {"query": normalized, "limit": 5}),
                    PlanStep("Summarize the most relevant current page or search results.", "summarize_page", {"max_chars": 3000}),
                ]
            )

        steps.extend(
            [
                PlanStep("Store key observations in memory.", "remember", {"content": "", "kind": "task_summary"}),
                PlanStep("Produce a concise final answer for the user.", None),
            ]
        )
        return TaskPlan(objective=normalized, steps=steps)

    @staticmethod
    def _looks_like_url(text: str) -> bool:
        return text.startswith(("http://", "https://")) or (" " not in text and "." in text)
