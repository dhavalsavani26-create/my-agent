"""Manus-style autonomous browser agent.

The autonomous layer adds planning, persistent memory, a tool registry, and a
bounded execution loop on top of :class:`browser_agent.agent.BrowserAgent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agent import BrowserAgent
from .memory import MemoryRecord, MemoryStore
from .planning import Planner, StepStatus, TaskPlan
from .tools import BrowserToolKit, ToolRegistry, ToolResult, ToolSpec


@dataclass(slots=True)
class AgentEvent:
    """One event emitted by the autonomous loop."""

    type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRun:
    """Structured result of an autonomous task run."""

    objective: str
    final_answer: str
    plan: TaskPlan
    events: list[AgentEvent]
    memories: list[MemoryRecord]


class AutonomousBrowserAgent:
    """A browser-capable autonomous agent with planning, memory, and tools.

    The agent is intentionally model-agnostic: it can execute deterministic
    browser plans out of the box, while applications can add custom tools or
    modify generated plans to connect a hosted LLM/controller.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        browser_name: str = "chromium",
        timeout_ms: int = 30_000,
        user_agent: str | None = None,
        memory_path: str | None = None,
        max_steps: int = 8,
        planner: Planner | None = None,
    ) -> None:
        self.browser = BrowserAgent(
            headless=headless,
            browser_name=browser_name,
            timeout_ms=timeout_ms,
            user_agent=user_agent,
        )
        self.memory = MemoryStore(memory_path)
        self.max_steps = max_steps
        self.planner = planner or Planner()
        self.tools = ToolRegistry()
        BrowserToolKit(self.browser).register(self.tools)
        self._register_memory_tools()

    async def __aenter__(self) -> "AutonomousBrowserAgent":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def start(self) -> None:
        """Start the controlled browser."""

        await self.browser.start()

    async def close(self) -> None:
        """Close the controlled browser."""

        await self.browser.close()

    def register_tool(self, spec: ToolSpec) -> None:
        """Register an additional async tool the agent may use."""

        self.tools.register(spec)

    async def run(self, objective: str, *, plan: TaskPlan | None = None) -> AgentRun:
        """Plan and execute an objective with bounded autonomous tool use."""

        task_plan = plan or self.planner.create_plan(objective)
        events = [AgentEvent("plan", "Created plan.", {"steps": [step.description for step in task_plan.steps]})]
        observations: list[ToolResult] = []

        for _ in range(self.max_steps):
            step = task_plan.next_step()
            if step is None:
                break

            step.status = StepStatus.IN_PROGRESS
            events.append(
                AgentEvent(
                    "step_started",
                    step.description,
                    {"tool_name": step.tool_name, "arguments": step.arguments},
                )
            )

            if step.tool_name is None:
                step.result = self._compose_final_answer(task_plan.objective, observations)
                step.status = StepStatus.COMPLETED
                events.append(AgentEvent("final", step.result))
                break

            if step.tool_name == "remember" and not step.arguments.get("content"):
                step.arguments["content"] = self._compose_memory_summary(task_plan.objective, observations)

            result = await self.tools.call(step.tool_name, step.arguments)
            observations.append(result)
            step.result = result.as_text()
            step.status = StepStatus.COMPLETED if result.ok else StepStatus.FAILED
            events.append(
                AgentEvent(
                    "tool_result",
                    f"{step.tool_name} {'succeeded' if result.ok else 'failed'}.",
                    {"result": result.as_text()},
                )
            )

            if not result.ok:
                self.memory.remember(
                    f"Tool {step.tool_name} failed while pursuing {task_plan.objective}: {result.error}",
                    kind="error",
                    metadata={"tool_name": step.tool_name},
                )

        final_answer = self._final_step_result(task_plan) or self._compose_final_answer(task_plan.objective, observations)
        return AgentRun(
            objective=task_plan.objective,
            final_answer=final_answer,
            plan=task_plan,
            events=events,
            memories=self.memory.recent(limit=10),
        )

    def _register_memory_tools(self) -> None:
        async def remember(content: str, kind: str = "observation", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
            record = self.memory.remember(content, kind=kind, metadata=metadata)
            return {"id": record.id, "kind": record.kind, "content": record.content}

        async def recall_memory(query: str, limit: int = 5, kind: str | None = None) -> list[dict[str, Any]]:
            return [
                {"id": record.id, "kind": record.kind, "content": record.content, "metadata": record.metadata}
                for record in self.memory.search(query, limit=limit, kind=kind)
            ]

        self.tools.register(
            ToolSpec(
                name="remember",
                description="Store an observation or task summary in persistent memory.",
                parameters={"content": "Text to remember", "kind": "Memory category"},
                handler=remember,
            )
        )
        self.tools.register(
            ToolSpec(
                name="recall_memory",
                description="Recall memories relevant to the current task.",
                parameters={"query": "Recall query", "limit": "Maximum memories"},
                handler=recall_memory,
            )
        )

    @staticmethod
    def _compose_memory_summary(objective: str, observations: list[ToolResult]) -> str:
        useful = [result.as_text(max_chars=500) for result in observations if result.ok and result.tool_name != "recall_memory"]
        if not useful:
            return f"Worked on objective: {objective}."
        return f"Objective: {objective}\nKey observations:\n" + "\n".join(f"- {item}" for item in useful[-3:])

    @staticmethod
    def _compose_final_answer(objective: str, observations: list[ToolResult]) -> str:
        lines = [f"Completed autonomous run for: {objective}", "", "Observations:"]
        for result in observations:
            status = "ok" if result.ok else "error"
            lines.append(f"- {result.tool_name} ({status}): {result.as_text(max_chars=700)}")
        return "\n".join(lines)

    @staticmethod
    def _final_step_result(plan: TaskPlan) -> str:
        final_steps = [step for step in plan.steps if step.tool_name is None and step.result]
        return final_steps[-1].result if final_steps else ""
