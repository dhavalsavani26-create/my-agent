"""Tool abstractions and browser tools for autonomous agents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

from .agent import BrowserAgent

ToolHandler = Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class ToolResult:
    """Result returned by an autonomous tool invocation."""

    tool_name: str
    ok: bool
    output: Any = None
    error: str = ""

    def as_text(self, *, max_chars: int = 2000) -> str:
        """Render tool output compactly for logs, memory, and final reports."""

        payload = self.output if self.ok else {"error": self.error}
        if isinstance(payload, str):
            return payload[:max_chars]
        return json.dumps(payload, ensure_ascii=False, default=str)[:max_chars]


@dataclass(slots=True)
class ToolSpec:
    """Description and callable for one autonomous tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler = field(repr=False)


class ToolRegistry:
    """Registry that executes named async tools and normalizes failures."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
            for spec in self._tools.values()
        ]

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(tool_name=name, ok=False, error=f"Unknown tool: {name}")
        try:
            output = await spec.handler(**(arguments or {}))
        except Exception as exc:  # Tool failures should not crash the whole loop.
            return ToolResult(tool_name=name, ok=False, error=str(exc))
        return ToolResult(tool_name=name, ok=True, output=output)


class BrowserToolKit:
    """Registers Playwright browser-control tools for autonomous execution."""

    def __init__(self, browser: BrowserAgent) -> None:
        self.browser = browser

    def register(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolSpec(
                name="open_url",
                description="Open a URL in the controlled browser and return title, URL, and page summary.",
                parameters={"url": "URL to open"},
                handler=self.open_url,
            )
        )
        registry.register(
            ToolSpec(
                name="search_web",
                description="Search Google in the browser and return normalized organic results.",
                parameters={"query": "Search query", "limit": "Maximum number of results"},
                handler=self.search_web,
            )
        )
        registry.register(
            ToolSpec(
                name="summarize_page",
                description="Summarize the active browser page with visible text and links.",
                parameters={"max_chars": "Maximum page text characters"},
                handler=self.summarize_page,
            )
        )
        registry.register(
            ToolSpec(
                name="extract_text",
                description="Extract visible text from a CSS selector on the active page.",
                parameters={"selector": "CSS selector", "max_chars": "Maximum characters"},
                handler=self.extract_text,
            )
        )
        registry.register(
            ToolSpec(
                name="extract_links",
                description="Extract visible HTTP links from the active page.",
                parameters={"limit": "Maximum links"},
                handler=self.extract_links,
            )
        )
        registry.register(
            ToolSpec(
                name="click",
                description="Click an element by CSS selector or by accessible role and name.",
                parameters={"selector": "CSS selector", "role": "ARIA role", "name": "Accessible name"},
                handler=self.click,
            )
        )
        registry.register(
            ToolSpec(
                name="fill",
                description="Fill a form field located by CSS selector.",
                parameters={"selector": "CSS selector", "text": "Text to enter"},
                handler=self.fill,
            )
        )
        registry.register(
            ToolSpec(
                name="press",
                description="Press a keyboard key in the active page.",
                parameters={"key": "Keyboard key, e.g. Enter"},
                handler=self.press,
            )
        )

    async def open_url(self, url: str) -> dict[str, Any]:
        await self.browser.open(url)
        return await self.browser.summarize_current_page(max_chars=1500)

    async def search_web(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        results = await self.browser.google_search(query, limit=limit)
        return [asdict(result) for result in results]

    async def summarize_page(self, max_chars: int = 2000) -> dict[str, Any]:
        return await self.browser.summarize_current_page(max_chars=max_chars)

    async def extract_text(self, selector: str = "body", max_chars: int = 4000) -> str:
        return await self.browser.extract_text(selector, max_chars=max_chars)

    async def extract_links(self, limit: int = 20) -> list[dict[str, str]]:
        return await self.browser.extract_links(limit=limit)

    async def click(self, selector: str | None = None, role: str | None = None, name: str | None = None) -> str:
        page = self.browser._require_page()
        if selector:
            await page.locator(selector).first.click()
            return f"Clicked selector {selector!r}."
        if role and name:
            await page.get_by_role(role, name=name).first.click()
            return f"Clicked role {role!r} named {name!r}."
        raise ValueError("Provide either selector or both role and name.")

    async def fill(self, selector: str, text: str) -> str:
        page = self.browser._require_page()
        await page.locator(selector).first.fill(text)
        return f"Filled {selector!r}."

    async def press(self, key: str) -> str:
        page = self.browser._require_page()
        await page.keyboard.press(key)
        return f"Pressed {key!r}."
