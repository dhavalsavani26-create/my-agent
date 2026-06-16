import asyncio

from browser_agent.agent import BrowserAgent, SearchResult, format_search_results
from browser_agent.memory import MemoryStore
from browser_agent.planning import Planner, StepStatus
from browser_agent.tools import BrowserToolKit, ToolRegistry, ToolSpec


def test_normalize_url_adds_https_when_scheme_missing():
    assert BrowserAgent._normalize_url("example.com") == "https://example.com"


def test_normalize_url_preserves_existing_scheme():
    assert BrowserAgent._normalize_url("http://example.com") == "http://example.com"
    assert BrowserAgent._normalize_url("https://example.com") == "https://example.com"


def test_compact_text_removes_extra_whitespace():
    assert BrowserAgent._compact_text(" hello\n\t world   ") == "hello world"


def test_format_search_results_includes_title_url_and_snippet():
    formatted = format_search_results([
        SearchResult(title="Example", url="https://example.com", snippet="A sample site")
    ])

    assert "1. Example" in formatted
    assert "https://example.com" in formatted
    assert "A sample site" in formatted


async def _sample_tool(value="ok"):
    return {"value": value}


def test_memory_store_remembers_and_searches_records():
    memory = MemoryStore()
    memory.remember("Browser agents can use tools", kind="fact")
    memory.remember("Unrelated note", kind="note")

    matches = memory.search("agents tools")

    assert matches[0].content == "Browser agents can use tools"
    assert memory.recent(1)[0].content == "Unrelated note"


def test_planner_creates_url_plan_with_memory_and_browser_steps():
    plan = Planner().create_plan("example.com")

    assert plan.objective == "example.com"
    assert plan.next_step().tool_name == "recall_memory"
    assert [step.tool_name for step in plan.steps] == ["recall_memory", "open_url", "summarize_page", "remember", None]
    assert plan.progress()[StepStatus.PENDING.value] == 5


def test_planner_creates_search_plan_for_natural_language_objective():
    plan = Planner().create_plan("find browser automation examples")

    assert [step.tool_name for step in plan.steps] == ["recall_memory", "search_web", "summarize_page", "remember", None]


def test_tool_registry_calls_registered_async_tool():
    registry = ToolRegistry()
    registry.register(ToolSpec("sample", "Sample tool", {"value": "Value"}, _sample_tool))

    result = asyncio.run(registry.call("sample", {"value": "done"}))

    assert result.ok is True
    assert result.output == {"value": "done"}


def test_tool_registry_reports_unknown_tools():
    result = asyncio.run(ToolRegistry().call("missing"))

    assert result.ok is False
    assert "Unknown tool" in result.error


class _FakePage:
    def __init__(self):
        self.screenshot_args = None

    async def screenshot(self, **kwargs):
        self.screenshot_args = kwargs


def test_browser_agent_screenshot_saves_png_path(tmp_path):
    agent = BrowserAgent()
    fake_page = _FakePage()
    agent.page = fake_page
    path = tmp_path / "screens" / "test.png"

    saved_path = asyncio.run(agent.screenshot(path=str(path), full_page=False))

    assert saved_path == str(path)
    assert path.parent.exists()
    assert fake_page.screenshot_args == {"path": str(path), "full_page": False}


def test_toolkit_registers_screenshot_tool():
    registry = ToolRegistry()
    BrowserToolKit(BrowserAgent()).register(registry)

    assert "screenshot" in registry.names()
    assert "submit_form" in registry.names()


def test_autonomous_agent_screenshot_policy_matches_browser_actions():
    from browser_agent.autonomous import AutonomousBrowserAgent

    should_capture = AutonomousBrowserAgent._should_capture_screenshot

    assert should_capture("open_url", {"url": "https://example.com"}, True)
    assert should_capture("navigate", {"url": "https://example.com"}, True)
    assert should_capture("search_web", {"query": "example"}, True)
    assert should_capture("click", {"selector": "button"}, True)
    assert should_capture("menu_click", {}, True)
    assert should_capture("press", {"key": "Tab"}, True)
    assert should_capture("fill", {"selector": "input", "text": "value"}, True)
    assert should_capture("submit_form", {}, True)
    assert should_capture("extract_text", {}, False)
    assert should_capture("extract_text", {"important": True}, True)
    assert should_capture("extract_text", {"uncertain": "yes"}, True)
    assert should_capture("extract_text", {"capture_screenshot": "true"}, True)
    assert not should_capture("extract_text", {}, True)


def test_autonomous_agent_screenshot_stem_is_safe():
    from browser_agent.autonomous import AutonomousBrowserAgent

    stem = AutonomousBrowserAgent._screenshot_stem(2, "click button!", True)

    assert stem.startswith("002-")
    assert stem.endswith("-click-button-ok")


def test_autonomous_agent_screenshot_metadata_matches_expected_shape():
    from browser_agent.autonomous import AutonomousBrowserAgent

    metadata = AutonomousBrowserAgent._screenshot_metadata(
        "click",
        {"reason": "click on login button", "confidence": 0.8},
        True,
    )

    assert metadata == {
        "path": "",
        "reason": "click on login button",
        "trigger": "click",
        "confidence": 0.8,
    }


def test_autonomous_agent_screenshot_confidence_is_clamped():
    from browser_agent.autonomous import AutonomousBrowserAgent

    assert AutonomousBrowserAgent._screenshot_confidence({"confidence": 2}, True) == 1.0
    assert AutonomousBrowserAgent._screenshot_confidence({"confidence": -1}, True) == 0.0
    assert AutonomousBrowserAgent._screenshot_confidence({"confidence": "unknown"}, True) == 0.8
