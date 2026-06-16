# my-agent

AI agent for browser automation, task execution, and workflow management.

## Manus-style autonomous browser agent

This repository includes a Python autonomous browser agent built with
[Playwright](https://playwright.dev/python/). It can:

- Create a bounded task plan from a natural-language objective.
- Recall and write persistent JSONL memory across runs.
- Use a tool registry for browser actions, search, extraction, form input, and memory.
- Control a Playwright browser to open pages, search Google, summarize pages, take screenshots, click, fill, submit forms, and press keys.
- Automatically capture screenshots after navigation, clicks, form submits, important/uncertain actions, and tool errors during autonomous runs.
- Return structured run events, plan progress, observations, and a final answer.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
playwright install chromium
```

## CLI usage

Run an autonomous task with planning, memory, tool use, and browser control:

```bash
browser-agent run "research Playwright browser agents" --memory .agent-memory.jsonl --json
```

Autonomous runs save automatic screenshots in `.agent-screenshots` after page navigation, clicks, form submits, important/uncertain actions, and tool errors. Use `--screenshot-dir` to choose a different output directory.

Open a website and print a JSON summary:

```bash
browser-agent open https://example.com
```

Search Google:

```bash
browser-agent search "Playwright Python browser automation" --limit 5
```

Extract text from a page selector:

```bash
browser-agent extract https://example.com --selector body --max-chars 1000
```

Save a full-page PNG screenshot:

```bash
browser-agent screenshot https://example.com test.png
```

Add `--headed` before the subcommand to watch the browser run:

```bash
browser-agent --headed open https://example.com
```

## Python usage

Autonomous task execution:

```python
import asyncio
from browser_agent import AutonomousBrowserAgent


async def main():
    async with AutonomousBrowserAgent(memory_path=".agent-memory.jsonl") as agent:
        run = await agent.run("https://example.com")
        print(run.final_answer)


asyncio.run(main())
```

Low-level browser automation remains available:

```python
import asyncio
from browser_agent import BrowserAgent


async def main():
    async with BrowserAgent(headless=True) as agent:
        await agent.open("https://example.com")
        await agent.screenshot(path="test.png", full_page=True)
        print(await agent.extract_text("h1"))

        results = await agent.google_search("OpenAI", limit=3)
        for result in results:
            print(result.title, result.url)


asyncio.run(main())
```
