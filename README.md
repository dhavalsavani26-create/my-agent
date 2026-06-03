# my-agent

AI agent for browser automation, task execution, and workflow management.

## Browser automation agent

This repository includes a Python browser automation agent built with
[Playwright](https://playwright.dev/python/). It can:

- Open websites and return page summaries.
- Search Google and extract normalized result titles, URLs, and snippets.
- Extract visible text and links from pages.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
playwright install chromium
```

## CLI usage

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

Add `--headed` before the subcommand to watch the browser run:

```bash
browser-agent --headed open https://example.com
```

## Python usage

```python
import asyncio
from browser_agent import BrowserAgent


async def main():
    async with BrowserAgent(headless=True) as agent:
        await agent.open("https://example.com")
        print(await agent.extract_text("h1"))

        results = await agent.google_search("OpenAI", limit=3)
        for result in results:
            print(result.title, result.url)


asyncio.run(main())
```
