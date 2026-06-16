"""Command-line interface for the browser automation agent."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from .agent import BrowserAgent, format_search_results
from .autonomous import AutonomousBrowserAgent


async def _run_open(args: argparse.Namespace) -> None:
    async with BrowserAgent(headless=not args.headed) as agent:
        title = await agent.open(args.url)
        summary = await agent.summarize_current_page(max_chars=args.max_chars)
        summary["title"] = title
        print(json.dumps(summary, indent=2, ensure_ascii=False))


async def _run_search(args: argparse.Namespace) -> None:
    async with BrowserAgent(headless=not args.headed) as agent:
        results = await agent.google_search(args.query, limit=args.limit)
        if args.json:
            print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
        else:
            print(format_search_results(results))


async def _run_extract(args: argparse.Namespace) -> None:
    async with BrowserAgent(headless=not args.headed) as agent:
        await agent.open(args.url)
        print(await agent.extract_text(args.selector, max_chars=args.max_chars))


async def _run_screenshot(args: argparse.Namespace) -> None:
    async with BrowserAgent(headless=not args.headed) as agent:
        await agent.open(args.url)
        saved_path = await agent.screenshot(path=args.path, full_page=not args.viewport_only)
        print(saved_path)


async def _run_autonomous(args: argparse.Namespace) -> None:
    async with AutonomousBrowserAgent(
        headless=not args.headed,
        memory_path=args.memory,
        max_steps=args.max_steps,
    ) as agent:
        run = await agent.run(args.objective)
        payload = {
            "objective": run.objective,
            "final_answer": run.final_answer,
            "progress": run.plan.progress(),
            "steps": [
                {
                    "description": step.description,
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                    "status": step.status.value,
                    "result": step.result,
                }
                for step in run.plan.steps
            ],
            "events": [asdict(event) for event in run.events],
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(run.final_answer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browser automation agent powered by Python and Playwright.")
    parser.add_argument("--headed", action="store_true", help="Show the browser window instead of running headless.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    open_parser = subparsers.add_parser("open", help="Open a website and print a JSON page summary.")
    open_parser.add_argument("url", help="Website URL to open. https:// is added when omitted.")
    open_parser.add_argument("--max-chars", type=int, default=2_000, help="Maximum text characters to include.")
    open_parser.set_defaults(func=_run_open)

    search_parser = subparsers.add_parser("search", help="Search Google and print result titles, URLs, and snippets.")
    search_parser.add_argument("query", help="Google search query.")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of search results.")
    search_parser.add_argument("--json", action="store_true", help="Print raw JSON instead of human-readable text.")
    search_parser.set_defaults(func=_run_search)

    extract_parser = subparsers.add_parser("extract", help="Open a website and extract visible text from a selector.")
    extract_parser.add_argument("url", help="Website URL to open. https:// is added when omitted.")
    extract_parser.add_argument("--selector", default="body", help="CSS selector to extract from.")
    extract_parser.add_argument("--max-chars", type=int, default=4_000, help="Maximum text characters to print.")
    extract_parser.set_defaults(func=_run_extract)

    screenshot_parser = subparsers.add_parser("screenshot", help="Open a website and save a PNG screenshot.")
    screenshot_parser.add_argument("url", help="Website URL to open. https:// is added when omitted.")
    screenshot_parser.add_argument("path", help="Output PNG path.")
    screenshot_parser.add_argument(
        "--viewport-only",
        action="store_true",
        help="Capture only the current viewport instead of the full page.",
    )
    screenshot_parser.set_defaults(func=_run_screenshot)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a Manus-style autonomous task with planning, memory, tools, and browser control.",
    )
    run_parser.add_argument("objective", help="Natural-language task or URL for the autonomous agent.")
    run_parser.add_argument("--memory", help="Optional JSONL path for persistent memory across runs.")
    run_parser.add_argument("--max-steps", type=int, default=8, help="Maximum autonomous loop steps.")
    run_parser.add_argument("--json", action="store_true", help="Print structured JSON run details.")
    run_parser.set_defaults(func=_run_autonomous)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
