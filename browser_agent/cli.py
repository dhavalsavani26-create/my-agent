"""Command-line interface for the browser automation agent."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from .agent import BrowserAgent, format_search_results


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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
