"""Playwright-powered browser automation agent.

The :class:`BrowserAgent` class provides a small, scriptable interface for
opening websites, running Google searches, and extracting readable information
from pages.  It keeps Playwright lifecycle management in one place so CLI tools
or other Python code can focus on the task being automated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
from urllib.parse import quote_plus

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page


@dataclass(slots=True)
class SearchResult:
    """A normalized Google search result."""

    title: str
    url: str
    snippet: str = ""


class BrowserAgent:
    """High-level automation helper built on Playwright.

    Use the agent as an async context manager to ensure the browser is closed:

    .. code-block:: python

        async with BrowserAgent(headless=True) as agent:
            await agent.open("https://example.com")
            facts = await agent.extract_text("h1")
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        browser_name: str = "chromium",
        timeout_ms: int = 30_000,
        user_agent: str | None = None,
    ) -> None:
        self.headless = headless
        self.browser_name = browser_name
        self.timeout_ms = timeout_ms
        self.user_agent = user_agent
        self._playwright: Any = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def __aenter__(self) -> "BrowserAgent":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def start(self) -> None:
        """Start Playwright and open a fresh page."""

        if self.page is not None:
            return

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        browser_launcher = getattr(self._playwright, self.browser_name)
        self.browser = await browser_launcher.launch(headless=self.headless)
        context_options = {"user_agent": self.user_agent} if self.user_agent else {}
        self.context = await self.browser.new_context(**context_options)
        self.context.set_default_timeout(self.timeout_ms)
        self.page = await self.context.new_page()

    async def close(self) -> None:
        """Close browser resources."""

        if self.context is not None:
            await self.context.close()
        if self.browser is not None:
            await self.browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def open(self, url: str, *, wait_until: str = "domcontentloaded") -> str:
        """Open a website and return the final page title."""

        page = self._require_page()
        normalized_url = self._normalize_url(url)
        await page.goto(normalized_url, wait_until=wait_until)
        return await page.title()

    async def google_search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Search Google and return normalized organic result cards.

        Google markup changes over time, so this method intentionally uses a few
        fallback selectors and filters out empty or non-HTTP links.
        """

        if limit < 1:
            return []

        page = self._require_page()
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en"
        await page.goto(search_url, wait_until="domcontentloaded")
        await self._accept_google_consent_if_present(page)

        result_locators = ["div.g", "div[data-sokoban-container]", "#search .MjjYud"]
        for selector in result_locators:
            results = await self._extract_google_results(selector, limit)
            if results:
                return results[:limit]
        return []

    async def screenshot(self, *, path: str | Path, full_page: bool = True) -> str:
        """Save a PNG screenshot of the current page and return the file path."""

        page = self._require_page()
        screenshot_path = Path(path).expanduser()
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(screenshot_path), full_page=full_page)
        return str(screenshot_path)

    async def extract_text(self, selector: str = "body", *, max_chars: int = 4_000) -> str:
        """Extract visible text from the current page or selector."""

        page = self._require_page()
        locator = page.locator(selector).first
        text = await locator.inner_text(timeout=self.timeout_ms)
        return self._compact_text(text)[:max_chars]

    async def extract_links(self, *, limit: int = 20) -> list[dict[str, str]]:
        """Extract visible links from the current page."""

        page = self._require_page()
        links = await page.locator("a[href]").evaluate_all(
            """
            anchors => anchors.map(a => ({
                text: (a.innerText || a.textContent || '').trim(),
                url: a.href
            }))
            """
        )
        clean_links: list[dict[str, str]] = []
        seen: set[str] = set()
        for link in links:
            url = link.get("url", "")
            text = self._compact_text(link.get("text", ""))
            if not url.startswith(("http://", "https://")) or url in seen:
                continue
            clean_links.append({"text": text, "url": url})
            seen.add(url)
            if len(clean_links) >= limit:
                break
        return clean_links

    async def summarize_current_page(self, *, max_chars: int = 2_000) -> dict[str, object]:
        """Return title, URL, text, and links for the active page."""

        page = self._require_page()
        return {
            "title": await page.title(),
            "url": page.url,
            "text": await self.extract_text(max_chars=max_chars),
            "links": await self.extract_links(limit=10),
        }

    async def _extract_google_results(self, selector: str, limit: int) -> list[SearchResult]:
        page = self._require_page()
        cards = await page.locator(selector).all()
        results: list[SearchResult] = []
        seen: set[str] = set()

        for card in cards:
            link = card.locator("a[href]").first
            href = await link.get_attribute("href") if await link.count() else None
            if not href or not href.startswith(("http://", "https://")) or href in seen:
                continue

            title = ""
            for title_selector in ("h3", "[role='heading']", "a"):
                candidate = card.locator(title_selector).first
                if await candidate.count():
                    title = self._compact_text(await candidate.inner_text())
                    if title:
                        break
            if not title:
                continue

            snippet = ""
            for snippet_selector in (".VwiC3b", "[data-sncf]", "span"):
                candidate = card.locator(snippet_selector).first
                if await candidate.count():
                    snippet = self._compact_text(await candidate.inner_text())
                    if snippet and snippet != title:
                        break

            results.append(SearchResult(title=title, url=href, snippet=snippet))
            seen.add(href)
            if len(results) >= limit:
                break
        return results

    async def _accept_google_consent_if_present(self, page: Page) -> None:
        for label in ("Accept all", "I agree", "Reject all"):
            button = page.get_by_role("button", name=label).first
            try:
                if await button.count():
                    await button.click(timeout=2_000)
                    await page.wait_for_load_state("domcontentloaded")
                    return
            except Exception:  # Google consent UIs vary by region; ignore misses.
                continue

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("BrowserAgent has not been started. Use 'async with BrowserAgent(...)'.")
        return self.page

    @staticmethod
    def _normalize_url(url: str) -> str:
        if url.startswith(("http://", "https://")):
            return url
        return f"https://{url}"

    @staticmethod
    def _compact_text(text: str) -> str:
        return " ".join(text.split())


def format_search_results(results: Iterable[SearchResult]) -> str:
    """Format search results for terminal output."""

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}\n   {result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet}")
    return "\n".join(lines)
