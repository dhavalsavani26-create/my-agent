from browser_agent.agent import BrowserAgent, SearchResult, format_search_results


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
