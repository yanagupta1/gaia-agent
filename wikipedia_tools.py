import json
import os
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "gaia-agent/0.6 (course final assignment)"


class WikipediaTool:
    """
    Deterministic Wikipedia access via the MediaWiki API. Replaces flaky
    HTML-scraping for the many GAIA questions that are Wikipedia-based: the API
    returns the same clean plain text every call, so answers stop depending on
    which page a web search happened to surface.
    """

    def __init__(
        self,
        timeout_seconds: int | None = None,
        page_char_limit: int | None = None,
        max_results: int | None = None,
    ):
        self.timeout_seconds = timeout_seconds or int(os.getenv("WIKI_TIMEOUT_SECONDS", "20"))
        self.page_char_limit = page_char_limit or int(os.getenv("WIKI_PAGE_CHAR_LIMIT", "8000"))
        self.max_results = max_results or int(os.getenv("WIKI_MAX_RESULTS", "5"))

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json", "formatversion": "2"}
        request = Request(
            f"{API_URL}?{urlencode(params)}",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def search(self, query: str) -> str:
        data = self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": self.max_results,
                "srnamespace": "0|4",  # articles and Wikipedia: (for FAC pages)
            }
        )
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results."
        lines = []
        for hit in hits:
            snippet = re.sub(r"<.*?>", "", hit.get("snippet", "")).strip()
            lines.append(f"- {hit['title']}: {snippet}")
        return "\n".join(lines)

    def get_page(self, title: str, wikitext: bool = False) -> str:
        """Return a page's clean plain text, or raw wikitext (better for tables)."""
        if wikitext:
            data = self._get(
                {"action": "parse", "page": title, "prop": "wikitext", "redirects": "1"}
            )
            text = data.get("parse", {}).get("wikitext", "")
            if not text:
                return f"No Wikipedia page found for {title!r}."
            return text[: self.page_char_limit]

        data = self._get(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "titles": title,
                "redirects": "1",
            }
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return f"No Wikipedia page found for {title!r}."
        extract = pages[0].get("extract", "")
        if not extract:
            return f"Wikipedia page {title!r} has no extractable text."
        return extract[: self.page_char_limit]
