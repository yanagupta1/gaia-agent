import html
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_result_link = False
        self._current_href = ""
        self._current_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "")
        if tag == "a" and "result__a" in class_name:
            self._in_result_link = True
            self._current_href = attrs_dict.get("href", "")
            self._current_text = []

    def handle_data(self, data):
        if self._in_result_link:
            self._current_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._in_result_link:
            title = html.unescape(" ".join(self._current_text)).strip()
            url = _normalize_duckduckgo_url(self._current_href)
            if title and url and url.startswith(("http://", "https://")):
                self.results.append(SearchResult(title=title, url=url))
            self._in_result_link = False
            self._current_href = ""
            self._current_text = []


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            text = data.strip()
            if text:
                self.parts.append(text)

    def text(self) -> str:
        joined = " ".join(self.parts)
        return re.sub(r"\s+", " ", html.unescape(joined)).strip()


class WebSearchTool:
    def __init__(
        self,
        max_results: int | None = None,
        page_char_limit: int | None = None,
        timeout_seconds: int | None = None,
    ):
        self.max_results = max_results or int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
        self.page_char_limit = page_char_limit or int(os.getenv("WEB_PAGE_CHAR_LIMIT", "6000"))
        self.timeout_seconds = timeout_seconds or int(os.getenv("WEB_TIMEOUT_SECONDS", "20"))

    def search(self, query: str) -> list[SearchResult]:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        document = self._fetch_text(url)
        parser = DuckDuckGoHTMLParser()
        parser.feed(document)
        return parser.results[: self.max_results]

    def read_page(self, url: str) -> str:
        document = self._fetch_text(url)
        extractor = TextExtractor()
        extractor.feed(document)
        return extractor.text()[: self.page_char_limit]

    def _fetch_text(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def _normalize_duckduckgo_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        if "uddg" in query:
            return unquote(query["uddg"][0])
    return url
