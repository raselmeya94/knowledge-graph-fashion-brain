"""HTML link and content extraction. Remote pages are not fetched."""

from __future__ import annotations

from html.parser import HTMLParser

from knowledge_graph.construction.detect import detect_kind, payload_text
from knowledge_graph.errors import ValidationFailed

_SKIP = {"script", "style", "noscript"}
_PRODUCT_WORDS = ("add to cart", "buy now", "sku", "price")
_NEWSLETTER_WORDS = ("unsubscribe", "newsletter", "view in browser")


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self.title: list[str] = []
        self._skip = 0
        self._in_title = False
        self._anchor: list[str] | None = None
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            values = {key: value or "" for key, value in attrs}
            self._href = values.get("href", "")
            self._anchor = []

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._anchor is not None:
            text = " ".join("".join(self._anchor).split())
            if self._href:
                self.links.append({"href": self._href, "text": text})
            self._anchor = None
            self._href = ""

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title.append(data)
        if self._anchor is not None:
            self._anchor.append(data)
        if data.strip():
            self.parts.append(data.strip())


class ExtractHtmlTool:
    name = "extract_html"
    description = "Extract title, visible text, and links from HTML, then label it as a product page, newsletter, or article."

    def run(self, payload: dict, ctx) -> dict:
        del ctx
        if detect_kind(payload) not in {"html", "text"} and not str(payload.get("text") or "").lstrip().startswith("<"):
            raise ValidationFailed("extract_html expects an HTML document.")
        parser = _HTMLText()
        parser.feed(payload_text(payload))
        parser.close()
        text = "\n".join(parser.parts).strip()
        title = " ".join("".join(parser.title).split())
        lowered = text.lower()
        if any(word in lowered for word in _PRODUCT_WORDS):
            document_type = "product"
        elif any(word in lowered for word in _NEWSLETTER_WORDS):
            document_type = "newsletter"
        else:
            document_type = "article"
        structured = {
            "title": title,
            "document_type": document_type,
            "text": text,
            "links": parser.links,
        }
        return {
            "kind": "html",
            "title": title,
            "text": text,
            "document_type": document_type,
            "links": parser.links,
            "structured": structured,
            "warnings": [],
        }
