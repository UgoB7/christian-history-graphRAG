from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

from christian_history_graphrag.config import Settings
from christian_history_graphrag.http_utils import FileHTTPCache, build_retry_session
from christian_history_graphrag.models import WikipediaPassage


logger = logging.getLogger(__name__)

INTRO_SECTION_TITLE = "Introduction"
IGNORED_SECTION_TITLES = {
    "references",
    "see also",
    "external links",
    "bibliography",
    "further reading",
    "notes",
    "citations",
    "sources",
}
DISALLOWED_CONTAINER_TAGS = {"table", "figure", "style", "script", "ol", "ul", "dl"}
DISALLOWED_CONTAINER_CLASSES = {
    "infobox",
    "navbox",
    "reference",
    "reflist",
    "toc",
    "thumb",
    "metadata",
}


@dataclass
class ParsedWikipediaParagraph:
    text: str
    section_title: str
    section_path: list[str] = field(default_factory=list)
    outgoing_links: list[str] = field(default_factory=list)


@dataclass
class ParsedWikipediaChunk:
    text: str
    section_title: str
    section_path: list[str] = field(default_factory=list)
    outgoing_links: list[str] = field(default_factory=list)


class WikipediaClient:
    def __init__(
        self,
        language: str = "en",
        session: Optional[requests.Session] = None,
        settings: Optional[Settings] = None,
    ):
        self.language = language if settings is None else settings.wikipedia_language
        self.settings = settings
        self.session = build_retry_session(
            user_agent="christian-history-graphrag/0.1",
            max_retries=settings.http_max_retries if settings else 4,
            backoff_factor=settings.http_backoff_factor if settings else 0.5,
            session=session,
        )
        self.cache = FileHTTPCache(
            settings.cache_dir if settings else ".graphrag/cache",
            enabled=settings.use_http_cache if settings else True,
            ttl_seconds=settings.cache_ttl_seconds if settings else 60 * 60 * 24 * 7,
        )

    @property
    def api_url(self) -> str:
        return f"https://{self.language}.wikipedia.org/w/api.php"

    def fetch_passages(
        self,
        page_title: str,
        max_paragraphs: int = 18,
        chunk_size: int = 1400,
        paragraph_overlap: int = 1,
    ) -> list[WikipediaPassage]:
        payload, paragraphs = self._fetch_article_payload_and_paragraphs(
            page_title=page_title,
            max_paragraphs=max_paragraphs,
        )
        chunks = self._chunk_sectioned_paragraphs(
            paragraphs,
            chunk_size=chunk_size,
            paragraph_overlap=paragraph_overlap,
        )

        url = f"https://{self.language}.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        revision_id = payload.get("parse", {}).get("revid")
        retrieved_at = datetime.now(timezone.utc).isoformat()
        rendered_text = self._render_sectioned_article_text(paragraphs)
        content_hash = hashlib.sha256(rendered_text.encode("utf-8")).hexdigest()
        source_document_id = f"wikipedia:{self.language}:{page_title.replace(' ', '_')}"
        passages = []
        for index, chunk in enumerate(chunks):
            passages.append(
                WikipediaPassage(
                    passage_id=f"wikipedia:{self.language}:{page_title.replace(' ', '_')}:{index}",
                    page_title=page_title,
                    url=url,
                    language=self.language,
                    chunk_index=index,
                    text=chunk.text,
                    section_title=chunk.section_title,
                    section_path=list(chunk.section_path),
                    outgoing_links=list(chunk.outgoing_links),
                    source_document_id=source_document_id,
                    retrieved_at=retrieved_at,
                    revision_id=revision_id,
                    content_hash=content_hash,
                )
            )
        return passages

    def fetch_article_text(self, page_title: str, max_paragraphs: int = 40) -> str:
        _, paragraphs = self._fetch_article_payload_and_paragraphs(
            page_title=page_title,
            max_paragraphs=max_paragraphs,
        )
        return self._render_sectioned_article_text(paragraphs)

    def fetch_source_metadata(
        self, page_title: str, max_paragraphs: int = 40
    ) -> dict[str, object]:
        payload, paragraphs = self._fetch_article_payload_and_paragraphs(
            page_title=page_title,
            max_paragraphs=max_paragraphs,
        )
        revision_id = payload.get("parse", {}).get("revid")
        text = self._render_sectioned_article_text(paragraphs)
        section_titles = list(
            dict.fromkeys(
                paragraph.section_title
                for paragraph in paragraphs
                if paragraph.section_title
            )
        )
        return {
            "source_id": f"wikipedia:{self.language}:{page_title.replace(' ', '_')}",
            "source_system": "wikipedia",
            "source_url": f"https://{self.language}.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
            "title": page_title,
            "language": self.language,
            "revision_id": revision_id,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "paragraph_count": len(paragraphs),
                "section_count": len(section_titles),
                "section_titles": section_titles,
            },
        }

    def _fetch_article_payload_and_paragraphs(
        self,
        *,
        page_title: str,
        max_paragraphs: int,
    ) -> tuple[dict, list[ParsedWikipediaParagraph]]:
        payload = self._fetch_parse_payload(page_title=page_title, max_paragraphs=max_paragraphs)
        html = payload["parse"]["text"]
        paragraphs = self._html_to_sectioned_paragraphs(html, max_paragraphs=max_paragraphs)
        return payload, paragraphs

    def _fetch_parse_payload(self, page_title: str, max_paragraphs: int) -> dict:
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "text|revid",
            "format": "json",
            "formatversion": "2",
        }
        cached_payload = self.cache.get_json("wikipedia-parse", self.api_url, params)
        if cached_payload is not None:
            return cached_payload

        response = self.session.get(
            self.api_url,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            logger.warning("Wikipedia API error for %s: %s", page_title, payload["error"])
        self.cache.set_json("wikipedia-parse", self.api_url, params, payload)
        return payload

    def _html_to_sectioned_paragraphs(
        self,
        html: str,
        max_paragraphs: int,
    ) -> list[ParsedWikipediaParagraph]:
        soup = BeautifulSoup(html, "html.parser")
        container = soup.select_one("div.mw-parser-output") or soup
        paragraphs: list[ParsedWikipediaParagraph] = []
        section_stack: list[tuple[int, str]] = []
        ignored_level: Optional[int] = None

        for element in self._iter_content_elements(container):
            tag_name = (element.name or "").lower()
            if tag_name in {"h2", "h3", "h4"}:
                title = self._extract_heading_title(element)
                if not title:
                    continue
                level = int(tag_name[1])
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                if ignored_level is not None and level <= ignored_level:
                    ignored_level = None
                if title.casefold() in IGNORED_SECTION_TITLES:
                    ignored_level = level
                    continue
                if ignored_level is None:
                    section_stack.append((level, title))
                continue

            if tag_name != "p" or ignored_level is not None:
                continue

            text = self._clean_paragraph_text(element.get_text(" ", strip=True))
            if len(text) < 40:
                continue

            section_path = [title for _, title in section_stack] or [INTRO_SECTION_TITLE]
            paragraphs.append(
                ParsedWikipediaParagraph(
                    text=text,
                    section_title=section_path[-1],
                    section_path=section_path,
                    outgoing_links=self._extract_wikipedia_links(element),
                )
            )
            if len(paragraphs) >= max_paragraphs:
                break
        return paragraphs

    def _iter_content_elements(self, container):
        for element in container.descendants:
            tag_name = getattr(element, "name", None)
            if tag_name is None:
                continue
            normalized_tag = tag_name.lower()
            if normalized_tag not in {"h2", "h3", "h4", "p"}:
                continue
            if self._is_inside_disallowed_container(element, container):
                continue
            yield element

    def _is_inside_disallowed_container(self, element, container) -> bool:
        for parent in element.parents:
            if parent == container:
                return False
            parent_name = getattr(parent, "name", None)
            if parent_name and parent_name.lower() in DISALLOWED_CONTAINER_TAGS:
                return True
            parent_classes = set(parent.get("class", []))
            if parent_classes & DISALLOWED_CONTAINER_CLASSES:
                return True
        return False

    def _extract_heading_title(self, element) -> Optional[str]:
        headline = element.find(class_="mw-headline")
        if headline is not None:
            text = headline.get_text(" ", strip=True)
        else:
            text = element.get_text(" ", strip=True)
        normalized = re.sub(r"\s+", " ", text).strip()
        return normalized or None

    def _clean_paragraph_text(self, text: str) -> str:
        normalized = re.sub(r"\[\d+\]", "", text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _extract_wikipedia_links(self, paragraph) -> list[str]:
        links: list[str] = []
        seen: set[str] = set()
        for anchor in paragraph.find_all("a", href=True):
            href = anchor["href"]
            if not href.startswith("/wiki/"):
                continue
            page_ref = href.removeprefix("/wiki/")
            if not page_ref or ":" in page_ref:
                continue
            title = unquote(page_ref).replace("_", " ").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            links.append(title)
        return links

    def _chunk_sectioned_paragraphs(
        self,
        paragraphs: list[ParsedWikipediaParagraph],
        chunk_size: int,
        paragraph_overlap: int,
    ) -> list[ParsedWikipediaChunk]:
        if not paragraphs:
            return []

        chunks: list[ParsedWikipediaChunk] = []
        cursor = 0
        while cursor < len(paragraphs):
            section_path = paragraphs[cursor].section_path
            section_rows: list[ParsedWikipediaParagraph] = []
            while cursor < len(paragraphs) and paragraphs[cursor].section_path == section_path:
                section_rows.append(paragraphs[cursor])
                cursor += 1
            chunks.extend(
                self._chunk_single_section(
                    section_rows,
                    chunk_size=chunk_size,
                    paragraph_overlap=paragraph_overlap,
                )
            )
        return chunks

    def _chunk_single_section(
        self,
        section_rows: list[ParsedWikipediaParagraph],
        *,
        chunk_size: int,
        paragraph_overlap: int,
    ) -> list[ParsedWikipediaChunk]:
        chunks: list[ParsedWikipediaChunk] = []
        start = 0
        while start < len(section_rows):
            current: list[ParsedWikipediaParagraph] = []
            current_length = 0
            cursor = start

            while cursor < len(section_rows):
                paragraph = section_rows[cursor]
                projected = current_length + len(paragraph.text) + (2 if current else 0)
                if current and projected > chunk_size:
                    break
                current.append(paragraph)
                current_length = projected
                cursor += 1

            if current:
                outgoing_links = list(
                    dict.fromkeys(
                        link
                        for row in current
                        for link in row.outgoing_links
                    )
                )
                chunks.append(
                    ParsedWikipediaChunk(
                        text="\n\n".join(row.text for row in current),
                        section_title=current[0].section_title,
                        section_path=list(current[0].section_path),
                        outgoing_links=outgoing_links,
                    )
                )

            if cursor >= len(section_rows):
                break

            start = max(start + 1, cursor - max(paragraph_overlap, 0))

        return chunks

    def _render_sectioned_article_text(
        self,
        paragraphs: list[ParsedWikipediaParagraph],
    ) -> str:
        if not paragraphs:
            return ""

        rendered_parts: list[str] = []
        current_section: list[str] | None = None
        for paragraph in paragraphs:
            if paragraph.section_path != current_section:
                current_section = list(paragraph.section_path)
                if rendered_parts:
                    rendered_parts.append("")
                if current_section != [INTRO_SECTION_TITLE]:
                    rendered_parts.append("## " + " > ".join(current_section))
                    rendered_parts.append("")
            rendered_parts.append(paragraph.text)
            rendered_parts.append("")
        while rendered_parts and not rendered_parts[-1]:
            rendered_parts.pop()
        return "\n".join(rendered_parts)
