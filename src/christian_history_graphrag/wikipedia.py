from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from christian_history_graphrag.config import Settings
from christian_history_graphrag.http_utils import FileHTTPCache, build_retry_session
from christian_history_graphrag.models import WikipediaPassage


logger = logging.getLogger(__name__)


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
        payload = self._fetch_parse_payload(
            page_title=page_title,
            max_paragraphs=max_paragraphs,
        )
        html = payload["parse"]["text"]
        paragraphs = self._html_to_paragraphs(html, max_paragraphs=max_paragraphs)
        chunks = self._chunk_paragraphs(
            paragraphs,
            chunk_size=chunk_size,
            paragraph_overlap=paragraph_overlap,
        )

        url = f"https://{self.language}.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        revision_id = payload.get("parse", {}).get("revid")
        retrieved_at = datetime.now(timezone.utc).isoformat()
        content_hash = hashlib.sha256(
            "\n\n".join(paragraphs).encode("utf-8")
        ).hexdigest()
        source_document_id = f"wikipedia:{self.language}:{page_title.replace(' ', '_')}"
        passages = []
        for index, chunk in enumerate(chunks):
            passages.append(
                WikipediaPassage(
                    passage_id=f"{page_title.replace(' ', '_')}:{index}",
                    page_title=page_title,
                    url=url,
                    language=self.language,
                    chunk_index=index,
                    text=chunk,
                    source_document_id=source_document_id,
                    retrieved_at=retrieved_at,
                    revision_id=revision_id,
                    content_hash=content_hash,
                )
            )
        return passages

    def fetch_article_text(self, page_title: str, max_paragraphs: int = 40) -> str:
        payload = self._fetch_parse_payload(page_title=page_title, max_paragraphs=max_paragraphs)
        html = payload["parse"]["text"]
        paragraphs = self._html_to_paragraphs(html, max_paragraphs=max_paragraphs)
        return "\n\n".join(paragraphs)

    def fetch_source_metadata(
        self, page_title: str, max_paragraphs: int = 40
    ) -> dict[str, object]:
        payload = self._fetch_parse_payload(page_title=page_title, max_paragraphs=max_paragraphs)
        html = payload["parse"]["text"]
        paragraphs = self._html_to_paragraphs(html, max_paragraphs=max_paragraphs)
        revision_id = payload.get("parse", {}).get("revid")
        text = "\n\n".join(paragraphs)
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
            },
        }

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

    def _html_to_paragraphs(self, html: str, max_paragraphs: int) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = []
        for paragraph in soup.find_all("p"):
            text = paragraph.get_text(" ", strip=True)
            text = re.sub(r"\[\d+\]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 40:
                paragraphs.append(text)
            if len(paragraphs) >= max_paragraphs:
                break
        return paragraphs

    def _chunk_paragraphs(
        self,
        paragraphs: list[str],
        chunk_size: int,
        paragraph_overlap: int,
    ) -> list[str]:
        if not paragraphs:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(paragraphs):
            current: list[str] = []
            current_length = 0
            cursor = start

            while cursor < len(paragraphs):
                paragraph = paragraphs[cursor]
                projected = current_length + len(paragraph) + (2 if current else 0)
                if current and projected > chunk_size:
                    break
                current.append(paragraph)
                current_length = projected
                cursor += 1

            if current:
                chunks.append("\n\n".join(current))

            if cursor >= len(paragraphs):
                break

            start = max(start + 1, cursor - max(paragraph_overlap, 0))

        return chunks
