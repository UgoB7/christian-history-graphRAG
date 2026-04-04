from __future__ import annotations

import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from christian_history_graphrag.models import WikipediaPassage


class WikipediaClient:
    def __init__(self, language: str = "en", session: Optional[requests.Session] = None):
        self.language = language
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "christian-history-graphrag/0.1",
            }
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
        response = self.session.get(
            self.api_url,
            params={
                "action": "parse",
                "page": page_title,
                "prop": "text",
                "format": "json",
                "formatversion": "2",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        html = payload["parse"]["text"]
        paragraphs = self._html_to_paragraphs(html, max_paragraphs=max_paragraphs)
        chunks = self._chunk_paragraphs(
            paragraphs,
            chunk_size=chunk_size,
            paragraph_overlap=paragraph_overlap,
        )

        url = f"https://{self.language}.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
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
                )
            )
        return passages

    def fetch_article_text(self, page_title: str, max_paragraphs: int = 40) -> str:
        response = self.session.get(
            self.api_url,
            params={
                "action": "parse",
                "page": page_title,
                "prop": "text",
                "format": "json",
                "formatversion": "2",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        html = payload["parse"]["text"]
        paragraphs = self._html_to_paragraphs(html, max_paragraphs=max_paragraphs)
        return "\n\n".join(paragraphs)

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
