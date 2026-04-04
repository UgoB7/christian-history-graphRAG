from __future__ import annotations

from typing import Iterable

from christian_history_graphrag.config import Settings
from christian_history_graphrag.models import EntityRecord
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.wikidata import WikidataClient
from christian_history_graphrag.wikipedia import WikipediaClient


def build_records(
    seed_qids: list[str],
    settings: Settings,
    max_depth: int = 1,
    fetch_wikipedia: bool = True,
) -> dict[str, EntityRecord]:
    wikidata = WikidataClient(language=settings.wikipedia_language)
    wikipedia = WikipediaClient(language=settings.wikipedia_language)
    records = wikidata.expand_subgraph(seed_qids=seed_qids, max_depth=max_depth)

    if fetch_wikipedia:
        for record in records.values():
            if not record.wikipedia_title:
                continue
            try:
                record.passages = wikipedia.fetch_passages(
                    record.wikipedia_title,
                    max_paragraphs=settings.wikipedia_max_paragraphs,
                    chunk_size=settings.passage_chunk_size,
                    paragraph_overlap=settings.passage_paragraph_overlap,
                )
            except Exception:
                continue

    return records


def persist_records(store: Neo4jStore, records: Iterable[EntityRecord]) -> None:
    materialized = list(records)
    store.setup()
    store.upsert_entities(materialized)
    store.upsert_passages(materialized)
    store.upsert_relations(materialized)
