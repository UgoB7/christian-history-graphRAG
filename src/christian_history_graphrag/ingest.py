from __future__ import annotations

from typing import Callable, Iterable, Optional

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
    wikipedia_progress: Optional[Callable[[int], None]] = None,
) -> dict[str, EntityRecord]:
    wikidata = WikidataClient(language=settings.wikipedia_language)
    records = wikidata.expand_subgraph(seed_qids=seed_qids, max_depth=max_depth)

    if fetch_wikipedia:
        populate_wikipedia_passages(
            records=records,
            settings=settings,
            progress=wikipedia_progress,
        )

    return records


def populate_wikipedia_passages(
    records: dict[str, EntityRecord],
    settings: Settings,
    progress: Optional[Callable[[int], None]] = None,
) -> None:
    wikipedia = WikipediaClient(language=settings.wikipedia_language)
    for record in records.values():
        if not record.wikipedia_title:
            if progress:
                progress(1)
            continue
        try:
            record.passages = wikipedia.fetch_passages(
                record.wikipedia_title,
                max_paragraphs=settings.wikipedia_max_paragraphs,
                chunk_size=settings.passage_chunk_size,
                paragraph_overlap=settings.passage_paragraph_overlap,
            )
        except Exception:
            pass
        finally:
            if progress:
                progress(1)


def persist_records(
    store: Neo4jStore,
    records: Iterable[EntityRecord],
    entity_progress: Optional[Callable[[int], None]] = None,
    passage_progress: Optional[Callable[[int], None]] = None,
    relation_progress: Optional[Callable[[int], None]] = None,
) -> None:
    materialized = list(records)
    store.setup()
    store.upsert_entities(materialized, progress=entity_progress)
    store.upsert_passages(materialized, progress=passage_progress)
    store.upsert_relations(materialized, progress=relation_progress)
