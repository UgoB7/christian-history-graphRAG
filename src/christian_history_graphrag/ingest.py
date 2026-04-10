from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

from christian_history_graphrag.checkpoints import IngestCheckpointManager
from christian_history_graphrag.config import Settings
from christian_history_graphrag.models import EntityRecord, SourceDocument
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.wikidata import WikidataClient
from christian_history_graphrag.wikipedia import WikipediaClient


logger = logging.getLogger(__name__)


def build_records(
    seed_qids: list[str],
    settings: Settings,
    max_depth: int = 1,
    fetch_wikipedia: bool = True,
    wikipedia_progress: Optional[Callable[[int], None]] = None,
    checkpoint_manager: Optional[IngestCheckpointManager] = None,
) -> dict[str, EntityRecord]:
    if checkpoint_manager:
        cached_records = checkpoint_manager.load_stage("wikipedia" if fetch_wikipedia else "wikidata")
        if cached_records is not None:
            logger.info("Loaded ingest checkpoint for stage %s", "wikipedia" if fetch_wikipedia else "wikidata")
            return cached_records

        cached_records = checkpoint_manager.load_stage("wikidata")
        if cached_records is not None:
            records = cached_records
            logger.info("Loaded ingest checkpoint for stage wikidata")
        else:
            wikidata = WikidataClient(settings=settings)
            records = wikidata.expand_subgraph(seed_qids=seed_qids, max_depth=max_depth)
            checkpoint_manager.save_stage("wikidata", records)
    else:
        wikidata = WikidataClient(settings=settings)
        records = wikidata.expand_subgraph(seed_qids=seed_qids, max_depth=max_depth)

    if fetch_wikipedia:
        populate_wikipedia_passages(
            records=records,
            settings=settings,
            progress=wikipedia_progress,
            checkpoint_manager=checkpoint_manager,
        )

    return records


def populate_wikipedia_passages(
    records: dict[str, EntityRecord],
    settings: Settings,
    progress: Optional[Callable[[int], None]] = None,
    checkpoint_manager: Optional[IngestCheckpointManager] = None,
) -> None:
    if checkpoint_manager:
        cached_records = checkpoint_manager.load_stage("wikipedia")
        if cached_records is not None:
            records.clear()
            records.update(cached_records)
            logger.info("Loaded ingest checkpoint for stage wikipedia")
            if progress:
                progress(len(records))
            return

    wikipedia = WikipediaClient(settings=settings)
    for record in records.values():
        if not record.wikipedia_title:
            if progress:
                progress(1)
            continue
        try:
            source_metadata = wikipedia.fetch_source_metadata(
                record.wikipedia_title,
                max_paragraphs=settings.wikipedia_max_paragraphs,
            )
            record.passages = wikipedia.fetch_passages(
                record.wikipedia_title,
                max_paragraphs=settings.wikipedia_max_paragraphs,
                chunk_size=settings.passage_chunk_size,
                paragraph_overlap=settings.passage_paragraph_overlap,
            )
            existing_source_ids = {source.source_id for source in record.source_documents}
            if source_metadata["source_id"] not in existing_source_ids:
                record.source_documents.append(
                    SourceDocument(
                        source_id=source_metadata["source_id"],
                        source_system=source_metadata["source_system"],
                        source_url=source_metadata["source_url"],
                        title=source_metadata["title"],
                        language=source_metadata.get("language"),
                        revision_id=source_metadata.get("revision_id"),
                        content_hash=source_metadata.get("content_hash"),
                        retrieved_at=source_metadata.get("retrieved_at"),
                        metadata=dict(source_metadata.get("metadata", {})),
                    )
                )
        except Exception as exc:
            logger.warning(
                "Wikipedia fetch failed for %s (%s): %s",
                record.qid,
                record.wikipedia_title,
                exc,
            )
        finally:
            if progress:
                progress(1)
    if checkpoint_manager:
        checkpoint_manager.save_stage("wikipedia", records)


def persist_records(
    store: Neo4jStore,
    records: Iterable[EntityRecord],
    entity_progress: Optional[Callable[[int], None]] = None,
    source_progress: Optional[Callable[[int], None]] = None,
    passage_progress: Optional[Callable[[int], None]] = None,
    relation_progress: Optional[Callable[[int], None]] = None,
) -> None:
    materialized = list(records)
    store.setup()
    store.upsert_entities(materialized, progress=entity_progress)
    store.upsert_source_documents(materialized, progress=source_progress)
    store.upsert_passages(materialized, progress=passage_progress)
    store.upsert_relations(materialized, progress=relation_progress)
