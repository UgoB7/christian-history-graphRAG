from __future__ import annotations

import asyncio
import logging
import re
from typing import Callable, Optional

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from christian_history_graphrag.config import Settings
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.providers import build_embedder, build_llm
from christian_history_graphrag.wikipedia import WikipediaClient

KG_DOCUMENT_LABEL = "KgDocument"
KG_CHUNK_LABEL = "KgChunk"
KG_FROM_DOCUMENT = "KG_FROM_DOCUMENT"
KG_NEXT_CHUNK = "KG_NEXT_CHUNK"
KG_FROM_CHUNK = "KG_FROM_CHUNK"
EXTRACTOR_LOGGER_NAME = (
    "neo4j_graphrag.experimental.components.entity_relation_extractor"
)

CHRISTIAN_HISTORY_SCHEMA = {
    "node_types": [
        {"label": "Person", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        {"label": "Place", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        {"label": "Organization", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        {"label": "Event", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        {"label": "Doctrine", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        {"label": "Text", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        {"label": "Office", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        {"label": "Group", "properties": [{"name": "name", "type": "STRING", "required": True}]},
        {"label": "Concept", "properties": [{"name": "name", "type": "STRING", "required": True}]},
    ],
    "relationship_types": [
        "INFLUENCED",
        "OPPOSED",
        "WROTE",
        "TAUGHT",
        "FOUNDED",
        "ATTENDED",
        "PARTICIPATED_IN",
        "TOOK_PLACE_IN",
        "BELONGED_TO",
        "HELD_OFFICE",
        "AFFIRMED",
        "CONDEMNED",
        "COMMENTED_ON",
        "MENTIONED_IN",
        "ASSOCIATED_WITH",
    ],
    "patterns": [
        ["Person", "INFLUENCED", "Person"],
        ["Person", "OPPOSED", "Person"],
        ["Person", "WROTE", "Text"],
        ["Person", "TAUGHT", "Doctrine"],
        ["Person", "FOUNDED", "Organization"],
        ["Person", "ATTENDED", "Event"],
        ["Person", "PARTICIPATED_IN", "Event"],
        ["Event", "TOOK_PLACE_IN", "Place"],
        ["Person", "BELONGED_TO", "Group"],
        ["Person", "HELD_OFFICE", "Office"],
        ["Organization", "AFFIRMED", "Doctrine"],
        ["Organization", "CONDEMNED", "Doctrine"],
        ["Text", "COMMENTED_ON", "Doctrine"],
        ["Concept", "MENTIONED_IN", "Text"],
        ["Doctrine", "ASSOCIATED_WITH", "Group"],
    ],
    "additional_node_types": True,
}


def get_kg_schema(settings: Settings):
    mode = settings.kg_builder_schema_mode.strip().upper()
    if mode == "FREE":
        return "FREE"
    if mode == "EXTRACTED":
        return "EXTRACTED"
    return CHRISTIAN_HISTORY_SCHEMA


def get_lexical_graph_config() -> LexicalGraphConfig:
    return LexicalGraphConfig(
        document_node_label=KG_DOCUMENT_LABEL,
        chunk_node_label=KG_CHUNK_LABEL,
        chunk_to_document_relationship_type=KG_FROM_DOCUMENT,
        next_chunk_relationship_type=KG_NEXT_CHUNK,
        node_to_chunk_relationship_type=KG_FROM_CHUNK,
    )


class KGBuilderLogTracker(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.invalid_json_chunks: set[int] = set()
        self.improper_format_chunks: set[int] = set()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        match = re.search(r"chunk_index=(\d+)", message)
        chunk_index = int(match.group(1)) if match else None
        if "not valid JSON" in message and chunk_index is not None:
            self.invalid_json_chunks.add(chunk_index)
        if "improper format" in message and chunk_index is not None:
            self.improper_format_chunks.add(chunk_index)


async def enrich_entities_with_kg_builder(
    store: Neo4jStore,
    settings: Settings,
    qids: Optional[list[str]] = None,
    limit: int = 25,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    replace_existing: bool = True,
    progress: Optional[Callable[[int], None]] = None,
    reporter: Optional[Callable[[str], None]] = None,
) -> int:
    candidates = store.list_entities_for_kg_enrichment(
        qids=qids,
        limit=limit,
        year_from=year_from,
        year_to=year_to,
    )
    if not candidates:
        return 0

    llm = build_llm(settings, model_name=settings.kg_builder_llm_model)
    embedder = build_embedder(settings)
    splitter = FixedSizeSplitter(
        chunk_size=settings.kg_builder_chunk_size,
        chunk_overlap=settings.kg_builder_chunk_overlap,
    )
    pipeline = SimpleKGPipeline(
        llm=llm,
        driver=store.driver,
        embedder=embedder,
        from_pdf=False,
        text_splitter=splitter,
        schema=get_kg_schema(settings),
        lexical_graph_config=get_lexical_graph_config(),
        perform_entity_resolution=True,
        on_error="IGNORE",
        neo4j_database=store.database,
    )
    wikipedia = WikipediaClient(language=settings.wikipedia_language)
    extractor_logger = logging.getLogger(EXTRACTOR_LOGGER_NAME)

    enriched = 0
    for entity in candidates:
        if reporter:
            reporter(
                f"Entity {enriched + 1}/{len(candidates)}: {entity['name']} "
                f"({entity['wikidata_id']})"
            )
            reporter(f"  Wikipedia page: {entity['wikipedia_title']}")

        article_text = wikipedia.fetch_article_text(
            entity["wikipedia_title"],
            max_paragraphs=settings.kg_builder_max_paragraphs,
        )
        if not article_text:
            if reporter:
                reporter("  Skipped: empty Wikipedia text")
            continue

        estimated_chunks = await splitter.run(article_text)
        if reporter:
            reporter(
                f"  Sending {len(estimated_chunks.chunks)} chunks to KG Builder "
                f"(chunk_size={settings.kg_builder_chunk_size}, overlap={settings.kg_builder_chunk_overlap})"
            )

        if replace_existing:
            store.delete_kg_subgraph_for_entity(entity["wikidata_id"])

        document_metadata = {
            "wikidata_id": str(entity["wikidata_id"]),
            "entity_name": str(entity["name"]),
            "wikipedia_title": str(entity["wikipedia_title"]),
            "wikipedia_url": str(entity["wikipedia_url"]),
            "source": "wikipedia_kg_builder",
        }
        if entity.get("time_start_year") is not None:
            document_metadata["time_start_year"] = str(entity["time_start_year"])
        if entity.get("time_end_year") is not None:
            document_metadata["time_end_year"] = str(entity["time_end_year"])

        tracker = KGBuilderLogTracker()
        extractor_logger.addHandler(tracker)
        try:
            try:
                await pipeline.run_async(
                    file_path=entity["wikipedia_url"],
                    text=article_text,
                    document_metadata=document_metadata,
                )
            except LLMGenerationError as exc:
                message = str(exc)
                if "model failed to load" in message and reporter:
                    reporter(
                        "  Ollama n'a pas pu charger le modele du KG Builder. "
                        f"Modele configure: {settings.kg_builder_llm_model}"
                    )
                    reporter(
                        "  Conseil: utilise un modele plus leger pour "
                        "KG_BUILDER_LLM_MODEL, par ex. qwen2.5:3b."
                    )
                raise
        finally:
            extractor_logger.removeHandler(tracker)

        store.link_entity_to_kg_document(
            wikidata_id=entity["wikidata_id"],
            wikipedia_url=entity["wikipedia_url"],
        )
        enriched += 1
        if reporter:
            proper_failures = sorted(tracker.improper_format_chunks)
            json_failures = sorted(tracker.invalid_json_chunks)
            total_failures = len(set(proper_failures) | set(json_failures))
            reporter(
                f"  Completed: {len(estimated_chunks.chunks) - total_failures}/"
                f"{len(estimated_chunks.chunks)} chunks extracted cleanly"
            )
            if json_failures:
                reporter(
                    f"  Invalid JSON chunks: {', '.join(str(index) for index in json_failures)}"
                )
            if proper_failures:
                reporter(
                    f"  Improper format chunks: {', '.join(str(index) for index in proper_failures)}"
                )
        if progress:
            progress(1)

    store.ensure_kg_indexes()
    store.create_kg_chunk_vector_index()
    return enriched


def run_kg_builder_enrichment(
    store: Neo4jStore,
    settings: Settings,
    qids: Optional[list[str]] = None,
    limit: int = 25,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    replace_existing: bool = True,
    progress: Optional[Callable[[int], None]] = None,
    reporter: Optional[Callable[[str], None]] = None,
) -> int:
    return asyncio.run(
        enrich_entities_with_kg_builder(
            store=store,
            settings=settings,
            qids=qids,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            replace_existing=replace_existing,
            progress=progress,
            reporter=reporter,
        )
    )
