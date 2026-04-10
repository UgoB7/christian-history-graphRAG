from __future__ import annotations

import json
import logging
from typing import Callable, Iterable, Optional

from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.schema import get_schema

from christian_history_graphrag.models import EntityRecord


logger = logging.getLogger(__name__)


def _normalize_neo4j_uri(uri: str) -> str:
    if uri.startswith("neo4j://localhost"):
        normalized = "bolt://" + uri.removeprefix("neo4j://")
        logger.info("Normalizing local Neo4j URI from %s to %s", uri, normalized)
        return normalized
    if uri.startswith("neo4j://127.0.0.1"):
        normalized = "bolt://" + uri.removeprefix("neo4j://")
        logger.info("Normalizing local Neo4j URI from %s to %s", uri, normalized)
        return normalized
    return uri


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.database = database
        self.driver = GraphDatabase.driver(
            _normalize_neo4j_uri(uri),
            auth=(username, password),
        )

    def close(self) -> None:
        self.driver.close()

    def setup(self) -> None:
        statements = [
            "CREATE CONSTRAINT entity_qid IF NOT EXISTS FOR (n:Entity) REQUIRE n.wikidata_id IS UNIQUE",
            "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (n:Passage) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT statement_id IF NOT EXISTS FOR (n:Statement) REQUIRE n.statement_id IS UNIQUE",
            "CREATE CONSTRAINT source_document_id IF NOT EXISTS FOR (n:SourceDocument) REQUIRE n.source_id IS UNIQUE",
            "CREATE INDEX entity_time_start IF NOT EXISTS FOR (n:Entity) ON (n.time_start_year)",
            "CREATE INDEX entity_time_end IF NOT EXISTS FOR (n:Entity) ON (n.time_end_year)",
            "CREATE INDEX entity_kind IF NOT EXISTS FOR (n:Entity) ON (n.entity_kind)",
            "CREATE INDEX statement_property IF NOT EXISTS FOR (n:Statement) ON (n.property_id)",
            "CREATE INDEX statement_rank IF NOT EXISTS FOR (n:Statement) ON (n.rank)",
            "CREATE INDEX source_document_system IF NOT EXISTS FOR (n:SourceDocument) ON (n.source_system)",
        ]
        for statement in statements:
            self.driver.execute_query(statement, database_=self.database)
        self.create_passage_fulltext_index()

    def ensure_kg_indexes(self) -> None:
        statements = [
            "CREATE CONSTRAINT kg_document_path IF NOT EXISTS FOR (n:KgDocument) REQUIRE n.path IS UNIQUE",
            "CREATE INDEX kg_document_wikidata IF NOT EXISTS FOR (n:KgDocument) ON (n.wikidata_id)",
            "CREATE INDEX kg_chunk_index IF NOT EXISTS FOR (n:KgChunk) ON (n.index)",
        ]
        for statement in statements:
            self.driver.execute_query(statement, database_=self.database)
        self.create_kg_chunk_fulltext_index()

    def upsert_entities(
        self,
        records: Iterable[EntityRecord],
        progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        for record in records:
            self.driver.execute_query(
                """
                MERGE (e:Entity {wikidata_id: $wikidata_id})
                SET e.name = $name,
                    e.description = $description,
                    e.entity_kind = $entity_kind,
                    e.aliases = $aliases,
                    e.instance_of = $instance_of,
                    e.time_start = $time_start,
                    e.time_end = $time_end,
                    e.time_start_year = $time_start_year,
                    e.time_end_year = $time_end_year,
                    e.latitude = $latitude,
                    e.longitude = $longitude,
                    e.wikipedia_title = $wikipedia_title,
                    e.wikipedia_url = $wikipedia_url,
                    e.wikidata_url = $wikidata_url,
                    e.retrieved_at = $retrieved_at,
                    e.seed_distance = $seed_distance
                """,
                {
                    "wikidata_id": record.qid,
                    "name": record.label,
                    "description": record.description,
                    "entity_kind": record.entity_kind,
                    "aliases": record.aliases,
                    "instance_of": record.instance_of,
                    "time_start": record.start_time,
                    "time_end": record.end_time,
                    "time_start_year": record.start_year,
                    "time_end_year": record.end_year,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "wikipedia_title": record.wikipedia_title,
                    "wikipedia_url": record.wikipedia_url,
                    "wikidata_url": record.wikidata_url,
                    "retrieved_at": record.retrieved_at,
                    "seed_distance": record.seed_distance,
                },
                database_=self.database,
            )
            if progress:
                progress(1)

    def upsert_source_documents(
        self,
        records: Iterable[EntityRecord],
        progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        for record in records:
            document_count = 0
            for source_document in record.source_documents:
                self.driver.execute_query(
                    """
                    MERGE (d:SourceDocument {source_id: $source_id})
                    SET d.source_system = $source_system,
                        d.source_url = $source_url,
                        d.title = $title,
                        d.language = $language,
                        d.revision_id = $revision_id,
                        d.content_hash = $content_hash,
                        d.retrieved_at = $retrieved_at,
                        d.metadata_json = $metadata_json
                    """,
                    {
                        "source_id": source_document.source_id,
                        "source_system": source_document.source_system,
                        "source_url": source_document.source_url,
                        "title": source_document.title,
                        "language": source_document.language,
                        "revision_id": source_document.revision_id,
                        "content_hash": source_document.content_hash,
                        "retrieved_at": source_document.retrieved_at,
                        "metadata_json": json.dumps(
                            source_document.metadata,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                    database_=self.database,
                )
                self.driver.execute_query(
                    """
                    MATCH (e:Entity {wikidata_id: $wikidata_id})
                    MATCH (d:SourceDocument {source_id: $source_id})
                    MERGE (e)-[:HAS_SOURCE]->(d)
                    """,
                    {
                        "wikidata_id": record.qid,
                        "source_id": source_document.source_id,
                    },
                    database_=self.database,
                )
                document_count += 1
            if progress:
                progress(max(document_count, 1))

    def upsert_relations(
        self,
        records: Iterable[EntityRecord],
        progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        for record in records:
            relation_count = 0
            for relation in record.relations:
                self.driver.execute_query(
                    f"""
                    MATCH (source:Entity {{wikidata_id: $source_qid}})
                    MATCH (target:Entity {{wikidata_id: $target_qid}})
                    OPTIONAL MATCH (d:SourceDocument {{source_id: $source_document_id}})
                    MERGE (s:Statement {{statement_id: $statement_id}})
                    SET s.property_id = $property_id,
                        s.relation_type = $relation_type,
                        s.target_qid = $target_qid,
                        s.rank = $rank,
                        s.qualifiers_json = $qualifiers_json,
                        s.qualifier_count = $qualifier_count,
                        s.reference_count = $reference_count,
                        s.reference_property_ids = $reference_property_ids,
                        s.source_system = $source_system,
                        s.source_url = $source_url,
                        s.source_document_id = $source_document_id,
                        s.retrieved_at = $retrieved_at
                    MERGE (source)-[:HAS_STATEMENT]->(s)
                    MERGE (s)-[:TARGETS]->(target)
                    FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
                        MERGE (s)-[:SUPPORTED_BY]->(d)
                    )
                    MERGE (source)-[r:{relation.relation_type}]->(target)
                    SET r.wikidata_property = $property_id,
                        r.statement_id = $statement_id,
                        r.rank = $rank,
                        r.reference_count = $reference_count,
                        r.source_document_id = $source_document_id,
                        r.retrieved_at = $retrieved_at
                    """,
                    {
                        "source_qid": record.qid,
                        "target_qid": relation.target_qid,
                        "statement_id": relation.statement_id
                        or f"{record.qid}:{relation.property_id}:{relation.target_qid}",
                        "property_id": relation.property_id,
                        "relation_type": relation.relation_type,
                        "rank": relation.rank,
                        "qualifiers_json": json.dumps(
                            relation.qualifiers,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "qualifier_count": len(relation.qualifiers),
                        "reference_count": relation.reference_count,
                        "reference_property_ids": relation.reference_property_ids,
                        "source_system": relation.source_system,
                        "source_url": relation.source_url,
                        "source_document_id": relation.source_document_id,
                        "retrieved_at": relation.retrieved_at,
                    },
                    database_=self.database,
                )
                relation_count += 1
            if progress:
                progress(max(relation_count, 1))

    def upsert_passages(
        self,
        records: Iterable[EntityRecord],
        progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        for record in records:
            if not record.passages:
                if progress:
                    progress(1)
                continue
            self.driver.execute_query(
                """
                MATCH (e:Entity {wikidata_id: $wikidata_id})-[rel:HAS_PASSAGE]->(p:Passage)
                DETACH DELETE p
                """,
                {"wikidata_id": record.qid},
                database_=self.database,
            )
            for passage in record.passages:
                if not passage.text:
                    continue
                self.driver.execute_query(
                    """
                    MATCH (e:Entity {wikidata_id: $wikidata_id})
                    OPTIONAL MATCH (d:SourceDocument {source_id: $source_document_id})
                    MERGE (p:Passage {id: $passage_id})
                    SET p.title = $title,
                        p.url = $url,
                        p.language = $language,
                        p.chunk_index = $chunk_index,
                        p.text = $text,
                        p.source = 'wikipedia',
                        p.source_system = $source_system,
                        p.source_document_id = $source_document_id,
                        p.retrieved_at = $retrieved_at,
                        p.revision_id = $revision_id,
                        p.content_hash = $content_hash,
                        p.time_start_year = $time_start_year,
                        p.time_end_year = $time_end_year
                    MERGE (e)-[:HAS_PASSAGE]->(p)
                    FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
                        MERGE (p)-[:DERIVED_FROM]->(d)
                    )
                    """,
                    {
                        "wikidata_id": record.qid,
                        "passage_id": f"wikipedia:{record.qid}:{passage.chunk_index}",
                        "title": passage.page_title,
                        "url": passage.url,
                        "language": passage.language,
                        "chunk_index": passage.chunk_index,
                        "text": passage.text,
                        "source_system": passage.source_system,
                        "source_document_id": passage.source_document_id,
                        "retrieved_at": passage.retrieved_at,
                        "revision_id": passage.revision_id,
                        "content_hash": passage.content_hash,
                        "time_start_year": record.start_year,
                        "time_end_year": record.end_year,
                    },
                    database_=self.database,
                )
            if progress:
                progress(max(len(record.passages), 1))

    def create_vector_index(self, dimensions: int) -> None:
        create_vector_index(
            self.driver,
            "passage_embeddings",
            label="Passage",
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine",
            neo4j_database=self.database,
        )
        self.create_passage_fulltext_index()

    def create_kg_chunk_vector_index(self) -> None:
        result = self.driver.execute_query(
            """
            MATCH (c:KgChunk)
            WHERE c.embedding IS NOT NULL
            RETURN size(c.embedding) AS dimensions
            LIMIT 1
            """,
            database_=self.database,
        )
        if not result.records:
            return
        dimensions = result.records[0]["dimensions"]
        create_vector_index(
            self.driver,
            "kg_chunk_embeddings",
            label="KgChunk",
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine",
            neo4j_database=self.database,
        )
        self.create_kg_chunk_fulltext_index()

    def create_passage_fulltext_index(self) -> None:
        create_fulltext_index(
            self.driver,
            "passage_fulltext",
            label="Passage",
            node_properties=["title", "text"],
            neo4j_database=self.database,
        )

    def create_kg_chunk_fulltext_index(self) -> None:
        create_fulltext_index(
            self.driver,
            "kg_chunk_fulltext",
            label="KgChunk",
            node_properties=["text"],
            neo4j_database=self.database,
        )

    def reset_database(self) -> None:
        statements = [
            "DROP INDEX passage_embeddings IF EXISTS",
            "DROP INDEX kg_chunk_embeddings IF EXISTS",
            "DROP INDEX passage_fulltext IF EXISTS",
            "DROP INDEX kg_chunk_fulltext IF EXISTS",
            "MATCH (n) DETACH DELETE n",
        ]
        for statement in statements:
            self.driver.execute_query(statement, database_=self.database)

    def drop_vector_index(self) -> None:
        self.driver.execute_query(
            "DROP INDEX passage_embeddings IF EXISTS",
            database_=self.database,
        )

    def drop_kg_chunk_vector_index(self) -> None:
        self.driver.execute_query(
            "DROP INDEX kg_chunk_embeddings IF EXISTS",
            database_=self.database,
        )

    def drop_passage_fulltext_index(self) -> None:
        self.driver.execute_query(
            "DROP INDEX passage_fulltext IF EXISTS",
            database_=self.database,
        )

    def drop_kg_chunk_fulltext_index(self) -> None:
        self.driver.execute_query(
            "DROP INDEX kg_chunk_fulltext IF EXISTS",
            database_=self.database,
        )

    def clear_embeddings(self) -> None:
        self.driver.execute_query(
            """
            MATCH (p:Passage)
            REMOVE p.embedding
            """,
            database_=self.database,
        )

    def clear_kg_embeddings(self) -> None:
        self.driver.execute_query(
            """
            MATCH (c:KgChunk)
            REMOVE c.embedding
            """,
            database_=self.database,
        )

    def list_entities_for_kg_enrichment(
        self,
        qids: Optional[list[str]] = None,
        limit: int = 25,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (e:Entity)
            WHERE e.wikipedia_title IS NOT NULL
              AND ($qids IS NULL OR e.wikidata_id IN $qids)
              AND ($year_from IS NULL OR coalesce(e.time_end_year, e.time_start_year, 999999) >= $year_from)
              AND ($year_to IS NULL OR coalesce(e.time_start_year, e.time_end_year, -999999) <= $year_to)
            RETURN e.wikidata_id AS wikidata_id,
                   e.name AS name,
                   e.wikipedia_title AS wikipedia_title,
                   e.wikipedia_url AS wikipedia_url,
                   e.time_start_year AS time_start_year,
                   e.time_end_year AS time_end_year
            ORDER BY e.name
            LIMIT $limit
            """,
            {
                "qids": qids or None,
                "limit": limit,
                "year_from": year_from,
                "year_to": year_to,
            },
            database_=self.database,
        )
        return [record.data() for record in result.records]

    def delete_kg_subgraph_for_entity(self, wikidata_id: str) -> None:
        self.driver.execute_query(
            """
            MATCH (d:KgDocument {wikidata_id: $wikidata_id})
            OPTIONAL MATCH (c:KgChunk)-[:KG_FROM_DOCUMENT]->(d)
            WITH collect(DISTINCT c) AS chunks, collect(DISTINCT d) AS docs
            FOREACH (chunk IN chunks | DETACH DELETE chunk)
            FOREACH (doc IN docs | DETACH DELETE doc)
            """,
            {"wikidata_id": wikidata_id},
            database_=self.database,
        )
        self.driver.execute_query(
            """
            MATCH (n)
            WHERE NOT n:Entity
              AND NOT n:Passage
              AND NOT n:KgDocument
              AND NOT n:KgChunk
              AND NOT (n)--()
            DELETE n
            """,
            database_=self.database,
        )

    def link_entity_to_kg_document(self, wikidata_id: str, wikipedia_url: str) -> None:
        self.driver.execute_query(
            """
            MATCH (e:Entity {wikidata_id: $wikidata_id})
            MATCH (d:KgDocument {path: $wikipedia_url})
            MERGE (e)-[:HAS_KG_DOCUMENT]->(d)
            """,
            {
                "wikidata_id": wikidata_id,
                "wikipedia_url": wikipedia_url,
            },
            database_=self.database,
        )

    def list_entities_for_resolution(self) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (e:Entity)
            RETURN e.wikidata_id AS wikidata_id,
                   e.name AS name,
                   coalesce(e.aliases, []) AS aliases,
                   e.wikipedia_title AS wikipedia_title,
                   e.entity_kind AS entity_kind
            """,
            database_=self.database,
        )
        return [record.data() for record in result.records]

    def list_extracted_nodes_for_resolution(self) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (n)-[:KG_FROM_CHUNK]->(:KgChunk)
            WHERE NOT n:Entity
              AND NOT n:Passage
              AND NOT n:KgDocument
              AND NOT n:KgChunk
              AND n.name IS NOT NULL
            RETURN elementId(n) AS element_id,
                   labels(n) AS labels,
                   n.name AS name
            """,
            database_=self.database,
        )
        return [record.data() for record in result.records]

    def clear_resolution_links(self) -> None:
        self.driver.execute_query(
            """
            MATCH ()-[r:RESOLVES_TO]->()
            DELETE r
            """,
            database_=self.database,
        )

    def link_resolution(
        self,
        *,
        node_element_id: str,
        wikidata_id: str,
        method: str,
        score: float,
        matched_text: str,
    ) -> None:
        self.driver.execute_query(
            """
            MATCH (n) WHERE elementId(n) = $node_element_id
            MATCH (e:Entity {wikidata_id: $wikidata_id})
            MERGE (n)-[r:RESOLVES_TO]->(e)
            SET r.method = $method,
                r.score = $score,
                r.matched_text = $matched_text
            """,
            {
                "node_element_id": node_element_id,
                "wikidata_id": wikidata_id,
                "method": method,
                "score": score,
                "matched_text": matched_text,
            },
            database_=self.database,
        )

    def get_graph_schema(self) -> str:
        return get_schema(
            self.driver,
            database=self.database,
            sanitize=True,
        )

    def get_period_subgraph(
        self, year_from: Optional[int] = None, year_to: Optional[int] = None, limit: int = 50
    ) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (e:Entity)
            WHERE ($year_from IS NULL OR coalesce(e.time_end_year, e.time_start_year, 999999) >= $year_from)
              AND ($year_to IS NULL OR coalesce(e.time_start_year, e.time_end_year, -999999) <= $year_to)
            OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
            RETURN e, r, neighbor
            LIMIT $limit
            """,
            {"year_from": year_from, "year_to": year_to, "limit": limit},
            database_=self.database,
        )
        return [record.data() for record in result.records]
