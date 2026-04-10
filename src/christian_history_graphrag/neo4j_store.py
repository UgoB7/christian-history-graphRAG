from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable, Optional

from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.schema import get_schema

from christian_history_graphrag.models import ClaimRecord, CommunityReportRecord, EntityRecord


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
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.claim_id IS UNIQUE",
            "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (n:Community) REQUIRE n.community_id IS UNIQUE",
            "CREATE CONSTRAINT community_report_id IF NOT EXISTS FOR (n:CommunityReport) REQUIRE n.report_id IS UNIQUE",
            "CREATE INDEX entity_time_start IF NOT EXISTS FOR (n:Entity) ON (n.time_start_year)",
            "CREATE INDEX entity_time_end IF NOT EXISTS FOR (n:Entity) ON (n.time_end_year)",
            "CREATE INDEX entity_kind IF NOT EXISTS FOR (n:Entity) ON (n.entity_kind)",
            "CREATE INDEX statement_property IF NOT EXISTS FOR (n:Statement) ON (n.property_id)",
            "CREATE INDEX statement_rank IF NOT EXISTS FOR (n:Statement) ON (n.rank)",
            "CREATE INDEX source_document_system IF NOT EXISTS FOR (n:SourceDocument) ON (n.source_system)",
            "CREATE INDEX claim_type IF NOT EXISTS FOR (n:Claim) ON (n.claim_type)",
            "CREATE INDEX claim_root_entity IF NOT EXISTS FOR (n:Claim) ON (n.root_entity_qid)",
            "CREATE INDEX community_focus_entity IF NOT EXISTS FOR (n:Community) ON (n.focus_entity_qid)",
            "CREATE INDEX community_report_focus_entity IF NOT EXISTS FOR (n:CommunityReport) ON (n.focus_entity_qid)",
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

    def ensure_claim_indexes(self) -> None:
        self.setup()
        self.create_claim_fulltext_index()

    def ensure_community_indexes(self) -> None:
        self.setup()
        self.create_community_report_fulltext_index()

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
                        p.section_title = $section_title,
                        p.section_path = $section_path,
                        p.section_path_text = $section_path_text,
                        p.outgoing_links = $outgoing_links,
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
                        "passage_id": passage.passage_id or f"wikipedia:{record.qid}:{passage.chunk_index}",
                        "title": passage.page_title,
                        "url": passage.url,
                        "language": passage.language,
                        "chunk_index": passage.chunk_index,
                        "text": passage.text,
                        "section_title": passage.section_title,
                        "section_path": passage.section_path,
                        "section_path_text": " > ".join(passage.section_path),
                        "outgoing_links": passage.outgoing_links,
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

    def create_claim_vector_index(self, dimensions: int) -> None:
        create_vector_index(
            self.driver,
            "claim_embeddings",
            label="Claim",
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine",
            neo4j_database=self.database,
        )
        self.create_claim_fulltext_index()

    def create_community_report_vector_index(self, dimensions: int) -> None:
        create_vector_index(
            self.driver,
            "community_report_embeddings",
            label="CommunityReport",
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine",
            neo4j_database=self.database,
        )
        self.create_community_report_fulltext_index()

    def create_passage_fulltext_index(self) -> None:
        create_fulltext_index(
            self.driver,
            "passage_fulltext",
            label="Passage",
            node_properties=["title", "section_title", "section_path_text", "text"],
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

    def create_claim_fulltext_index(self) -> None:
        create_fulltext_index(
            self.driver,
            "claim_fulltext",
            label="Claim",
            node_properties=["claim_text", "subject", "predicate", "object_value", "provenance_quote"],
            neo4j_database=self.database,
        )

    def create_community_report_fulltext_index(self) -> None:
        create_fulltext_index(
            self.driver,
            "community_report_fulltext",
            label="CommunityReport",
            node_properties=["title", "summary", "report_text", "themes_text", "key_entities_text"],
            neo4j_database=self.database,
        )

    def reset_database(self) -> None:
        statements = [
            "DROP INDEX passage_embeddings IF EXISTS",
            "DROP INDEX kg_chunk_embeddings IF EXISTS",
            "DROP INDEX claim_embeddings IF EXISTS",
            "DROP INDEX community_report_embeddings IF EXISTS",
            "DROP INDEX passage_fulltext IF EXISTS",
            "DROP INDEX kg_chunk_fulltext IF EXISTS",
            "DROP INDEX claim_fulltext IF EXISTS",
            "DROP INDEX community_report_fulltext IF EXISTS",
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

    def drop_claim_vector_index(self) -> None:
        self.driver.execute_query(
            "DROP INDEX claim_embeddings IF EXISTS",
            database_=self.database,
        )

    def drop_community_report_vector_index(self) -> None:
        self.driver.execute_query(
            "DROP INDEX community_report_embeddings IF EXISTS",
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

    def drop_claim_fulltext_index(self) -> None:
        self.driver.execute_query(
            "DROP INDEX claim_fulltext IF EXISTS",
            database_=self.database,
        )

    def drop_community_report_fulltext_index(self) -> None:
        self.driver.execute_query(
            "DROP INDEX community_report_fulltext IF EXISTS",
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

    def clear_claim_embeddings(self) -> None:
        self.driver.execute_query(
            """
            MATCH (c:Claim)
            REMOVE c.embedding
            """,
            database_=self.database,
        )

    def clear_community_report_embeddings(self) -> None:
        self.driver.execute_query(
            """
            MATCH (r:CommunityReport)
            REMOVE r.embedding
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
            OPTIONAL MATCH (:Entity {wikidata_id: $wikidata_id})-[:HAS_CLAIM]->(claim:Claim)
            OPTIONAL MATCH (:Entity {wikidata_id: $wikidata_id})-[:HAS_COMMUNITY]->(community:Community)
            OPTIONAL MATCH (community)-[:HAS_REPORT]->(report:CommunityReport)
            WITH collect(DISTINCT c) AS chunks,
                 collect(DISTINCT d) AS docs,
                 collect(DISTINCT claim) AS claims,
                 collect(DISTINCT community) AS communities,
                 collect(DISTINCT report) AS reports
            FOREACH (chunk IN chunks | DETACH DELETE chunk)
            FOREACH (doc IN docs | DETACH DELETE doc)
            FOREACH (claim IN claims | DETACH DELETE claim)
            FOREACH (community IN communities | DETACH DELETE community)
            FOREACH (report IN reports | DETACH DELETE report)
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

    def list_entities_with_kg_documents(
        self,
        qids: Optional[list[str]] = None,
        limit: int = 25,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (e:Entity)-[:HAS_KG_DOCUMENT]->(d:KgDocument)
            WHERE ($qids IS NULL OR e.wikidata_id IN $qids)
              AND ($year_from IS NULL OR coalesce(e.time_end_year, e.time_start_year, 999999) >= $year_from)
              AND ($year_to IS NULL OR coalesce(e.time_start_year, e.time_end_year, -999999) <= $year_to)
            RETURN DISTINCT e.wikidata_id AS wikidata_id,
                   e.name AS name,
                   e.wikipedia_title AS wikipedia_title,
                   e.wikipedia_url AS wikipedia_url,
                   e.time_start_year AS time_start_year,
                   e.time_end_year AS time_end_year,
                   d.path AS document_path
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

    def list_kg_chunks_for_entity(self, wikidata_id: str) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (:Entity {wikidata_id: $wikidata_id})-[:HAS_KG_DOCUMENT]->(d:KgDocument)
            MATCH (c:KgChunk)-[:KG_FROM_DOCUMENT]->(d)
            RETURN elementId(c) AS chunk_element_id,
                   coalesce(c.id, elementId(c)) AS chunk_id,
                   c.index AS chunk_index,
                   c.text AS chunk_text,
                   d.path AS document_path,
                   d.wikipedia_title AS wikipedia_title,
                   d.wikipedia_url AS wikipedia_url
            ORDER BY c.index
            """,
            {"wikidata_id": wikidata_id},
            database_=self.database,
        )
        return [record.data() for record in result.records]

    def delete_claims_for_entity(self, wikidata_id: str) -> None:
        self.driver.execute_query(
            """
            MATCH (:Entity {wikidata_id: $wikidata_id})-[:HAS_CLAIM]->(claim:Claim)
            DETACH DELETE claim
            """,
            {"wikidata_id": wikidata_id},
            database_=self.database,
        )

    def upsert_claims(
        self,
        claims: Iterable[ClaimRecord],
        progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        for claim in claims:
            self.driver.execute_query(
                """
                MERGE (c:Claim {claim_id: $claim_id})
                SET c.claim_text = $claim_text,
                    c.subject = $subject,
                    c.predicate = $predicate,
                    c.object_value = $object_value,
                    c.claim_type = $claim_type,
                    c.confidence = $confidence,
                    c.provenance_quote = $provenance_quote,
                    c.chunk_id = $chunk_id,
                    c.chunk_index = $chunk_index,
                    c.document_path = $document_path,
                    c.root_entity_qid = $root_entity_qid,
                    c.root_entity_name = $root_entity_name,
                    c.subject_entity_qid = $subject_entity_qid,
                    c.object_entity_qid = $object_entity_qid,
                    c.extracted_at = $extracted_at,
                    c.extraction_model = $extraction_model
                WITH c
                OPTIONAL MATCH (chunk:KgChunk) WHERE elementId(chunk) = $chunk_element_id
                FOREACH (_ IN CASE WHEN chunk IS NULL THEN [] ELSE [1] END |
                    MERGE (c)-[:CLAIM_FROM_CHUNK]->(chunk)
                )
                WITH c
                OPTIONAL MATCH (doc:KgDocument {path: $document_path})
                FOREACH (_ IN CASE WHEN doc IS NULL THEN [] ELSE [1] END |
                    MERGE (c)-[:CLAIM_FROM_DOCUMENT]->(doc)
                )
                WITH c
                OPTIONAL MATCH (root:Entity {wikidata_id: $root_entity_qid})
                FOREACH (_ IN CASE WHEN root IS NULL THEN [] ELSE [1] END |
                    MERGE (root)-[:HAS_CLAIM]->(c)
                )
                WITH c
                OPTIONAL MATCH (subject_entity:Entity {wikidata_id: $subject_entity_qid})
                FOREACH (_ IN CASE WHEN subject_entity IS NULL THEN [] ELSE [1] END |
                    MERGE (c)-[:CLAIM_SUBJECT]->(subject_entity)
                )
                WITH c
                OPTIONAL MATCH (object_entity:Entity {wikidata_id: $object_entity_qid})
                FOREACH (_ IN CASE WHEN object_entity IS NULL THEN [] ELSE [1] END |
                    MERGE (c)-[:CLAIM_OBJECT]->(object_entity)
                )
                """,
                {
                    "claim_id": claim.claim_id,
                    "claim_text": claim.claim_text,
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object_value": claim.object_value,
                    "claim_type": claim.claim_type,
                    "confidence": claim.confidence,
                    "provenance_quote": claim.provenance_quote,
                    "chunk_element_id": claim.chunk_element_id,
                    "chunk_id": claim.chunk_id,
                    "chunk_index": claim.chunk_index,
                    "document_path": claim.document_path,
                    "root_entity_qid": claim.root_entity_qid,
                    "root_entity_name": claim.root_entity_name,
                    "subject_entity_qid": claim.subject_entity_qid,
                    "object_entity_qid": claim.object_entity_qid,
                    "extracted_at": claim.extracted_at,
                    "extraction_model": claim.extraction_model,
                },
                database_=self.database,
            )
            if progress:
                progress(1)

    def list_claims_for_embedding(self) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (c:Claim)
            WHERE NOT 'embedding' IN keys(c) OR c.embedding IS NULL
            RETURN elementId(c) AS element_id,
                   c.claim_text AS text
            """,
            database_=self.database,
        )
        return [record.data() for record in result.records]

    def has_claims(self) -> bool:
        result = self.driver.execute_query(
            """
            MATCH (c:Claim)
            RETURN count(c) > 0 AS present
            """,
            database_=self.database,
        )
        return bool(result.records and result.records[0]["present"])

    def has_passages(self) -> bool:
        result = self.driver.execute_query(
            """
            MATCH (p:Passage)
            RETURN count(p) > 0 AS present
            """,
            database_=self.database,
        )
        return bool(result.records and result.records[0]["present"])

    def has_kg_chunks(self) -> bool:
        result = self.driver.execute_query(
            """
            MATCH (c:KgChunk)
            RETURN count(c) > 0 AS present
            """,
            database_=self.database,
        )
        return bool(result.records and result.records[0]["present"])

    def list_claim_entities(self, limit: int = 10) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (e:Entity)-[:HAS_CLAIM]->(c:Claim)
            RETURN e.wikidata_id AS wikidata_id,
                   e.name AS name,
                   count(c) AS claim_count
            ORDER BY claim_count DESC, name
            LIMIT $limit
            """,
            {"limit": limit},
            database_=self.database,
        )
        return [record.data() for record in result.records]

    def delete_communities_for_entity(self, wikidata_id: str) -> None:
        self.driver.execute_query(
            """
            MATCH (community:Community {focus_entity_qid: $wikidata_id})
            DETACH DELETE community
            """,
            {"wikidata_id": wikidata_id},
            database_=self.database,
        )
        self.driver.execute_query(
            """
            MATCH (report:CommunityReport {focus_entity_qid: $wikidata_id})
            DETACH DELETE report
            """,
            {"wikidata_id": wikidata_id},
            database_=self.database,
        )

    def upsert_community(
        self,
        *,
        community_id: str,
        title: str,
        focus_entity_qid: str,
        focus_entity_name: str,
        source_url: Optional[str],
        time_start_year: Optional[int],
        time_end_year: Optional[int],
    ) -> None:
        self.driver.execute_query(
            """
            MERGE (community:Community {community_id: $community_id})
            SET community.title = $title,
                community.focus_entity_qid = $focus_entity_qid,
                community.focus_entity_name = $focus_entity_name,
                community.source_url = $source_url,
                community.time_start_year = $time_start_year,
                community.time_end_year = $time_end_year
            WITH community
            MATCH (entity:Entity {wikidata_id: $focus_entity_qid})
            MERGE (entity)-[:HAS_COMMUNITY]->(community)
            """,
            {
                "community_id": community_id,
                "title": title,
                "focus_entity_qid": focus_entity_qid,
                "focus_entity_name": focus_entity_name,
                "source_url": source_url,
                "time_start_year": time_start_year,
                "time_end_year": time_end_year,
            },
            database_=self.database,
        )

    def replace_community_members(self, community_id: str, member_element_ids: list[str]) -> None:
        self.driver.execute_query(
            """
            MATCH (community:Community {community_id: $community_id})-[r:HAS_MEMBER]->()
            DELETE r
            """,
            {"community_id": community_id},
            database_=self.database,
        )
        for member_element_id in member_element_ids:
            self.driver.execute_query(
                """
                MATCH (community:Community {community_id: $community_id})
                MATCH (member) WHERE elementId(member) = $member_element_id
                MERGE (community)-[:HAS_MEMBER]->(member)
                """,
                {
                    "community_id": community_id,
                    "member_element_id": member_element_id,
                },
                database_=self.database,
            )

    def replace_community_claims(self, community_id: str, claim_ids: list[str]) -> None:
        self.driver.execute_query(
            """
            MATCH (community:Community {community_id: $community_id})-[r:HAS_CLAIM]->()
            DELETE r
            """,
            {"community_id": community_id},
            database_=self.database,
        )
        for claim_id in claim_ids:
            self.driver.execute_query(
                """
                MATCH (community:Community {community_id: $community_id})
                MATCH (claim:Claim {claim_id: $claim_id})
                MERGE (community)-[:HAS_CLAIM]->(claim)
                """,
                {
                    "community_id": community_id,
                    "claim_id": claim_id,
                },
                database_=self.database,
            )

    def upsert_community_report(self, report: CommunityReportRecord) -> None:
        self.driver.execute_query(
            """
            MERGE (report:CommunityReport {report_id: $report_id})
            SET report.community_id = $community_id,
                report.title = $title,
                report.summary = $summary,
                report.themes = $themes,
                report.themes_text = $themes_text,
                report.key_entities = $key_entities,
                report.key_entities_text = $key_entities_text,
                report.key_claims = $key_claims,
                report.question_suggestions = $question_suggestions,
                report.report_text = $report_text,
                report.focus_entity_qid = $focus_entity_qid,
                report.focus_entity_name = $focus_entity_name,
                report.source_url = $source_url,
                report.time_start_year = $time_start_year,
                report.time_end_year = $time_end_year,
                report.generated_at = $generated_at,
                report.generation_model = $generation_model
            WITH report
            MATCH (community:Community {community_id: $community_id})
            MERGE (community)-[:HAS_REPORT]->(report)
            """,
            {
                "report_id": report.report_id,
                "community_id": report.community_id,
                "title": report.title,
                "summary": report.summary,
                "themes": report.themes,
                "themes_text": ", ".join(report.themes),
                "key_entities": report.key_entities,
                "key_entities_text": ", ".join(report.key_entities),
                "key_claims": report.key_claims,
                "question_suggestions": report.question_suggestions,
                "report_text": report.report_text,
                "focus_entity_qid": report.focus_entity_qid,
                "focus_entity_name": report.focus_entity_name,
                "source_url": report.source_url,
                "time_start_year": report.time_start_year,
                "time_end_year": report.time_end_year,
                "generated_at": report.generated_at,
                "generation_model": report.generation_model,
            },
            database_=self.database,
        )

    def list_community_reports_for_embedding(self) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (report:CommunityReport)
            WHERE NOT 'embedding' IN keys(report) OR report.embedding IS NULL
            RETURN elementId(report) AS element_id,
                   report.report_text AS text
            """,
            database_=self.database,
        )
        return [record.data() for record in result.records]

    def has_community_reports(self) -> bool:
        result = self.driver.execute_query(
            """
            MATCH (report:CommunityReport)
            RETURN count(report) > 0 AS present
            """,
            database_=self.database,
        )
        return bool(result.records and result.records[0]["present"])

    def get_entity_community_source_pack(
        self,
        wikidata_id: str,
        *,
        member_limit: int,
        relation_limit: int,
        claim_limit: int,
    ) -> dict[str, Any]:
        entity_result = self.driver.execute_query(
            """
            MATCH (e:Entity {wikidata_id: $wikidata_id})-[:HAS_KG_DOCUMENT]->(d:KgDocument)
            RETURN elementId(e) AS element_id,
                   e.wikidata_id AS wikidata_id,
                   e.name AS name,
                   e.wikipedia_url AS wikipedia_url,
                   e.time_start_year AS time_start_year,
                   e.time_end_year AS time_end_year,
                   d.path AS document_path
            LIMIT 1
            """,
            {"wikidata_id": wikidata_id},
            database_=self.database,
        )
        node_result = self.driver.execute_query(
            """
            MATCH (:Entity {wikidata_id: $wikidata_id})-[:HAS_KG_DOCUMENT]->(d:KgDocument)
            MATCH (n)-[:KG_FROM_CHUNK]->(:KgChunk)-[:KG_FROM_DOCUMENT]->(d)
            WHERE NOT n:KgChunk AND NOT n:KgDocument
            RETURN elementId(n) AS element_id,
                   labels(n) AS labels,
                   n.name AS name,
                   count(*) AS mentions
            ORDER BY mentions DESC, name
            LIMIT $member_limit
            """,
            {"wikidata_id": wikidata_id, "member_limit": member_limit},
            database_=self.database,
        )
        relation_result = self.driver.execute_query(
            """
            MATCH (:Entity {wikidata_id: $wikidata_id})-[:HAS_KG_DOCUMENT]->(d:KgDocument)
            MATCH (a)-[:KG_FROM_CHUNK]->(:KgChunk)-[:KG_FROM_DOCUMENT]->(d)
            MATCH (a)-[r]-(b)
            WHERE type(r) <> 'KG_FROM_CHUNK'
              AND NOT b:KgChunk
              AND NOT b:KgDocument
            RETURN coalesce(a.name, elementId(a)) AS source,
                   type(r) AS relation,
                   coalesce(b.name, elementId(b)) AS target,
                   count(*) AS mentions
            ORDER BY mentions DESC, source, relation, target
            LIMIT $relation_limit
            """,
            {"wikidata_id": wikidata_id, "relation_limit": relation_limit},
            database_=self.database,
        )
        resolved_result = self.driver.execute_query(
            """
            MATCH (:Entity {wikidata_id: $wikidata_id})-[:HAS_KG_DOCUMENT]->(d:KgDocument)
            MATCH (n)-[:KG_FROM_CHUNK]->(:KgChunk)-[:KG_FROM_DOCUMENT]->(d)
            MATCH (n)-[r:RESOLVES_TO]->(entity:Entity)
            RETURN elementId(entity) AS element_id,
                   entity.wikidata_id AS wikidata_id,
                   entity.name AS name,
                   count(r) AS mentions
            ORDER BY mentions DESC, name
            LIMIT $member_limit
            """,
            {"wikidata_id": wikidata_id, "member_limit": member_limit},
            database_=self.database,
        )
        claim_result = self.driver.execute_query(
            """
            MATCH (:Entity {wikidata_id: $wikidata_id})-[:HAS_CLAIM]->(claim:Claim)
            RETURN claim.claim_id AS claim_id,
                   claim.claim_text AS claim_text,
                   claim.confidence AS confidence
            ORDER BY confidence DESC, claim.claim_text
            LIMIT $claim_limit
            """,
            {"wikidata_id": wikidata_id, "claim_limit": claim_limit},
            database_=self.database,
        )
        return {
            "entity": entity_result.records[0].data() if entity_result.records else None,
            "nodes": [record.data() for record in node_result.records],
            "relations": [record.data() for record in relation_result.records],
            "resolved_entities": [record.data() for record in resolved_result.records],
            "claims": [record.data() for record in claim_result.records],
        }

    def list_entities_for_resolution(self) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (e:Entity)
            RETURN e.wikidata_id AS wikidata_id,
                   e.name AS name,
                   coalesce(e.aliases, []) AS aliases,
                   e.wikipedia_title AS wikipedia_title,
                   e.wikipedia_url AS wikipedia_url,
                   e.entity_kind AS entity_kind,
                   e.time_start_year AS time_start_year,
                   e.time_end_year AS time_end_year
            """,
            database_=self.database,
        )
        return [record.data() for record in result.records]

    def list_extracted_nodes_for_resolution(self) -> list[dict]:
        result = self.driver.execute_query(
            """
            MATCH (n)-[:KG_FROM_CHUNK]->(:KgChunk)
            OPTIONAL MATCH (n)-[:KG_FROM_CHUNK]->(:KgChunk)-[:KG_FROM_DOCUMENT]->(doc:KgDocument)
            OPTIONAL MATCH (root:Entity)-[:HAS_KG_DOCUMENT]->(doc)
            WHERE NOT n:Entity
              AND NOT n:Passage
              AND NOT n:KgDocument
              AND NOT n:KgChunk
              AND n.name IS NOT NULL
            RETURN elementId(n) AS element_id,
                   labels(n) AS labels,
                   n.name AS name,
                   doc.wikipedia_title AS document_title,
                   doc.wikipedia_url AS document_url,
                   root.wikidata_id AS source_entity_qid,
                   root.name AS source_entity_name,
                   root.entity_kind AS source_entity_kind,
                   root.time_start_year AS source_time_start_year,
                   root.time_end_year AS source_time_end_year
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
        alias_score: Optional[float] = None,
        embedding_score: Optional[float] = None,
        type_score: Optional[float] = None,
        context_score: Optional[float] = None,
    ) -> None:
        self.driver.execute_query(
            """
            MATCH (n) WHERE elementId(n) = $node_element_id
            MATCH (e:Entity {wikidata_id: $wikidata_id})
            MERGE (n)-[r:RESOLVES_TO]->(e)
            SET r.method = $method,
                r.score = $score,
                r.matched_text = $matched_text,
                r.alias_score = $alias_score,
                r.embedding_score = $embedding_score,
                r.type_score = $type_score,
                r.context_score = $context_score
            """,
            {
                "node_element_id": node_element_id,
                "wikidata_id": wikidata_id,
                "method": method,
                "score": score,
                "matched_text": matched_text,
                "alias_score": alias_score,
                "embedding_score": embedding_score,
                "type_score": type_score,
                "context_score": context_score,
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
