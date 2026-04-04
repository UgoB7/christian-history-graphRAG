from __future__ import annotations

from typing import Iterable, Optional

from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

from christian_history_graphrag.models import EntityRecord


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def setup(self) -> None:
        statements = [
            "CREATE CONSTRAINT entity_qid IF NOT EXISTS FOR (n:Entity) REQUIRE n.wikidata_id IS UNIQUE",
            "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (n:Passage) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX entity_time_start IF NOT EXISTS FOR (n:Entity) ON (n.time_start_year)",
            "CREATE INDEX entity_time_end IF NOT EXISTS FOR (n:Entity) ON (n.time_end_year)",
            "CREATE INDEX entity_kind IF NOT EXISTS FOR (n:Entity) ON (n.entity_kind)",
        ]
        for statement in statements:
            self.driver.execute_query(statement, database_=self.database)

    def ensure_kg_indexes(self) -> None:
        statements = [
            "CREATE CONSTRAINT kg_document_path IF NOT EXISTS FOR (n:KgDocument) REQUIRE n.path IS UNIQUE",
            "CREATE INDEX kg_document_wikidata IF NOT EXISTS FOR (n:KgDocument) ON (n.wikidata_id)",
            "CREATE INDEX kg_chunk_index IF NOT EXISTS FOR (n:KgChunk) ON (n.index)",
        ]
        for statement in statements:
            self.driver.execute_query(statement, database_=self.database)

    def upsert_entities(self, records: Iterable[EntityRecord]) -> None:
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
                    e.wikipedia_url = $wikipedia_url
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
                },
                database_=self.database,
            )

    def upsert_relations(self, records: Iterable[EntityRecord]) -> None:
        for record in records:
            for relation in record.relations:
                self.driver.execute_query(
                    f"""
                    MATCH (source:Entity {{wikidata_id: $source_qid}})
                    MATCH (target:Entity {{wikidata_id: $target_qid}})
                    MERGE (source)-[r:{relation.relation_type}]->(target)
                    SET r.wikidata_property = $property_id
                    """,
                    {
                        "source_qid": record.qid,
                        "target_qid": relation.target_qid,
                        "property_id": relation.property_id,
                    },
                    database_=self.database,
                )

    def upsert_passages(self, records: Iterable[EntityRecord]) -> None:
        for record in records:
            if not record.passages:
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
                    MERGE (p:Passage {id: $passage_id})
                    SET p.title = $title,
                        p.url = $url,
                        p.language = $language,
                        p.chunk_index = $chunk_index,
                        p.text = $text,
                        p.source = 'wikipedia',
                        p.time_start_year = $time_start_year,
                        p.time_end_year = $time_end_year
                    MERGE (e)-[:HAS_PASSAGE]->(p)
                    """,
                    {
                        "wikidata_id": record.qid,
                        "passage_id": f"wikipedia:{record.qid}:{passage.chunk_index}",
                        "title": passage.page_title,
                        "url": passage.url,
                        "language": passage.language,
                        "chunk_index": passage.chunk_index,
                        "text": passage.text,
                        "time_start_year": record.start_year,
                        "time_end_year": record.end_year,
                    },
                    database_=self.database,
                )

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

    def reset_database(self) -> None:
        statements = [
            "DROP INDEX passage_embeddings IF EXISTS",
            "DROP INDEX kg_chunk_embeddings IF EXISTS",
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
