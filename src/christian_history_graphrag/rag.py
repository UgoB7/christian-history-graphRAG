from __future__ import annotations

from typing import Optional

from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import EntityType, RetrieverResultItem

from christian_history_graphrag.config import Settings
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.providers import build_embedder, build_llm


RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)<-[:HAS_PASSAGE]-(entity:Entity)
OPTIONAL MATCH (entity)-[r]-(neighbor:Entity)
WHERE ($year_from IS NULL OR coalesce(entity.time_end_year, entity.time_start_year, 999999) >= $year_from)
  AND ($year_to IS NULL OR coalesce(entity.time_start_year, entity.time_end_year, -999999) <= $year_to)
RETURN
  node.id AS passage_id,
  node.text AS passage,
  node.title AS page_title,
  node.chunk_index AS chunk_index,
  entity.name AS entity_name,
  entity.wikidata_id AS entity_qid,
  entity.entity_kind AS entity_kind,
  entity.time_start_year AS start_year,
  entity.time_end_year AS end_year,
  entity.wikipedia_url AS wikipedia_url,
  collect(DISTINCT {
    relation: type(r),
    target: neighbor.name,
    target_qid: neighbor.wikidata_id
  })[0..12] AS graph_neighbors,
  score
"""


def format_retrieval_record(record) -> RetrieverResultItem:
    data = record.data()
    neighbors = data.get("graph_neighbors") or []
    neighbor_lines = []
    for neighbor in neighbors:
        if not neighbor:
            continue
        relation = neighbor.get("relation")
        target = neighbor.get("target")
        target_qid = neighbor.get("target_qid")
        if relation and target:
            neighbor_lines.append(f"- {relation}: {target} ({target_qid})")

    content_parts = [
        f"Entity: {data.get('entity_name')} ({data.get('entity_qid')})",
        f"Type: {data.get('entity_kind')}",
        f"Period: {data.get('start_year')} -> {data.get('end_year')}",
        f"Page: {data.get('page_title')}",
        f"Chunk: {data.get('chunk_index')}",
        f"Passage ID: {data.get('passage_id')}",
        f"Source: {data.get('wikipedia_url')}",
        f"Score: {data.get('score')}",
        "Passage:",
        data.get("passage") or "",
    ]
    if neighbor_lines:
        content_parts.append("Graph neighbors:")
        content_parts.extend(neighbor_lines)

    return RetrieverResultItem(
        content="\n".join(str(part) for part in content_parts if part is not None),
        metadata=data,
    )
def embed_passages(store: Neo4jStore, settings: Settings, rebuild: bool = False) -> None:
    embedder = build_embedder(settings)
    if rebuild:
        store.drop_vector_index()
        store.clear_embeddings()
    result = store.driver.execute_query(
        """
        MATCH (p:Passage)
        WHERE NOT 'embedding' IN keys(p) OR p.embedding IS NULL
        RETURN elementId(p) AS element_id, p.text AS text
        """,
        database_=store.database,
    )
    rows = result.records
    if not rows:
        return

    texts = [record["text"] for record in rows]
    ids = [record["element_id"] for record in rows]
    if hasattr(embedder, "embed_documents"):
        embeddings = embedder.embed_documents(
            texts,
            batch_size=settings.embedding_batch_size,
        )
    else:
        embeddings = [embedder.embed_query(text) for text in texts]

    store.create_vector_index(dimensions=len(embeddings[0]))
    upsert_vectors(
        store.driver,
        ids=ids,
        embedding_property="embedding",
        embeddings=embeddings,
        entity_type=EntityType.NODE,
        neo4j_database=store.database,
    )


def ask_question(
    store: Neo4jStore,
    settings: Settings,
    question: str,
    top_k: int = 5,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    return_context: bool = False,
) -> RagResultModel:
    query_params = {}
    if year_from is not None:
        query_params["year_from"] = year_from
    if year_to is not None:
        query_params["year_to"] = year_to

    retriever = VectorCypherRetriever(
        store.driver,
        index_name="passage_embeddings",
        retrieval_query=RETRIEVAL_QUERY,
        embedder=build_embedder(settings),
        result_formatter=format_retrieval_record,
        neo4j_database=store.database,
    )
    rag = GraphRAG(retriever=retriever, llm=build_llm(settings))
    response = rag.search(
        query_text=question,
        retriever_config={
            "top_k": top_k,
            "query_params": query_params or None,
        },
        return_context=return_context,
    )
    return response
