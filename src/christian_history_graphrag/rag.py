from __future__ import annotations

import json
from typing import Callable, Optional

from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.retrievers import HybridCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.types import EntityType, RetrieverResultItem

from christian_history_graphrag.config import Settings
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.providers import build_embedder, build_llm


RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)<-[:HAS_PASSAGE]-(entity:Entity)
OPTIONAL MATCH (node)-[:DERIVED_FROM]->(source_doc:SourceDocument)
OPTIONAL MATCH (entity)-[r]-(neighbor:Entity)
WHERE ($year_from IS NULL OR coalesce(entity.time_end_year, entity.time_start_year, 999999) >= $year_from)
  AND ($year_to IS NULL OR coalesce(entity.time_start_year, entity.time_end_year, -999999) <= $year_to)
RETURN
  coalesce(node.id, elementId(node)) AS passage_id,
  node.text AS passage,
  node.title AS page_title,
  node.chunk_index AS chunk_index,
  entity.name AS entity_name,
  entity.wikidata_id AS entity_qid,
  entity.entity_kind AS entity_kind,
  entity.time_start_year AS start_year,
  entity.time_end_year AS end_year,
  entity.wikipedia_url AS wikipedia_url,
  source_doc.source_url AS provenance_url,
  collect(DISTINCT {
    relation: type(r),
    target: neighbor.name,
    target_qid: neighbor.wikidata_id
  })[0..12] AS graph_neighbors,
  score
"""

HYBRID_RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[:KG_FROM_DOCUMENT]->(doc:KgDocument)
OPTIONAL MATCH (entity:Entity)-[:HAS_KG_DOCUMENT]->(doc)
OPTIONAL MATCH (entity)-[:HAS_SOURCE]->(entity_source:SourceDocument {source_system: 'wikipedia'})
OPTIONAL MATCH (kg_node)-[:KG_FROM_CHUNK]->(node)
OPTIONAL MATCH (kg_node)-[r]-(neighbor)
WHERE ($year_from IS NULL OR coalesce(entity.time_end_year, entity.time_start_year, 999999) >= $year_from)
  AND ($year_to IS NULL OR coalesce(entity.time_start_year, entity.time_end_year, -999999) <= $year_to)
  AND (r IS NULL OR type(r) <> 'KG_FROM_CHUNK')
  AND (neighbor IS NULL OR (NOT neighbor:KgChunk AND NOT neighbor:KgDocument))
RETURN
  coalesce(node.id, elementId(node)) AS passage_id,
  node.text AS passage,
  doc.wikipedia_title AS page_title,
  node.index AS chunk_index,
  entity.name AS entity_name,
  entity.wikidata_id AS entity_qid,
  entity.entity_kind AS entity_kind,
  entity.time_start_year AS start_year,
  entity.time_end_year AS end_year,
  doc.wikipedia_url AS wikipedia_url,
  entity_source.source_url AS provenance_url,
  collect(DISTINCT {
    node: kg_node.name,
    relation: type(r),
    target: neighbor.name
  })[0..20] AS graph_neighbors,
  score
"""

TEXT2CYPHER_EXAMPLES = [
    """Question: Which entities influenced Augustine of Hippo?
Cypher:
MATCH (e:Entity {name: "Augustine of Hippo"})-[:INFLUENCED_BY]->(influencer:Entity)
RETURN influencer.name AS influencer, influencer.wikidata_id AS wikidata_id
ORDER BY influencer.name""",
    """Question: Which councils took place in the fourth century?
Cypher:
MATCH (e:Entity)
WHERE e.entity_kind = "Event"
  AND coalesce(e.time_end_year, e.time_start_year, 999999) >= 300
  AND coalesce(e.time_start_year, e.time_end_year, -999999) <= 399
RETURN e.name AS event, e.wikidata_id AS wikidata_id, e.time_start_year AS start_year, e.time_end_year AS end_year
ORDER BY e.time_start_year, e.name""",
    """Question: Show source-backed statements about the First Council of Nicaea.
Cypher:
MATCH (e:Entity {name: "First Council of Nicaea"})-[:HAS_STATEMENT]->(s:Statement)
OPTIONAL MATCH (s)-[:TARGETS]->(target:Entity)
OPTIONAL MATCH (s)-[:SUPPORTED_BY]->(source:SourceDocument)
RETURN s.relation_type AS relation, target.name AS target, s.rank AS rank, s.reference_count AS reference_count, source.source_url AS source_url
ORDER BY s.reference_count DESC, relation""",
]


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
            if target_qid:
                neighbor_lines.append(f"- {relation}: {target} ({target_qid})")
            else:
                neighbor_lines.append(f"- {relation}: {target}")

    content_parts = [
        f"Entity: {data.get('entity_name')} ({data.get('entity_qid')})",
        f"Type: {data.get('entity_kind')}",
        f"Period: {data.get('start_year')} -> {data.get('end_year')}",
        f"Page: {data.get('page_title')}",
        f"Chunk: {data.get('chunk_index')}",
        f"Passage ID: {data.get('passage_id')}",
        f"Source: {data.get('wikipedia_url')}",
        f"Provenance: {data.get('provenance_url')}",
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


def format_text2cypher_record(record) -> RetrieverResultItem:
    data = record.data()
    content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    return RetrieverResultItem(
        content=content,
        metadata=data,
    )


def build_hybrid_rag_response(
    store: Neo4jStore,
    settings: Settings,
    question: str,
    top_k: int,
    year_from: Optional[int],
    year_to: Optional[int],
    return_context: bool,
    vector_index_name: str,
    fulltext_index_name: str,
    retrieval_query: str,
) -> RagResultModel:
    # HybridCypherRetriever forwards query params to Neo4j exactly as provided.
    # The retrieval queries reference year_from/year_to even when they are NULL,
    # so both parameters must always be present.
    query_params = {
        "year_from": year_from,
        "year_to": year_to,
    }

    retriever = HybridCypherRetriever(
        store.driver,
        vector_index_name=vector_index_name,
        fulltext_index_name=fulltext_index_name,
        retrieval_query=retrieval_query,
        embedder=build_embedder(settings),
        result_formatter=format_retrieval_record,
        neo4j_database=store.database,
    )
    rag = GraphRAG(retriever=retriever, llm=build_llm(settings))
    return rag.search(
        query_text=question,
        retriever_config={
            "top_k": top_k,
            "query_params": query_params,
        },
        return_context=return_context,
    )


def build_text2cypher_rag_response(
    store: Neo4jStore,
    settings: Settings,
    question: str,
    return_context: bool,
) -> RagResultModel:
    retriever = Text2CypherRetriever(
        store.driver,
        llm=build_llm(settings),
        neo4j_schema=store.get_graph_schema(),
        examples=TEXT2CYPHER_EXAMPLES,
        result_formatter=format_text2cypher_record,
        neo4j_database=store.database,
    )
    rag = GraphRAG(retriever=retriever, llm=build_llm(settings))
    return rag.search(
        query_text=question,
        return_context=return_context,
    )


def ask_llm_only(settings: Settings, question: str) -> str:
    llm = build_llm(settings)
    response = llm.invoke(
        input=(
            "Answer the question using only your own model knowledge. "
            "Do not assume any retrieved graph context or external retrieved documents.\n\n"
            f"Question: {question}"
        )
    )
    return response.content


def embed_passages(
    store: Neo4jStore,
    settings: Settings,
    rebuild: bool = False,
    progress: Optional[Callable[[int], None]] = None,
) -> None:
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
        embeddings = []
        batch_size = max(settings.embedding_batch_size, 1)
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            embeddings.extend(
                embedder.embed_documents(
                    batch_texts,
                    batch_size=batch_size,
                )
            )
            if progress:
                progress(len(batch_texts))
    else:
        embeddings = []
        for text in texts:
            embeddings.append(embedder.embed_query(text))
            if progress:
                progress(1)

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
    return build_hybrid_rag_response(
        store=store,
        settings=settings,
        question=question,
        top_k=top_k,
        year_from=year_from,
        year_to=year_to,
        return_context=return_context,
        vector_index_name="passage_embeddings",
        fulltext_index_name="passage_fulltext",
        retrieval_query=RETRIEVAL_QUERY,
    )


def ask_hybrid_question(
    store: Neo4jStore,
    settings: Settings,
    question: str,
    top_k: int = 5,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    return_context: bool = False,
) -> RagResultModel:
    return build_hybrid_rag_response(
        store=store,
        settings=settings,
        question=question,
        top_k=top_k,
        year_from=year_from,
        year_to=year_to,
        return_context=return_context,
        vector_index_name="kg_chunk_embeddings",
        fulltext_index_name="kg_chunk_fulltext",
        retrieval_query=HYBRID_RETRIEVAL_QUERY,
    )


def ask_cypher_question(
    store: Neo4jStore,
    settings: Settings,
    question: str,
    return_context: bool = False,
) -> RagResultModel:
    return build_text2cypher_rag_response(
        store=store,
        settings=settings,
        question=question,
        return_context=return_context,
    )
