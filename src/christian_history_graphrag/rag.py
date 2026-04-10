from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.retrievers import HybridCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.types import EntityType, RetrieverResultItem

from christian_history_graphrag.config import Settings
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.providers import build_embedder, build_llm, build_reranker


logger = logging.getLogger(__name__)


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
  node.section_title AS section_title,
  coalesce(node.section_path, []) AS section_path,
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
  NULL AS section_title,
  [] AS section_path,
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

CLAIM_RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[:CLAIM_FROM_CHUNK]->(chunk:KgChunk)
OPTIONAL MATCH (node)-[:CLAIM_FROM_DOCUMENT]->(doc:KgDocument)
OPTIONAL MATCH (entity:Entity)-[:HAS_CLAIM]->(node)
OPTIONAL MATCH (node)-[:CLAIM_SUBJECT]->(subject_entity:Entity)
OPTIONAL MATCH (node)-[:CLAIM_OBJECT]->(object_entity:Entity)
WHERE ($year_from IS NULL OR coalesce(entity.time_end_year, entity.time_start_year, 999999) >= $year_from)
  AND ($year_to IS NULL OR coalesce(entity.time_start_year, entity.time_end_year, -999999) <= $year_to)
RETURN
  node.claim_id AS passage_id,
  node.claim_text AS passage,
  doc.wikipedia_title AS page_title,
  NULL AS section_title,
  [] AS section_path,
  node.chunk_index AS chunk_index,
  entity.name AS entity_name,
  entity.wikidata_id AS entity_qid,
  node.claim_type AS entity_kind,
  entity.time_start_year AS start_year,
  entity.time_end_year AS end_year,
  doc.wikipedia_url AS wikipedia_url,
  doc.wikipedia_url AS provenance_url,
  [item IN [
    CASE
      WHEN node.subject IS NULL THEN NULL
      ELSE {relation: 'SUBJECT', target: node.subject, target_qid: subject_entity.wikidata_id}
    END,
    CASE
      WHEN node.object_value IS NULL THEN NULL
      ELSE {relation: coalesce(node.predicate, 'OBJECT'), target: node.object_value, target_qid: object_entity.wikidata_id}
    END
  ] WHERE item IS NOT NULL] AS graph_neighbors,
  node.confidence AS claim_confidence,
  node.provenance_quote AS provenance_quote,
  score
"""

COMMUNITY_RETRIEVAL_QUERY = """
OPTIONAL MATCH (community:Community)-[:HAS_REPORT]->(node)
OPTIONAL MATCH (entity:Entity)-[:HAS_COMMUNITY]->(community)
OPTIONAL MATCH (community)-[:HAS_MEMBER]->(member)
WHERE ($year_from IS NULL OR coalesce(node.time_end_year, node.time_start_year, 999999) >= $year_from)
  AND ($year_to IS NULL OR coalesce(node.time_start_year, node.time_end_year, -999999) <= $year_to)
RETURN
  node.report_id AS passage_id,
  node.report_text AS passage,
  node.title AS page_title,
  NULL AS section_title,
  [] AS section_path,
  0 AS chunk_index,
  coalesce(entity.name, node.focus_entity_name) AS entity_name,
  coalesce(entity.wikidata_id, node.focus_entity_qid) AS entity_qid,
  'CommunityReport' AS entity_kind,
  node.time_start_year AS start_year,
  node.time_end_year AS end_year,
  node.source_url AS wikipedia_url,
  node.source_url AS provenance_url,
  collect(DISTINCT {
    relation: 'HAS_MEMBER',
    target: coalesce(member.name, member.title, member.claim_text),
    target_qid: member.wikidata_id
  })[0..12] AS graph_neighbors,
  node.summary AS report_summary,
  node.themes AS report_themes,
  node.key_entities AS key_entities,
  node.key_claims AS key_claims,
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


def _format_score(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _build_reranker_text(item: RetrieverResultItem) -> str:
    metadata = dict(item.metadata or {})
    passage_text = metadata.get("passage") or metadata.get("report_summary") or item.content
    section_path = metadata.get("section_path") or []
    section_text = " > ".join(str(part) for part in section_path if part)

    parts = [
        f"Entity: {metadata.get('entity_name')}",
        f"Type: {metadata.get('entity_kind')}",
        f"Page: {metadata.get('page_title')}",
        f"Section: {metadata.get('section_title')}",
        f"Section path: {section_text or None}",
        f"Passage: {passage_text}",
    ]
    return "\n".join(str(part) for part in parts if part is not None)


def _replace_retriever_result(result, *, items, metadata):
    if hasattr(result, "model_copy"):
        return result.model_copy(update={"items": items, "metadata": metadata})
    if hasattr(result, "copy"):
        try:
            return result.copy(update={"items": items, "metadata": metadata})
        except TypeError:
            pass

    try:
        result.items = items
    except Exception:
        pass
    try:
        result.metadata = metadata
    except Exception:
        pass
    return result


def _replace_result_item_metadata(item: RetrieverResultItem, metadata: dict[str, Any]) -> RetrieverResultItem:
    updated_content = item.content
    rerank_score = metadata.get("rerank_score")
    if rerank_score is not None and "Rerank score:" not in updated_content:
        updated_content = f"{updated_content}\nRerank score: {_format_score(rerank_score)}"

    if hasattr(item, "model_copy"):
        return item.model_copy(update={"metadata": metadata, "content": updated_content})
    if hasattr(item, "copy"):
        try:
            return item.copy(update={"metadata": metadata, "content": updated_content})
        except TypeError:
            pass

    try:
        item.metadata = metadata
    except Exception:
        pass
    try:
        item.content = updated_content
    except Exception:
        pass
    return item


class RerankingRetriever:
    def __init__(self, inner_retriever, reranker, *, candidate_pool_size: int) -> None:
        self.inner_retriever = inner_retriever
        self.reranker = reranker
        self.candidate_pool_size = max(candidate_pool_size, 1)

    def __getattr__(self, name):
        return getattr(self.inner_retriever, name)

    def search(self, *args, **kwargs):
        query_text = kwargs.get("query_text")
        if query_text is None and args:
            query_text = args[0]

        requested_top_k = kwargs.get("top_k", 5)
        try:
            requested_top_k = max(int(requested_top_k), 1)
        except (TypeError, ValueError):
            requested_top_k = 5

        inner_kwargs = dict(kwargs)
        inner_kwargs["top_k"] = max(requested_top_k, self.candidate_pool_size)
        result = self.inner_retriever.search(*args, **inner_kwargs)
        items = list(getattr(result, "items", []) or [])
        if not items or not query_text:
            return result

        try:
            reranker_texts = [_build_reranker_text(item) for item in items]
            rerank_scores = self.reranker.score(query_text, reranker_texts)
            reranked_pairs = sorted(
                zip(rerank_scores, items),
                key=lambda pair: pair[0],
                reverse=True,
            )
        except Exception as exc:
            logger.warning("Reranking failed; falling back to retrieval order: %s", exc)
            return result

        reranked_items = []
        for score, item in reranked_pairs[:requested_top_k]:
            metadata = dict(item.metadata or {})
            metadata["rerank_score"] = float(score)
            reranked_items.append(_replace_result_item_metadata(item, metadata))

        result_metadata = dict(getattr(result, "metadata", {}) or {})
        result_metadata["reranked"] = True
        result_metadata["reranker_model"] = getattr(self.reranker, "model_name", None)
        result_metadata["reranker_candidate_pool_size"] = len(items)
        return _replace_retriever_result(
            result,
            items=reranked_items,
            metadata=result_metadata,
        )


def format_retrieval_record(record) -> RetrieverResultItem:
    data = record.data()
    neighbors = data.get("graph_neighbors") or []
    section_path = data.get("section_path") or []
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
        f"Section: {data.get('section_title')}",
        "Section path: " + " > ".join(str(item) for item in section_path) if section_path else None,
        f"Chunk: {data.get('chunk_index')}",
        f"Passage ID: {data.get('passage_id')}",
        f"Source: {data.get('wikipedia_url')}",
        f"Provenance: {data.get('provenance_url')}",
        f"Score: {_format_score(data.get('score'))}",
        f"Rerank score: {_format_score(data.get('rerank_score'))}" if data.get("rerank_score") is not None else None,
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


def format_claim_record(record) -> RetrieverResultItem:
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
        f"Claim type: {data.get('entity_kind')}",
        f"Page: {data.get('page_title')}",
        f"Chunk: {data.get('chunk_index')}",
        f"Claim ID: {data.get('passage_id')}",
        f"Source: {data.get('wikipedia_url')}",
        f"Provenance: {data.get('provenance_url')}",
        f"Claim confidence: {data.get('claim_confidence')}",
        f"Score: {_format_score(data.get('score'))}",
        f"Rerank score: {_format_score(data.get('rerank_score'))}" if data.get("rerank_score") is not None else None,
        "Claim:",
        data.get("passage") or "",
        f"Evidence quote: {data.get('provenance_quote')}",
    ]
    if neighbor_lines:
        content_parts.append("Claim structure:")
        content_parts.extend(neighbor_lines)
    return RetrieverResultItem(
        content="\n".join(str(part) for part in content_parts if part is not None),
        metadata=data,
    )


def format_community_report_record(record) -> RetrieverResultItem:
    data = record.data()
    themes = data.get("report_themes") or []
    key_entities = data.get("key_entities") or []
    key_claims = data.get("key_claims") or []
    neighbors = data.get("graph_neighbors") or []
    neighbor_lines = []
    for neighbor in neighbors:
        target = neighbor.get("target")
        relation = neighbor.get("relation")
        target_qid = neighbor.get("target_qid")
        if relation and target:
            if target_qid:
                neighbor_lines.append(f"- {relation}: {target} ({target_qid})")
            else:
                neighbor_lines.append(f"- {relation}: {target}")

    content_parts = [
        f"Community report: {data.get('page_title')}",
        f"Focus entity: {data.get('entity_name')} ({data.get('entity_qid')})",
        f"Period: {data.get('start_year')} -> {data.get('end_year')}",
        f"Report ID: {data.get('passage_id')}",
        f"Source: {data.get('wikipedia_url')}",
        f"Score: {_format_score(data.get('score'))}",
        f"Rerank score: {_format_score(data.get('rerank_score'))}" if data.get("rerank_score") is not None else None,
        "Summary:",
        data.get("report_summary") or data.get("passage") or "",
    ]
    if themes:
        content_parts.append("Themes: " + ", ".join(str(item) for item in themes))
    if key_entities:
        content_parts.append("Key entities: " + ", ".join(str(item) for item in key_entities))
    if key_claims:
        content_parts.append("Key claims: " + " | ".join(str(item) for item in key_claims))
    if neighbor_lines:
        content_parts.append("Community members:")
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
    result_formatter=format_retrieval_record,
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
        result_formatter=result_formatter,
        neo4j_database=store.database,
    )
    try:
        reranker = build_reranker(settings)
    except Exception as exc:
        logger.warning("Reranker unavailable; using base retrieval order: %s", exc)
        reranker = None
    if reranker is not None:
        retriever = RerankingRetriever(
            retriever,
            reranker,
            candidate_pool_size=settings.reranker_candidate_pool_size,
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
        result_formatter=format_retrieval_record,
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
        result_formatter=format_retrieval_record,
    )


def ask_claims_question(
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
        vector_index_name="claim_embeddings",
        fulltext_index_name="claim_fulltext",
        retrieval_query=CLAIM_RETRIEVAL_QUERY,
        result_formatter=format_claim_record,
    )


def ask_global_question(
    store: Neo4jStore,
    settings: Settings,
    question: str,
    top_k: int = 4,
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
        vector_index_name="community_report_embeddings",
        fulltext_index_name="community_report_fulltext",
        retrieval_query=COMMUNITY_RETRIEVAL_QUERY,
        result_formatter=format_community_report_record,
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
