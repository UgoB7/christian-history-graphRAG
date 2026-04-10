from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Callable, Optional

from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.types import EntityType

from christian_history_graphrag.config import Settings
from christian_history_graphrag.entity_resolution import (
    EntityResolutionResources,
    build_entity_resolution_resources,
    match_entity_name,
)
from christian_history_graphrag.llm_json import extract_json_payload
from christian_history_graphrag.models import ClaimRecord
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.providers import build_embedder, build_llm


logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _build_claim_prompt(
    *,
    entity_name: str,
    wikipedia_title: str,
    chunk_index: int,
    chunk_text: str,
    max_claims: int,
) -> str:
    return f"""
You extract atomic claims from Christian history source text.
Return only valid JSON with this exact shape:
{{
  "claims": [
    {{
      "claim_text": "short standalone sentence",
      "subject": "main subject",
      "predicate": "SHORT_RELATION_LABEL",
      "object": "main object",
      "claim_type": "factual|traditional|textual|chronology|location|attribution",
      "confidence": 0.0,
      "provenance_quote": "short evidence quote"
    }}
  ]
}}

Rules:
- Extract at most {max_claims} claims.
- Only keep claims explicitly supported by the text.
- Prefer atomic, sourceable claims over broad summaries.
- Use a confidence between 0.0 and 1.0.
- Keep provenance_quote under 18 words.
- If there are no useful claims, return {{"claims": []}}.

Document focus: {entity_name}
Wikipedia page: {wikipedia_title}
Chunk index: {chunk_index}

Text:
\"\"\"
{chunk_text}
\"\"\"
""".strip()


def _normalize_claim_rows(payload: object, max_claims: int) -> list[dict]:
    if isinstance(payload, dict):
        rows = payload.get("claims", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    normalized: list[dict] = []
    seen_claims: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        claim_text = str(row.get("claim_text", "")).strip()
        if not claim_text or claim_text.casefold() in seen_claims:
            continue
        seen_claims.add(claim_text.casefold())
        normalized.append(row)
        if len(normalized) >= max_claims:
            break
    return normalized


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(confidence, 1.0))


def _build_claim_id(root_entity_qid: str, chunk_id: str, claim_text: str) -> str:
    digest = hashlib.sha1(
        f"{root_entity_qid}|{chunk_id}|{claim_text}".encode("utf-8")
    ).hexdigest()
    return f"claim:{root_entity_qid}:{digest[:16]}"


def _resolve_claim_entity_name(
    value: Optional[str],
    *,
    resources: EntityResolutionResources,
    threshold: float,
    candidate_limit: int,
) -> Optional[str]:
    if not value:
        return None
    match = match_entity_name(
        value,
        resources=resources,
        threshold=threshold,
        candidate_limit=candidate_limit,
    )
    if not match:
        return None
    candidate = match["candidate"]
    return candidate.get("wikidata_id")


def extract_claims(
    store: Neo4jStore,
    settings: Settings,
    *,
    qids: Optional[list[str]] = None,
    limit: int = 25,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    replace_existing: bool = True,
    progress: Optional[Callable[[int], None]] = None,
    reporter: Optional[Callable[[str], None]] = None,
) -> dict[str, int]:
    entities = store.list_entities_with_kg_documents(
        qids=qids,
        limit=limit,
        year_from=year_from,
        year_to=year_to,
    )
    if not entities:
        return {"entities": 0, "claims": 0}

    store.ensure_claim_indexes()
    llm = build_llm(settings, model_name=settings.claim_extraction_llm_model)
    entity_index = store.list_entities_for_resolution()
    resolution_resources = build_entity_resolution_resources(
        entity_index,
        settings=settings,
        reporter=reporter,
    )

    processed_entities = 0
    created_claims = 0
    for entity in entities:
        if replace_existing:
            store.delete_claims_for_entity(entity["wikidata_id"])

        chunks = store.list_kg_chunks_for_entity(entity["wikidata_id"])
        entity_claims: list[ClaimRecord] = []
        if reporter:
            reporter(
                f"Entity {processed_entities + 1}/{len(entities)}: {entity['name']} "
                f"({entity['wikidata_id']})"
            )
            reporter(f"  Claim extraction over {len(chunks)} chunks")

        for chunk in chunks:
            prompt = _build_claim_prompt(
                entity_name=entity["name"],
                wikipedia_title=chunk["wikipedia_title"],
                chunk_index=chunk["chunk_index"],
                chunk_text=chunk["chunk_text"],
                max_claims=settings.claim_max_per_chunk,
            )
            response = llm.invoke(input=prompt)
            try:
                payload = extract_json_payload(response.content)
            except ValueError:
                logger.warning(
                    "Claim extraction returned non-JSON output for %s chunk %s",
                    entity["wikidata_id"],
                    chunk["chunk_index"],
                )
                continue

            rows = _normalize_claim_rows(payload, settings.claim_max_per_chunk)
            for row in rows:
                subject = str(row.get("subject") or entity["name"]).strip() or entity["name"]
                predicate = str(row.get("predicate") or "MENTIONS").strip() or "MENTIONS"
                object_value = str(row.get("object") or "").strip() or None
                claim_text = str(row.get("claim_text") or "").strip()
                if not claim_text:
                    continue
                claim_id = _build_claim_id(
                    entity["wikidata_id"],
                    chunk["chunk_id"],
                    claim_text,
                )
                entity_claims.append(
                    ClaimRecord(
                        claim_id=claim_id,
                        claim_text=claim_text,
                        subject=subject,
                        predicate=predicate,
                        object_value=object_value,
                        claim_type=str(row.get("claim_type") or "factual").strip() or "factual",
                        confidence=_coerce_confidence(row.get("confidence")),
                        provenance_quote=str(row.get("provenance_quote") or "").strip() or None,
                        chunk_element_id=chunk["chunk_element_id"],
                        chunk_id=chunk["chunk_id"],
                        chunk_index=chunk["chunk_index"],
                        document_path=chunk["document_path"],
                        root_entity_qid=entity["wikidata_id"],
                        root_entity_name=entity["name"],
                        subject_entity_qid=_resolve_claim_entity_name(
                            subject,
                            resources=resolution_resources,
                            threshold=settings.entity_resolution_similarity_threshold,
                            candidate_limit=settings.entity_resolution_candidate_limit,
                        ),
                        object_entity_qid=_resolve_claim_entity_name(
                            object_value,
                            resources=resolution_resources,
                            threshold=settings.entity_resolution_similarity_threshold,
                            candidate_limit=settings.entity_resolution_candidate_limit,
                        ),
                        extracted_at=_utcnow_iso(),
                        extraction_model=settings.claim_extraction_llm_model,
                    )
                )

        store.upsert_claims(entity_claims)
        processed_entities += 1
        created_claims += len(entity_claims)
        if reporter:
            reporter(f"  Stored {len(entity_claims)} claims")
        if progress:
            progress(1)

    _embed_claims(store, settings)
    return {"entities": processed_entities, "claims": created_claims}


def _embed_claims(store: Neo4jStore, settings: Settings) -> None:
    rows = store.list_claims_for_embedding()
    if not rows:
        return
    embedder = build_embedder(settings)
    texts = [row["text"] for row in rows]
    ids = [row["element_id"] for row in rows]

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
    else:
        embeddings = [embedder.embed_query(text) for text in texts]

    if not embeddings:
        return
    store.create_claim_vector_index(dimensions=len(embeddings[0]))
    upsert_vectors(
        store.driver,
        ids=ids,
        embedding_property="embedding",
        embeddings=embeddings,
        entity_type=EntityType.NODE,
        neo4j_database=store.database,
    )
