from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

from christian_history_graphrag.config import Settings
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.providers import build_embedder


TYPE_EQUIVALENTS = {
    "person": {"person"},
    "event": {"event"},
    "place": {"place", "location"},
    "organization": {"organization", "group"},
    "group": {"group", "organization"},
    "text": {"text", "work"},
    "office": {"office"},
    "doctrine": {"doctrine", "concept"},
    "concept": {"concept", "doctrine"},
    "entity": {"entity"},
}


@dataclass
class EntityResolutionResources:
    alias_index: dict[str, list[dict]]
    alias_rows: list[tuple[str, str, dict]]
    alias_embeddings: Optional[list[list[float]]] = None
    embedder: Any = None


def normalize_name(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z\s]", " ", value.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_alias_index(entities: list[dict]) -> tuple[dict[str, list[dict]], list[tuple[str, str, dict]]]:
    alias_index: dict[str, list[dict]] = defaultdict(list)
    alias_rows: list[tuple[str, str, dict]] = []
    for entity in entities:
        candidate_texts = [entity.get("name") or ""]
        candidate_texts.extend(entity.get("aliases") or [])
        if entity.get("wikipedia_title"):
            candidate_texts.append(entity["wikipedia_title"])
        seen: set[str] = set()
        for candidate_text in candidate_texts:
            normalized = normalize_name(candidate_text)
            if not normalized or normalized in seen:
                continue
            alias_index[normalized].append(entity)
            alias_rows.append((normalized, candidate_text, entity))
            seen.add(normalized)
    return alias_index, alias_rows


def build_entity_resolution_resources(
    entities: list[dict],
    *,
    settings: Optional[Settings] = None,
    reporter: Optional[Callable[[str], None]] = None,
) -> EntityResolutionResources:
    alias_index, alias_rows = build_alias_index(entities)
    alias_embeddings: Optional[list[list[float]]] = None
    embedder = None

    if settings and settings.entity_resolution_semantic_enabled and alias_rows:
        try:
            embedder = build_embedder(settings)
            alias_texts = [alias_text for _, alias_text, _ in alias_rows]
            if hasattr(embedder, "embed_documents"):
                alias_embeddings = embedder.embed_documents(
                    alias_texts,
                    batch_size=max(settings.embedding_batch_size, 1),
                )
            else:
                alias_embeddings = [embedder.embed_query(text) for text in alias_texts]
        except Exception as exc:
            if reporter:
                reporter(
                    "Entity resolution semantic matching unavailable; "
                    f"falling back to lexical matching only ({exc})."
                )
            embedder = None
            alias_embeddings = None

    return EntityResolutionResources(
        alias_index=alias_index,
        alias_rows=alias_rows,
        alias_embeddings=alias_embeddings,
        embedder=embedder,
    )


def _alias_similarity(normalized_name: str, alias_normalized: str) -> float:
    if not normalized_name or not alias_normalized:
        return 0.0
    if normalized_name == alias_normalized:
        return 1.0

    query_tokens = set(normalized_name.split())
    alias_tokens = set(alias_normalized.split())
    token_overlap = 0.0
    if query_tokens and alias_tokens:
        token_overlap = len(query_tokens & alias_tokens) / max(len(query_tokens), len(alias_tokens))

    prefix_bonus = 0.0
    if alias_normalized.startswith(normalized_name) or normalized_name.startswith(alias_normalized):
        prefix_bonus = 0.08

    sequence_score = SequenceMatcher(None, normalized_name, alias_normalized).ratio()
    return min(max(sequence_score, token_overlap * 0.9 + sequence_score * 0.1) + prefix_bonus, 1.0)


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    denominator = _vector_norm(left) * _vector_norm(right)
    if not denominator:
        return 0.0
    return _dot_product(left, right) / denominator


def _type_compatibility(candidate: dict, labels: Optional[list[str]]) -> float:
    normalized_labels = {label.casefold() for label in (labels or []) if label}
    if not normalized_labels:
        return 0.5

    candidate_kind = (candidate.get("entity_kind") or "").casefold()
    if not candidate_kind or candidate_kind == "entity":
        return 0.45

    compatible_kinds = TYPE_EQUIVALENTS.get(candidate_kind, {candidate_kind})
    if compatible_kinds & normalized_labels:
        return 1.0
    return 0.0


def _period_overlap(
    left_start: Optional[int],
    left_end: Optional[int],
    right_start: Optional[int],
    right_end: Optional[int],
) -> bool:
    if left_start is None and left_end is None:
        return False
    if right_start is None and right_end is None:
        return False

    resolved_left_start = left_start if left_start is not None else left_end
    resolved_left_end = left_end if left_end is not None else left_start
    resolved_right_start = right_start if right_start is not None else right_end
    resolved_right_end = right_end if right_end is not None else right_start

    if resolved_left_start is None or resolved_left_end is None:
        return False
    if resolved_right_start is None or resolved_right_end is None:
        return False
    return not (resolved_left_end < resolved_right_start or resolved_right_end < resolved_left_start)


def _context_alignment(candidate: dict, source_context: Optional[dict]) -> float:
    if not source_context:
        return 0.0

    score = 0.0
    candidate_names = {
        normalize_name(candidate.get("name") or ""),
        normalize_name(candidate.get("wikipedia_title") or ""),
    }
    candidate_names.discard("")

    source_qid = source_context.get("source_entity_qid")
    if source_qid and source_qid == candidate.get("wikidata_id"):
        score += 0.45

    document_title = normalize_name(source_context.get("document_title") or "")
    if document_title and document_title in candidate_names:
        score += 0.25

    if _period_overlap(
        candidate.get("time_start_year"),
        candidate.get("time_end_year"),
        source_context.get("source_time_start_year"),
        source_context.get("source_time_end_year"),
    ):
        score += 0.15

    source_kind = (source_context.get("source_entity_kind") or "").casefold()
    candidate_kind = (candidate.get("entity_kind") or "").casefold()
    if source_kind and candidate_kind and source_kind == candidate_kind:
        score += 0.05

    return min(score, 1.0)


def _build_candidate_pool(
    normalized_name: str,
    *,
    original_name: str,
    resources: EntityResolutionResources,
    candidate_limit: int,
    query_embedding: Optional[list[float]],
) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}

    def register_candidate(
        candidate: dict,
        *,
        method: str,
        matched_text: str,
        alias_score: Optional[float] = None,
        embedding_score: Optional[float] = None,
    ) -> None:
        candidate_key = candidate.get("wikidata_id") or candidate.get("name") or matched_text
        state = pool.setdefault(
            candidate_key,
            {
                "candidate": candidate,
                "matched_text": matched_text,
                "methods": set(),
                "alias_score": 0.0,
                "embedding_score": 0.0,
            },
        )
        state["methods"].add(method)
        if alias_score is not None and alias_score >= state["alias_score"]:
            state["alias_score"] = alias_score
            state["matched_text"] = matched_text
        if embedding_score is not None and embedding_score >= state["embedding_score"]:
            state["embedding_score"] = embedding_score
            if state["alias_score"] == 0.0:
                state["matched_text"] = matched_text

    for candidate in resources.alias_index.get(normalized_name, []):
        register_candidate(
            candidate,
            method="exact_alias",
            matched_text=original_name,
            alias_score=1.0,
        )

    fuzzy_rows: list[tuple[float, str, dict]] = []
    for alias_normalized, alias_text, candidate in resources.alias_rows:
        score = _alias_similarity(normalized_name, alias_normalized)
        if score < 0.55:
            continue
        fuzzy_rows.append((score, alias_text, candidate))
    fuzzy_rows.sort(key=lambda row: row[0], reverse=True)
    for score, alias_text, candidate in fuzzy_rows[: candidate_limit * 4]:
        register_candidate(
            candidate,
            method="fuzzy_alias",
            matched_text=alias_text,
            alias_score=score,
        )

    if query_embedding is not None and resources.alias_embeddings:
        semantic_rows: list[tuple[float, int]] = []
        for index, alias_embedding in enumerate(resources.alias_embeddings):
            score = _cosine_similarity(query_embedding, alias_embedding)
            if score < 0.45:
                continue
            semantic_rows.append((score, index))
        semantic_rows.sort(key=lambda row: row[0], reverse=True)
        for score, index in semantic_rows[: candidate_limit * 4]:
            _, alias_text, candidate = resources.alias_rows[index]
            register_candidate(
                candidate,
                method="semantic_alias",
                matched_text=alias_text,
                embedding_score=score,
            )

    return pool


def match_entity_name(
    name: str,
    *,
    alias_index: Optional[dict[str, list[dict]]] = None,
    alias_rows: Optional[list[tuple[str, str, dict]]] = None,
    threshold: float,
    labels: Optional[list[str]] = None,
    source_context: Optional[dict] = None,
    resources: Optional[EntityResolutionResources] = None,
    candidate_limit: int = 12,
) -> Optional[dict]:
    normalized_name = normalize_name(name)
    if not normalized_name:
        return None

    if resources is None:
        if alias_index is None or alias_rows is None:
            raise ValueError("alias_index and alias_rows are required when resources are not provided")
        resources = EntityResolutionResources(
            alias_index=alias_index,
            alias_rows=alias_rows,
        )

    query_embedding: Optional[list[float]] = None
    if resources.embedder is not None:
        query_embedding = resources.embedder.embed_query(name)

    candidate_pool = _build_candidate_pool(
        normalized_name,
        original_name=name,
        resources=resources,
        candidate_limit=max(candidate_limit, 1),
        query_embedding=query_embedding,
    )
    if not candidate_pool:
        return None

    best_match: Optional[dict[str, Any]] = None
    for state in candidate_pool.values():
        candidate = state["candidate"]
        alias_score = float(state.get("alias_score") or 0.0)
        embedding_score = float(state.get("embedding_score") or 0.0)
        type_score = _type_compatibility(candidate, labels)
        context_score = _context_alignment(candidate, source_context)

        if "exact_alias" in state["methods"]:
            score = 0.88 + 0.06 * max(embedding_score, 0.0) + 0.04 * type_score + 0.02 * context_score
            method = "exact_alias"
        else:
            score = (
                0.62 * alias_score
                + 0.18 * max(embedding_score, 0.0)
                + 0.12 * type_score
                + 0.08 * context_score
            )
            if embedding_score > alias_score:
                method = "semantic_alias"
            elif embedding_score > 0.0:
                method = "hybrid_alias"
            else:
                method = "fuzzy_alias"

        score = max(0.0, min(score, 1.0))
        match = {
            "candidate": candidate,
            "method": method,
            "score": score,
            "matched_text": state["matched_text"],
            "alias_score": alias_score,
            "embedding_score": embedding_score,
            "type_score": type_score,
            "context_score": context_score,
        }
        if best_match is None or (
            match["score"],
            match["type_score"],
            match["alias_score"],
            candidate.get("name") or "",
            candidate.get("wikidata_id") or "",
        ) > (
            best_match["score"],
            best_match["type_score"],
            best_match["alias_score"],
            best_match["candidate"].get("name") or "",
            best_match["candidate"].get("wikidata_id") or "",
        ):
            best_match = match

    if best_match and best_match["score"] >= threshold:
        return best_match
    return None


def resolve_extracted_entities(
    store: Neo4jStore,
    settings: Settings,
    *,
    replace_existing: bool = True,
    reporter: Optional[Callable[[str], None]] = None,
) -> int:
    entities = store.list_entities_for_resolution()
    extracted_nodes = store.list_extracted_nodes_for_resolution()
    if replace_existing:
        store.clear_resolution_links()

    if not entities or not extracted_nodes:
        return 0

    resources = build_entity_resolution_resources(
        entities,
        settings=settings,
        reporter=reporter,
    )

    resolved = 0
    threshold = settings.entity_resolution_similarity_threshold
    for node in extracted_nodes:
        normalized_name = normalize_name(node["name"])
        if not normalized_name:
            continue

        match = match_entity_name(
            node["name"],
            threshold=threshold,
            labels=node.get("labels", []),
            source_context=node,
            resources=resources,
            candidate_limit=settings.entity_resolution_candidate_limit,
        )
        if match:
            store.link_resolution(
                node_element_id=node["element_id"],
                wikidata_id=match["candidate"]["wikidata_id"],
                method=match["method"],
                score=match["score"],
                matched_text=match["matched_text"],
                alias_score=match["alias_score"],
                embedding_score=match["embedding_score"],
                type_score=match["type_score"],
                context_score=match["context_score"],
            )
            resolved += 1

    if reporter:
        reporter(f"Entity resolution linked {resolved} extracted nodes to canonical entities.")
    return resolved
