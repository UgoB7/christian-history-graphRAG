from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Callable, Optional

from christian_history_graphrag.config import Settings
from christian_history_graphrag.neo4j_store import Neo4jStore


def normalize_name(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z\s]", " ", value.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _best_entity_match(candidates: list[dict], labels: list[str]) -> dict:
    if not candidates:
        raise ValueError("Expected at least one candidate")
    normalized_labels = {label.casefold() for label in labels}
    candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.get("entity_kind", "").casefold() not in normalized_labels,
            candidate.get("name") or "",
            candidate.get("wikidata_id") or "",
        ),
    )
    return candidates[0]


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


def match_entity_name(
    name: str,
    *,
    alias_index: dict[str, list[dict]],
    alias_rows: list[tuple[str, str, dict]],
    threshold: float,
    labels: Optional[list[str]] = None,
) -> Optional[dict]:
    normalized_name = normalize_name(name)
    if not normalized_name:
        return None

    exact_candidates = alias_index.get(normalized_name, [])
    if exact_candidates:
        candidate = _best_entity_match(exact_candidates, labels or [])
        return {
            "candidate": candidate,
            "method": "exact_alias",
            "score": 1.0,
            "matched_text": name,
        }

    best_score = 0.0
    best_alias = None
    best_candidate = None
    for alias_normalized, alias_text, candidate in alias_rows:
        score = SequenceMatcher(None, normalized_name, alias_normalized).ratio()
        if score > best_score:
            best_score = score
            best_alias = alias_text
            best_candidate = candidate
    if best_candidate and best_score >= threshold:
        return {
            "candidate": best_candidate,
            "method": "fuzzy_alias",
            "score": best_score,
            "matched_text": best_alias or name,
        }
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

    alias_index, alias_rows = build_alias_index(entities)

    resolved = 0
    threshold = settings.entity_resolution_similarity_threshold
    for node in extracted_nodes:
        normalized_name = normalize_name(node["name"])
        if not normalized_name:
            continue

        match = match_entity_name(
            node["name"],
            alias_index=alias_index,
            alias_rows=alias_rows,
            threshold=threshold,
            labels=node.get("labels", []),
        )
        if match:
            store.link_resolution(
                node_element_id=node["element_id"],
                wikidata_id=match["candidate"]["wikidata_id"],
                method=match["method"],
                score=match["score"],
                matched_text=match["matched_text"],
            )
            resolved += 1

    if reporter:
        reporter(f"Entity resolution linked {resolved} extracted nodes to canonical entities.")
    return resolved
