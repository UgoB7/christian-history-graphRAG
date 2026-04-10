from __future__ import annotations

from dataclasses import dataclass

from christian_history_graphrag.config import Settings
from christian_history_graphrag.llm_json import extract_json_payload
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.providers import build_llm


@dataclass
class RouterDecision:
    route: str
    reason: str


ROUTE_DESCRIPTIONS = {
    "local": "Base passage retrieval over canonical Passage nodes from ingest+embed.",
    "hybrid": "Hybrid retrieval over KG chunks and extracted subgraph context from kg-enrich.",
    "claims": "Claim-centric retrieval over source-backed Claim nodes extracted from KG chunks.",
    "global": "Global retrieval over CommunityReport summaries for broader thematic questions.",
    "cypher": "Structured graph querying with Text2Cypher for list/count/filter questions.",
}


def _heuristic_route(question: str, available_routes: set[str]) -> RouterDecision:
    lowered = question.casefold()
    if "claim" in lowered or "assert" in lowered or "source-backed" in lowered:
        if "claims" in available_routes:
            return RouterDecision("claims", "The question asks for claims or explicit source-backed assertions.")
    if any(
        token in lowered
        for token in (
            "global",
            "overall",
            "across the corpus",
            "big picture",
            "theme",
            "themes",
            "trend",
            "tendances",
            "panorama",
            "ensemble du corpus",
        )
    ):
        if "global" in available_routes:
            return RouterDecision("global", "The question is broad and benefits from summarized community reports.")
    if any(
        token in lowered
        for token in (
            "list",
            "show",
            "which",
            "what are",
            "count",
            "how many",
            "combien",
            "liste",
            "montre",
            "quelle liste",
            "statements",
            "source-backed statements",
        )
    ):
        if "cypher" in available_routes:
            return RouterDecision("cypher", "The question looks structured and is a good fit for graph querying.")
    if "hybrid" in available_routes:
        return RouterDecision("hybrid", "Defaulting to hybrid retrieval for entity-centric narrative context.")
    if "local" in available_routes:
        return RouterDecision("local", "Falling back to base passage retrieval.")
    return RouterDecision("cypher", "Falling back to the structured graph route.")


def choose_route(store: Neo4jStore, settings: Settings, question: str) -> RouterDecision:
    available_routes = {"cypher"}
    if store.has_passages():
        available_routes.add("local")
    if store.has_kg_chunks():
        available_routes.add("hybrid")
    if store.has_claims():
        available_routes.add("claims")
    if store.has_community_reports():
        available_routes.add("global")

    prompt = f"""
You are routing a Christian history GraphRAG question to the best retrieval tool.
Available routes:
{chr(10).join(f"- {name}: {ROUTE_DESCRIPTIONS[name]}" for name in sorted(available_routes))}

Return only valid JSON:
{{
  "route": "one available route",
  "reason": "short reason"
}}

Question: {question}
""".strip()

    try:
        response = build_llm(settings, model_name=settings.router_llm_model).invoke(input=prompt)
        payload = extract_json_payload(response.content)
        route = str(payload.get("route", "")).strip().lower()
        reason = str(payload.get("reason", "")).strip()
        if route in available_routes:
            return RouterDecision(route=route, reason=reason or "LLM-selected route.")
    except Exception:
        pass

    return _heuristic_route(question, available_routes)
