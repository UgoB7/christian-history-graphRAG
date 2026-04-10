from __future__ import annotations

from datetime import UTC, datetime
from typing import Callable, Optional

from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.types import EntityType

from christian_history_graphrag.config import Settings
from christian_history_graphrag.llm_json import extract_json_payload
from christian_history_graphrag.models import CommunityReportRecord
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.providers import build_embedder, build_llm


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _build_report_prompt(source_pack: dict) -> str:
    entity = source_pack["entity"]
    nodes = source_pack["nodes"]
    relations = source_pack["relations"]
    claims = source_pack["claims"]
    resolved = source_pack["resolved_entities"]

    node_lines = [
        f"- {node['name']} ({', '.join(node['labels'])}) mentions={node['mentions']}"
        for node in nodes
        if node.get("name")
    ]
    relation_lines = [
        f"- {row['source']} -[{row['relation']}]-> {row['target']} mentions={row['mentions']}"
        for row in relations
    ]
    claim_lines = [
        f"- {row['claim_text']} (confidence={row['confidence']})"
        for row in claims
    ]
    resolved_lines = [
        f"- {row['name']} ({row['wikidata_id']}) mentions={row['mentions']}"
        for row in resolved
    ]
    return f"""
You are writing a community report for a Christian history GraphRAG system.
Return only valid JSON with this exact shape:
{{
  "title": "short title",
  "summary": "120-220 word synthesis",
  "themes": ["theme 1", "theme 2"],
  "key_entities": ["entity 1", "entity 2"],
  "key_claims": ["claim 1", "claim 2"],
  "question_suggestions": ["question 1", "question 2", "question 3"]
}}

Rules:
- Summarize the neighborhood around the focus entity.
- Prefer grounded themes that are visible in the nodes, relations, and claims.
- Keep `themes` to 3-6 items.
- Keep `key_entities` to 3-8 items.
- Keep `key_claims` to 2-6 items.

Focus entity: {entity['name']} ({entity['wikidata_id']})
Period: {entity.get('time_start_year')} -> {entity.get('time_end_year')}
Wikipedia: {entity.get('wikipedia_url')}

Resolved canonical entities:
{chr(10).join(resolved_lines) if resolved_lines else "- none"}

Top extracted nodes:
{chr(10).join(node_lines) if node_lines else "- none"}

Top extracted relations:
{chr(10).join(relation_lines) if relation_lines else "- none"}

Top claims:
{chr(10).join(claim_lines) if claim_lines else "- none"}
""".strip()


def _fallback_report(source_pack: dict) -> dict:
    entity = source_pack["entity"]
    nodes = [row["name"] for row in source_pack["nodes"][:6] if row.get("name")]
    claims = [row["claim_text"] for row in source_pack["claims"][:4] if row.get("claim_text")]
    themes = []
    if source_pack["claims"]:
        themes.append("Source-backed claims")
    if source_pack["relations"]:
        themes.append("Extracted semantic relations")
    if source_pack["resolved_entities"]:
        themes.append("Resolved canonical entities")
    if not themes:
        themes.append("Entity neighborhood")
    summary = (
        f"This report summarizes the enriched graph neighborhood around {entity['name']}. "
        f"It includes {len(source_pack['nodes'])} extracted nodes, "
        f"{len(source_pack['relations'])} salient relations, and "
        f"{len(source_pack['claims'])} claims linked to the entity's KG documents."
    )
    return {
        "title": f"{entity['name']} community report",
        "summary": summary,
        "themes": themes,
        "key_entities": nodes[:6],
        "key_claims": claims[:4],
        "question_suggestions": [
            f"What themes dominate the neighborhood around {entity['name']}?",
            f"Which related people, places, or texts cluster around {entity['name']}?",
            f"What claims in the graph are best supported for {entity['name']}?",
        ],
    }


def _coerce_report(payload: object, source_pack: dict) -> dict:
    if not isinstance(payload, dict):
        return _fallback_report(source_pack)
    report = _fallback_report(source_pack)
    for key in ("title", "summary"):
        if isinstance(payload.get(key), str) and payload[key].strip():
            report[key] = payload[key].strip()
    for key in ("themes", "key_entities", "key_claims", "question_suggestions"):
        if isinstance(payload.get(key), list):
            values = [
                str(item).strip()
                for item in payload[key]
                if str(item).strip()
            ]
            if values:
                report[key] = values
    return report


def _render_report_text(report: dict, entity: dict) -> str:
    lines = [
        f"Community: {report['title']}",
        f"Focus entity: {entity['name']} ({entity['wikidata_id']})",
        f"Period: {entity.get('time_start_year')} -> {entity.get('time_end_year')}",
        "Summary:",
        report["summary"],
    ]
    if report["themes"]:
        lines.append("Themes: " + "; ".join(report["themes"]))
    if report["key_entities"]:
        lines.append("Key entities: " + "; ".join(report["key_entities"]))
    if report["key_claims"]:
        lines.append("Key claims: " + "; ".join(report["key_claims"]))
    if report["question_suggestions"]:
        lines.append("Suggested questions: " + "; ".join(report["question_suggestions"]))
    return "\n".join(lines)


def build_community_reports(
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
        return {"communities": 0, "reports": 0}

    store.ensure_community_indexes()
    llm = build_llm(settings, model_name=settings.community_report_llm_model)

    built_communities = 0
    built_reports = 0
    for entity in entities:
        if replace_existing:
            store.delete_communities_for_entity(entity["wikidata_id"])

        source_pack = store.get_entity_community_source_pack(
            entity["wikidata_id"],
            member_limit=settings.community_report_member_limit,
            relation_limit=settings.community_report_relation_limit,
            claim_limit=settings.community_report_claim_limit,
        )
        focus = source_pack.get("entity")
        if not focus:
            if progress:
                progress(1)
            continue

        if reporter:
            reporter(
                f"Entity {built_communities + 1}/{len(entities)}: {focus['name']} "
                f"({focus['wikidata_id']})"
            )
            reporter(
                f"  Community source pack: {len(source_pack['nodes'])} nodes, "
                f"{len(source_pack['relations'])} relations, {len(source_pack['claims'])} claims"
            )

        prompt = _build_report_prompt(source_pack)
        response = llm.invoke(input=prompt)
        try:
            payload = extract_json_payload(response.content)
        except ValueError:
            payload = _fallback_report(source_pack)
        report_payload = _coerce_report(payload, source_pack)

        community_id = f"community:entity:{focus['wikidata_id']}"
        report_id = f"community-report:entity:{focus['wikidata_id']}"
        store.upsert_community(
            community_id=community_id,
            title=report_payload["title"],
            focus_entity_qid=focus["wikidata_id"],
            focus_entity_name=focus["name"],
            source_url=focus.get("wikipedia_url"),
            time_start_year=focus.get("time_start_year"),
            time_end_year=focus.get("time_end_year"),
        )

        member_ids = [focus["element_id"]]
        member_ids.extend(row["element_id"] for row in source_pack["nodes"] if row.get("element_id"))
        member_ids.extend(
            row["element_id"]
            for row in source_pack["resolved_entities"]
            if row.get("element_id")
        )
        deduped_member_ids = list(dict.fromkeys(member_ids))
        store.replace_community_members(community_id, deduped_member_ids)
        store.replace_community_claims(
            community_id,
            [row["claim_id"] for row in source_pack["claims"] if row.get("claim_id")],
        )

        report_record = CommunityReportRecord(
            community_id=community_id,
            report_id=report_id,
            title=report_payload["title"],
            summary=report_payload["summary"],
            themes=report_payload["themes"],
            key_entities=report_payload["key_entities"],
            key_claims=report_payload["key_claims"],
            question_suggestions=report_payload["question_suggestions"],
            report_text=_render_report_text(report_payload, focus),
            focus_entity_qid=focus["wikidata_id"],
            focus_entity_name=focus["name"],
            source_url=focus.get("wikipedia_url"),
            time_start_year=focus.get("time_start_year"),
            time_end_year=focus.get("time_end_year"),
            generated_at=_utcnow_iso(),
            generation_model=settings.community_report_llm_model,
        )
        store.upsert_community_report(report_record)
        built_communities += 1
        built_reports += 1
        if reporter:
            reporter(f"  Community report stored with {len(deduped_member_ids)} members")
        if progress:
            progress(1)

    _embed_community_reports(store, settings)
    return {"communities": built_communities, "reports": built_reports}


def _embed_community_reports(store: Neo4jStore, settings: Settings) -> None:
    rows = store.list_community_reports_for_embedding()
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
    store.create_community_report_vector_index(dimensions=len(embeddings[0]))
    upsert_vectors(
        store.driver,
        ids=ids,
        embedding_property="embedding",
        embeddings=embeddings,
        entity_type=EntityType.NODE,
        neo4j_database=store.database,
    )
