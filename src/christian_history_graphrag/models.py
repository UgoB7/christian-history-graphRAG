from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class EntityRelation:
    property_id: str
    relation_type: str
    target_qid: str
    statement_id: Optional[str] = None
    rank: str = "normal"
    qualifiers: list[dict[str, Any]] = field(default_factory=list)
    reference_count: int = 0
    reference_property_ids: list[str] = field(default_factory=list)
    source_system: str = "wikidata"
    source_url: Optional[str] = None
    source_document_id: Optional[str] = None
    retrieved_at: Optional[str] = None


@dataclass
class SourceDocument:
    source_id: str
    source_system: str
    source_url: str
    title: str
    language: Optional[str] = None
    revision_id: Optional[int] = None
    content_hash: Optional[str] = None
    retrieved_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WikipediaPassage:
    passage_id: str
    page_title: str
    url: str
    language: str
    chunk_index: int
    text: str
    source_document_id: Optional[str] = None
    source_system: str = "wikipedia"
    retrieved_at: Optional[str] = None
    revision_id: Optional[int] = None
    content_hash: Optional[str] = None


@dataclass
class EntityRecord:
    qid: str
    label: str
    description: Optional[str] = None
    entity_kind: str = "Entity"
    aliases: list[str] = field(default_factory=list)
    instance_of: list[str] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    wikipedia_title: Optional[str] = None
    wikipedia_url: Optional[str] = None
    wikidata_url: Optional[str] = None
    retrieved_at: Optional[str] = None
    seed_distance: Optional[int] = None
    passages: list[WikipediaPassage] = field(default_factory=list)
    relations: list[EntityRelation] = field(default_factory=list)
    source_documents: list[SourceDocument] = field(default_factory=list)


@dataclass
class ClaimRecord:
    claim_id: str
    claim_text: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_value: Optional[str] = None
    claim_type: str = "factual"
    confidence: float = 0.5
    provenance_quote: Optional[str] = None
    chunk_element_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    document_path: Optional[str] = None
    root_entity_qid: Optional[str] = None
    root_entity_name: Optional[str] = None
    subject_entity_qid: Optional[str] = None
    object_entity_qid: Optional[str] = None
    extracted_at: Optional[str] = None
    extraction_model: Optional[str] = None


@dataclass
class CommunityReportRecord:
    community_id: str
    report_id: str
    title: str
    summary: str
    themes: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    key_claims: list[str] = field(default_factory=list)
    question_suggestions: list[str] = field(default_factory=list)
    report_text: str = ""
    focus_entity_qid: Optional[str] = None
    focus_entity_name: Optional[str] = None
    source_url: Optional[str] = None
    time_start_year: Optional[int] = None
    time_end_year: Optional[int] = None
    generated_at: Optional[str] = None
    generation_model: Optional[str] = None


def entity_record_to_dict(record: EntityRecord) -> dict[str, Any]:
    return asdict(record)


def entity_record_from_dict(payload: dict[str, Any]) -> EntityRecord:
    return EntityRecord(
        qid=payload["qid"],
        label=payload["label"],
        description=payload.get("description"),
        entity_kind=payload.get("entity_kind", "Entity"),
        aliases=list(payload.get("aliases", [])),
        instance_of=list(payload.get("instance_of", [])),
        start_time=payload.get("start_time"),
        end_time=payload.get("end_time"),
        start_year=payload.get("start_year"),
        end_year=payload.get("end_year"),
        latitude=payload.get("latitude"),
        longitude=payload.get("longitude"),
        wikipedia_title=payload.get("wikipedia_title"),
        wikipedia_url=payload.get("wikipedia_url"),
        wikidata_url=payload.get("wikidata_url"),
        retrieved_at=payload.get("retrieved_at"),
        seed_distance=payload.get("seed_distance"),
        passages=[
            WikipediaPassage(
                passage_id=passage["passage_id"],
                page_title=passage["page_title"],
                url=passage["url"],
                language=passage["language"],
                chunk_index=passage["chunk_index"],
                text=passage["text"],
                source_document_id=passage.get("source_document_id"),
                source_system=passage.get("source_system", "wikipedia"),
                retrieved_at=passage.get("retrieved_at"),
                revision_id=passage.get("revision_id"),
                content_hash=passage.get("content_hash"),
            )
            for passage in payload.get("passages", [])
        ],
        relations=[
            EntityRelation(
                property_id=relation["property_id"],
                relation_type=relation["relation_type"],
                target_qid=relation["target_qid"],
                statement_id=relation.get("statement_id"),
                rank=relation.get("rank", "normal"),
                qualifiers=list(relation.get("qualifiers", [])),
                reference_count=relation.get("reference_count", 0),
                reference_property_ids=list(relation.get("reference_property_ids", [])),
                source_system=relation.get("source_system", "wikidata"),
                source_url=relation.get("source_url"),
                source_document_id=relation.get("source_document_id"),
                retrieved_at=relation.get("retrieved_at"),
            )
            for relation in payload.get("relations", [])
        ],
        source_documents=[
            SourceDocument(
                source_id=source["source_id"],
                source_system=source["source_system"],
                source_url=source["source_url"],
                title=source["title"],
                language=source.get("language"),
                revision_id=source.get("revision_id"),
                content_hash=source.get("content_hash"),
                retrieved_at=source.get("retrieved_at"),
                metadata=dict(source.get("metadata", {})),
            )
            for source in payload.get("source_documents", [])
        ],
    )
