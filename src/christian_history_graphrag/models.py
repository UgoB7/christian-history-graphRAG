from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EntityRelation:
    property_id: str
    relation_type: str
    target_qid: str


@dataclass
class WikipediaPassage:
    passage_id: str
    page_title: str
    url: str
    language: str
    chunk_index: int
    text: str


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
    passages: list[WikipediaPassage] = field(default_factory=list)
    relations: list[EntityRelation] = field(default_factory=list)
