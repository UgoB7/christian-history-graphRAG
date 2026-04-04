from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Optional, Tuple

import requests

from christian_history_graphrag.constants import (
    EVENT_INSTANCE_IDS,
    EXPANSION_PROPERTIES,
    ORGANIZATION_INSTANCE_IDS,
    PERSON_INSTANCE_IDS,
    PLACE_INSTANCE_IDS,
    RELATION_PROPERTY_MAP,
    TIME_PROPERTIES,
)
from christian_history_graphrag.models import EntityRecord, EntityRelation

WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"


class WikidataClient:
    def __init__(self, language: str = "en", session: Optional[requests.Session] = None):
        self.language = language
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "christian-history-graphrag/0.1",
            }
        )

    def fetch_entity(self, qid: str) -> EntityRecord:
        response = self.session.get(
            WIKIDATA_ENTITY_URL.format(qid=qid),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        entity = payload["entities"][qid]
        claims = entity.get("claims", {})

        instance_of = self._collect_qids(claims, "P31")
        wikipedia_title = self._extract_wikipedia_title(entity)

        start_time = self._extract_first_time(claims, TIME_PROPERTIES["start"])
        end_time = self._extract_first_time(claims, TIME_PROPERTIES["end"])
        latitude, longitude = self._extract_coordinates(claims)

        relations = []
        for property_id in EXPANSION_PROPERTIES:
            for target_qid in self._collect_qids(claims, property_id):
                relations.append(
                    EntityRelation(
                        property_id=property_id,
                        relation_type=RELATION_PROPERTY_MAP.get(property_id, "RELATED_TO"),
                        target_qid=target_qid,
                    )
                )

        return EntityRecord(
            qid=qid,
            label=self._localized_value(entity.get("labels", {})) or qid,
            description=self._localized_value(entity.get("descriptions", {})),
            aliases=self._localized_aliases(entity.get("aliases", {})),
            entity_kind=self._infer_entity_kind(instance_of, latitude, longitude),
            instance_of=instance_of,
            start_time=start_time,
            end_time=end_time,
            start_year=self._year_from_time(start_time),
            end_year=self._year_from_time(end_time),
            latitude=latitude,
            longitude=longitude,
            wikipedia_title=wikipedia_title,
            wikipedia_url=(
                f"https://{self.language}.wikipedia.org/wiki/{wikipedia_title.replace(' ', '_')}"
                if wikipedia_title
                else None
            ),
            relations=relations,
        )

    def expand_subgraph(self, seed_qids: list[str], max_depth: int = 1) -> dict[str, EntityRecord]:
        queue: deque[tuple[str, int]] = deque((qid, 0) for qid in seed_qids)
        visited: set[str] = set()
        records: dict[str, EntityRecord] = {}

        while queue:
            qid, depth = queue.popleft()
            if qid in visited:
                continue
            visited.add(qid)

            record = self.fetch_entity(qid)
            records[qid] = record

            if depth >= max_depth:
                continue

            for relation in record.relations:
                if relation.target_qid not in visited:
                    queue.append((relation.target_qid, depth + 1))

        return records

    def _extract_wikipedia_title(self, entity: dict[str, Any]) -> Optional[str]:
        site_key = f"{self.language}wiki"
        sitelink = entity.get("sitelinks", {}).get(site_key)
        if not sitelink:
            return None
        return sitelink.get("title")

    def _localized_value(self, values: dict[str, dict[str, str]]) -> Optional[str]:
        if self.language in values:
            return values[self.language].get("value")
        if "en" in values:
            return values["en"].get("value")
        for candidate in values.values():
            return candidate.get("value")
        return None

    def _localized_aliases(self, aliases: dict[str, list[dict[str, str]]]) -> list[str]:
        candidates = aliases.get(self.language) or aliases.get("en") or []
        return [entry["value"] for entry in candidates]

    def _collect_qids(self, claims: dict[str, list[dict[str, Any]]], property_id: str) -> list[str]:
        qids: list[str] = []
        for claim in claims.get(property_id, []):
            datavalue = self._datavalue(claim)
            if not datavalue or datavalue.get("type") != "wikibase-entityid":
                continue
            entity_id = datavalue["value"].get("id")
            if entity_id:
                qids.append(entity_id)
        return qids

    def _extract_first_time(
        self, claims: dict[str, list[dict[str, Any]]], property_ids: tuple[str, ...]
    ) -> Optional[str]:
        for property_id in property_ids:
            for claim in claims.get(property_id, []):
                datavalue = self._datavalue(claim)
                if not datavalue or datavalue.get("type") != "time":
                    continue
                return datavalue["value"].get("time")
        return None

    def _extract_coordinates(
        self, claims: dict[str, list[dict[str, Any]]]
    ) -> Tuple[Optional[float], Optional[float]]:
        for claim in claims.get("P625", []):
            datavalue = self._datavalue(claim)
            if not datavalue or datavalue.get("type") != "globecoordinate":
                continue
            value = datavalue["value"]
            return value.get("latitude"), value.get("longitude")
        return None, None

    def _datavalue(self, claim: dict[str, Any]) -> Optional[dict[str, Any]]:
        mainsnak = claim.get("mainsnak", {})
        return mainsnak.get("datavalue")

    def _year_from_time(self, value: Optional[str]) -> Optional[int]:
        if not value:
            return None
        normalized = value.lstrip("+")
        if normalized.startswith("-"):
            digits = normalized[1:].split("-")[0]
            if digits.isdigit():
                return -int(digits)
            return None
        prefix = normalized.split("T")[0]
        year = prefix.split("-")[0]
        if year.isdigit():
            return int(year)
        try:
            return datetime.fromisoformat(prefix).year
        except ValueError:
            return None

    def _infer_entity_kind(
        self,
        instance_of: list[str],
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> str:
        if any(item in PERSON_INSTANCE_IDS for item in instance_of):
            return "Person"
        if any(item in EVENT_INSTANCE_IDS for item in instance_of):
            return "Event"
        if latitude is not None and longitude is not None:
            return "Place"
        if any(item in PLACE_INSTANCE_IDS for item in instance_of):
            return "Place"
        if any(item in ORGANIZATION_INSTANCE_IDS for item in instance_of):
            return "Organization"
        return "Entity"
