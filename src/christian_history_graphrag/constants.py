from __future__ import annotations

DEFAULT_WIKIPEDIA_LANGUAGE = "en"

RELATION_PROPERTY_MAP = {
    "P22": "FATHER_OF",
    "P25": "MOTHER_OF",
    "P26": "SPOUSE_OF",
    "P31": "INSTANCE_OF",
    "P39": "HELD_POSITION",
    "P40": "PARENT_OF",
    "P69": "EDUCATED_AT",
    "P106": "HAS_OCCUPATION",
    "P112": "FOUNDED_BY",
    "P131": "IN_ADMIN_AREA",
    "P140": "HAS_RELIGION",
    "P276": "LOCATED_IN",
    "P361": "PART_OF",
    "P463": "MEMBER_OF",
    "P527": "HAS_PART",
    "P569": "BORN_ON",
    "P570": "DIED_ON",
    "P571": "FOUNDED_ON",
    "P580": "STARTED_ON",
    "P582": "ENDED_ON",
    "P607": "PART_OF_CONFLICT",
    "P710": "HAS_PARTICIPANT",
    "P737": "INFLUENCED_BY",
    "P802": "STUDENT_OF",
    "P859": "SPONSORED_BY",
    "P937": "WORK_LOCATION",
}

EXPANSION_PROPERTIES = tuple(
    property_id
    for property_id in RELATION_PROPERTY_MAP
    if property_id
    not in {
        "P569",
        "P570",
        "P571",
        "P580",
        "P582",
    }
)

TIME_PROPERTIES = {
    "start": ("P580", "P571", "P569", "P585"),
    "end": ("P582", "P570", "P585"),
}

PERSON_INSTANCE_IDS = {
    "Q5",
}

EVENT_INSTANCE_IDS = {
    "Q1190554",
    "Q1656682",
    "Q198",
    "Q178561",
    "Q40231",
}

PLACE_INSTANCE_IDS = {
    "Q618123",
    "Q486972",
    "Q2221906",
    "Q515",
    "Q6256",
}

ORGANIZATION_INSTANCE_IDS = {
    "Q43229",
    "Q163740",
    "Q245065",
    "Q327333",
}
