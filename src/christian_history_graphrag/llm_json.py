from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", fenced)
        fenced = re.sub(r"\s*```$", "", fenced)
    return fenced.strip()


def extract_json_payload(text: str) -> Any:
    cleaned = _strip_code_fences(text)
    candidates = [cleaned]

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start != -1 and object_end != -1 and object_start < object_end:
        candidates.append(cleaned[object_start : object_end + 1])

    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start != -1 and array_end != -1 and array_start < array_end:
        candidates.append(cleaned[array_start : array_end + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON payload found in LLM response")
