from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from christian_history_graphrag.models import (
    EntityRecord,
    entity_record_from_dict,
    entity_record_to_dict,
)


class IngestCheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        *,
        seed_qids: list[str],
        depth: int,
        language: str,
        wikipedia_enabled: bool,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.checkpoint_dir = Path(checkpoint_dir)
        if self.enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        key = json.dumps(
            {
                "seed_qids": sorted(seed_qids),
                "depth": depth,
                "language": language,
                "wikipedia_enabled": wikipedia_enabled,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        self.path = self.checkpoint_dir / f"ingest-{digest}.json"

    def load_stage(self, stage: str) -> Optional[dict[str, EntityRecord]]:
        if not self.enabled or not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        stage_payload = payload.get("stages", {}).get(stage)
        if not stage_payload:
            return None
        return {
            qid: entity_record_from_dict(record_payload)
            for qid, record_payload in stage_payload["records"].items()
        }

    def save_stage(self, stage: str, records: dict[str, EntityRecord]) -> None:
        if not self.enabled:
            return
        payload = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload.setdefault("stages", {})
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["stages"][stage] = {
            "saved_at": payload["updated_at"],
            "records": {
                qid: entity_record_to_dict(record)
                for qid, record in records.items()
            },
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
