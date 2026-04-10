from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


def build_retry_session(
    *,
    user_agent: str,
    max_retries: int,
    backoff_factor: float,
    session: Optional[requests.Session] = None,
) -> requests.Session:
    resolved_session = session or requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    resolved_session.mount("https://", adapter)
    resolved_session.mount("http://", adapter)
    resolved_session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": user_agent,
        }
    )
    return resolved_session


class FileHTTPCache:
    def __init__(
        self,
        cache_dir: str,
        *,
        enabled: bool = True,
        ttl_seconds: int = 60 * 60 * 24 * 7,
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.cache_dir = Path(cache_dir)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_json(self, namespace: str, url: str, params: Optional[dict[str, Any]] = None) -> Any:
        payload = self._read(namespace, url, params)
        if payload is None:
            return None
        return json.loads(payload)

    def set_json(
        self,
        namespace: str,
        url: str,
        params: Optional[dict[str, Any]],
        payload: Any,
    ) -> None:
        self._write(namespace, url, params, json.dumps(payload))

    def _read(self, namespace: str, url: str, params: Optional[dict[str, Any]]) -> Optional[str]:
        if not self.enabled:
            return None
        path = self._cache_path(namespace, url, params)
        if not path.exists():
            return None
        age_seconds = time.time() - path.stat().st_mtime
        if self.ttl_seconds >= 0 and age_seconds > self.ttl_seconds:
            logger.debug("Cache expired for %s", path)
            return None
        logger.debug("Cache hit for %s", path)
        return path.read_text(encoding="utf-8")

    def _write(
        self,
        namespace: str,
        url: str,
        params: Optional[dict[str, Any]],
        payload: str,
    ) -> None:
        if not self.enabled:
            return
        path = self._cache_path(namespace, url, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    def _cache_path(
        self,
        namespace: str,
        url: str,
        params: Optional[dict[str, Any]],
    ) -> Path:
        normalized = json.dumps(
            {
                "url": url,
                "params": params or {},
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.cache_dir / namespace / f"{digest}.json"
