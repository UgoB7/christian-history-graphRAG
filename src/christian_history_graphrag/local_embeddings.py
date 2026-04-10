from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from neo4j_graphrag.embeddings.base import Embedder


logger = logging.getLogger(__name__)


def _resolve_local_sentence_transformer_path(model_name: str) -> str | None:
    model_slug = model_name.replace("/", "--")
    snapshot_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_slug}"
        / "snapshots"
    )
    if not snapshot_root.exists():
        return None

    candidates = [path for path in snapshot_root.iterdir() if path.is_dir()]
    if not candidates:
        return None

    preferred_candidates = [
        path
        for path in candidates
        if (path / "modules.json").exists() and (path / "config.json").exists()
    ]
    if preferred_candidates:
        preferred_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return str(preferred_candidates[0])

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


class LocalSentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        query_prefix: str = "",
        passage_prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install dependencies with `pip install -e .`."
            ) from exc

        local_model_path = _resolve_local_sentence_transformer_path(model_name)
        resolved_model_name = local_model_path or model_name
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        local_files_only = bool(local_model_path)

        try:
            self.model = SentenceTransformer(
                resolved_model_name,
                device=device,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                **kwargs,
            )
        except RuntimeError as exc:
            if device.lower() != "cpu" and "MPS backend is supported on MacOS 14.0+" in str(exc):
                logger.warning(
                    "Embedding device %s unavailable on this host; retrying with cpu.",
                    device,
                )
                self.model = SentenceTransformer(
                    resolved_model_name,
                    device="cpu",
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    **kwargs,
                )
            else:
                raise
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed_query(self, text: str) -> list[float]:
        encoded = self.model.encode(
            self.query_prefix + text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return encoded.tolist()

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        encoded = self.model.encode(
            [self.passage_prefix + text for text in texts],
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return encoded.tolist()
