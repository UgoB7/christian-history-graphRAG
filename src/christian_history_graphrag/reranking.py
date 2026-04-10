from __future__ import annotations

import logging
from typing import Any

from christian_history_graphrag.local_embeddings import _resolve_local_sentence_transformer_path


logger = logging.getLogger(__name__)


class LocalCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. Install dependencies with `pip install -e .`."
            ) from exc

        local_model_path = _resolve_local_sentence_transformer_path(model_name)
        resolved_model_name = local_model_path or model_name
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        local_files_only = bool(local_model_path)

        try:
            self.model = CrossEncoder(
                resolved_model_name,
                device=device,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                **kwargs,
            )
        except RuntimeError as exc:
            if device.lower() != "cpu" and "MPS backend is supported on MacOS 14.0+" in str(exc):
                logger.warning(
                    "Reranker device %s unavailable on this host; retrying with cpu.",
                    device,
                )
                self.model = CrossEncoder(
                    resolved_model_name,
                    device="cpu",
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    **kwargs,
                )
            else:
                raise

        self.batch_size = max(batch_size, 1)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]
