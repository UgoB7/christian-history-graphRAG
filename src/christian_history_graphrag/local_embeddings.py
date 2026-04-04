from __future__ import annotations

from typing import Any

from neo4j_graphrag.embeddings.base import Embedder


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

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=kwargs.pop("trust_remote_code", True),
            **kwargs,
        )
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
