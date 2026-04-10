from __future__ import annotations

from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.llm.ollama_llm import OllamaLLM

from christian_history_graphrag.config import Settings
from christian_history_graphrag.local_embeddings import LocalSentenceTransformerEmbedder


def build_embedder(settings: Settings):
    provider = settings.embedding_provider.lower()
    if provider == "sentence-transformers":
        return LocalSentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            normalize_embeddings=settings.embedding_normalize,
            query_prefix=settings.embedding_query_prefix,
            passage_prefix=settings.embedding_passage_prefix,
        )
    if provider == "ollama":
        return OllamaEmbeddings(
            model=settings.embedding_model,
            host=settings.llm_base_url,
        )
    raise ValueError(
        f"Unsupported embedding provider: {settings.embedding_provider}"
    )


def build_llm(settings: Settings, model_name: str | None = None) -> OllamaLLM:
    provider = settings.llm_provider.lower()
    if provider != "ollama":
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    return OllamaLLM(
        model_name=model_name or settings.llm_model,
        host=settings.llm_base_url,
        model_params={
            "options": {
                "temperature": settings.llm_temperature,
                "num_ctx": settings.llm_num_ctx,
            }
        },
    )
