from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from christian_history_graphrag.constants import DEFAULT_WIKIPEDIA_LANGUAGE


@dataclass
class Settings:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    llm_provider: str
    llm_model: str
    kg_builder_llm_model: str
    llm_base_url: str
    llm_temperature: float
    llm_num_ctx: int
    embedding_provider: str
    embedding_model: str
    embedding_device: str
    embedding_normalize: bool
    embedding_query_prefix: str
    embedding_passage_prefix: str
    embedding_batch_size: int
    wikipedia_language: str
    wikipedia_max_paragraphs: int
    passage_chunk_size: int
    passage_paragraph_overlap: int
    kg_builder_max_paragraphs: int
    kg_builder_chunk_size: int
    kg_builder_chunk_overlap: int
    kg_builder_schema_mode: str
    cache_dir: str
    cache_ttl_seconds: int
    checkpoint_dir: str
    use_http_cache: bool
    use_ingest_checkpoints: bool
    http_max_retries: int
    http_backoff_factor: float
    log_level: str
    entity_resolution_similarity_threshold: float
    entity_resolution_candidate_limit: int
    entity_resolution_semantic_enabled: bool
    claim_extraction_llm_model: str
    claim_max_per_chunk: int
    community_report_llm_model: str
    community_report_member_limit: int
    community_report_claim_limit: int
    community_report_relation_limit: int
    router_llm_model: str
    reranker_enabled: bool
    reranker_model: str
    reranker_device: str
    reranker_batch_size: int
    reranker_candidate_pool_size: int


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "please-change-me"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "gemma4:e2b"),
        kg_builder_llm_model=os.getenv(
            "KG_BUILDER_LLM_MODEL",
            os.getenv("LLM_MODEL", "qwen2.5:3b"),
        ),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.2),
        llm_num_ctx=_get_int("LLM_NUM_CTX", 8192),
        embedding_provider=os.getenv(
            "EMBEDDING_PROVIDER", "sentence-transformers"
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        embedding_normalize=_get_bool("EMBEDDING_NORMALIZE", True),
        embedding_query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", ""),
        embedding_passage_prefix=os.getenv("EMBEDDING_PASSAGE_PREFIX", ""),
        embedding_batch_size=_get_int("EMBEDDING_BATCH_SIZE", 16),
        wikipedia_language=os.getenv(
            "WIKIPEDIA_LANGUAGE", DEFAULT_WIKIPEDIA_LANGUAGE
        ),
        wikipedia_max_paragraphs=_get_int("WIKIPEDIA_MAX_PARAGRAPHS", 18),
        passage_chunk_size=_get_int("PASSAGE_CHUNK_SIZE", 1400),
        passage_paragraph_overlap=_get_int("PASSAGE_PARAGRAPH_OVERLAP", 1),
        kg_builder_max_paragraphs=_get_int("KG_BUILDER_MAX_PARAGRAPHS", 40),
        kg_builder_chunk_size=_get_int("KG_BUILDER_CHUNK_SIZE", 1800),
        kg_builder_chunk_overlap=_get_int("KG_BUILDER_CHUNK_OVERLAP", 150),
        kg_builder_schema_mode=os.getenv("KG_BUILDER_SCHEMA_MODE", "GUIDED"),
        cache_dir=os.getenv("CACHE_DIR", ".graphrag/cache"),
        cache_ttl_seconds=_get_int("CACHE_TTL_SECONDS", 60 * 60 * 24 * 7),
        checkpoint_dir=os.getenv("CHECKPOINT_DIR", ".graphrag/checkpoints"),
        use_http_cache=_get_bool("USE_HTTP_CACHE", True),
        use_ingest_checkpoints=_get_bool("USE_INGEST_CHECKPOINTS", True),
        http_max_retries=_get_int("HTTP_MAX_RETRIES", 4),
        http_backoff_factor=_get_float("HTTP_BACKOFF_FACTOR", 0.5),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        entity_resolution_similarity_threshold=_get_float(
            "ENTITY_RESOLUTION_SIMILARITY_THRESHOLD", 0.9
        ),
        entity_resolution_candidate_limit=_get_int(
            "ENTITY_RESOLUTION_CANDIDATE_LIMIT",
            12,
        ),
        entity_resolution_semantic_enabled=_get_bool(
            "ENTITY_RESOLUTION_SEMANTIC_ENABLED",
            True,
        ),
        claim_extraction_llm_model=os.getenv(
            "CLAIM_EXTRACTION_LLM_MODEL",
            os.getenv("KG_BUILDER_LLM_MODEL", os.getenv("LLM_MODEL", "qwen2.5:3b")),
        ),
        claim_max_per_chunk=_get_int("CLAIM_MAX_PER_CHUNK", 5),
        community_report_llm_model=os.getenv(
            "COMMUNITY_REPORT_LLM_MODEL",
            os.getenv("LLM_MODEL", "gemma4:e2b"),
        ),
        community_report_member_limit=_get_int("COMMUNITY_REPORT_MEMBER_LIMIT", 20),
        community_report_claim_limit=_get_int("COMMUNITY_REPORT_CLAIM_LIMIT", 12),
        community_report_relation_limit=_get_int(
            "COMMUNITY_REPORT_RELATION_LIMIT",
            20,
        ),
        router_llm_model=os.getenv(
            "ROUTER_LLM_MODEL",
            os.getenv("LLM_MODEL", "gemma4:e2b"),
        ),
        reranker_enabled=_get_bool("RERANKER_ENABLED", True),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        reranker_device=os.getenv(
            "RERANKER_DEVICE",
            os.getenv("EMBEDDING_DEVICE", "cpu"),
        ),
        reranker_batch_size=_get_int("RERANKER_BATCH_SIZE", 8),
        reranker_candidate_pool_size=_get_int("RERANKER_CANDIDATE_POOL_SIZE", 24),
    )
