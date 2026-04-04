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
        llm_model=os.getenv("LLM_MODEL", "qwen3:8b"),
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
    )
