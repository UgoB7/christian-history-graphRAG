from __future__ import annotations

import json
from typing import Optional

import typer

from christian_history_graphrag.config import load_settings
from christian_history_graphrag.ingest import build_records, persist_records
from christian_history_graphrag.kg_builder import run_kg_builder_enrichment
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.rag import ask_hybrid_question, ask_question, embed_passages

app = typer.Typer(help="Christian history GraphRAG starter.")


@app.command()
def ingest(
    seed_qid: list[str] = typer.Option(
        ..., "--seed-qid", help="Wikidata QIDs to use as graph seeds."
    ),
    depth: int = typer.Option(1, min=0, help="How many hops to expand in Wikidata."),
    wikipedia: bool = typer.Option(
        True, "--wikipedia/--no-wikipedia", help="Fetch Wikipedia passages."
    ),
    reset_db: bool = typer.Option(
        False, "--reset-db", help="Delete existing graph data before ingesting."
    ),
) -> None:
    settings = load_settings()
    store = Neo4jStore(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        if reset_db:
            store.reset_database()
        records = build_records(
            seed_qids=seed_qid,
            settings=settings,
            max_depth=depth,
            fetch_wikipedia=wikipedia,
        )
        persist_records(store, records.values())
        typer.echo(f"Ingested {len(records)} entities into Neo4j.")
    finally:
        store.close()


@app.command()
def embed(
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Delete all stored embeddings and rebuild them."
    ),
) -> None:
    settings = load_settings()
    store = Neo4jStore(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        embed_passages(store, settings, rebuild=rebuild)
        typer.echo("Passage embeddings created and stored in Neo4j.")
    finally:
        store.close()


@app.command("reset-db")
def reset_db() -> None:
    settings = load_settings()
    store = Neo4jStore(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        store.reset_database()
        typer.echo("Neo4j graph data and vector index removed.")
    finally:
        store.close()


@app.command("kg-enrich")
def kg_enrich(
    qid: Optional[list[str]] = typer.Option(
        None,
        "--qid",
        help="Restrict KG Builder enrichment to existing Entity wikidata IDs.",
    ),
    limit: int = typer.Option(25, help="Maximum number of entities to enrich."),
    year_from: Optional[int] = typer.Option(None, help="Lower time bound."),
    year_to: Optional[int] = typer.Option(None, help="Upper time bound."),
    replace_existing: bool = typer.Option(
        True,
        "--replace-existing/--keep-existing",
        help="Delete the previous KG Builder subgraph for each entity before rebuilding.",
    ),
) -> None:
    settings = load_settings()
    store = Neo4jStore(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        enriched = run_kg_builder_enrichment(
            store=store,
            settings=settings,
            qids=qid or None,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            replace_existing=replace_existing,
        )
        typer.echo(f"KG Builder enriched {enriched} entities.")
    finally:
        store.close()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask over the graph."),
    top_k: int = typer.Option(5, help="How many passages to retrieve."),
    year_from: Optional[int] = typer.Option(None, help="Lower time bound."),
    year_to: Optional[int] = typer.Option(None, help="Upper time bound."),
    show_context: bool = typer.Option(
        False, "--show-context", help="Display the retrieved graph context and sources."
    ),
) -> None:
    settings = load_settings()
    store = Neo4jStore(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        result = ask_question(
            store=store,
            settings=settings,
            question=question,
            top_k=top_k,
            year_from=year_from,
            year_to=year_to,
            return_context=show_context,
        )
        typer.echo(result.answer)
        if show_context and result.retriever_result:
            typer.echo("\n=== Context Used ===")
            for index, item in enumerate(result.retriever_result.items, start=1):
                typer.echo(f"\n[{index}]")
                typer.echo(str(item.content))
    finally:
        store.close()


@app.command("ask-hybrid")
def ask_hybrid(
    question: str = typer.Argument(..., help="Question to ask over the hybrid graph."),
    top_k: int = typer.Option(5, help="How many KG chunks to retrieve."),
    year_from: Optional[int] = typer.Option(None, help="Lower time bound."),
    year_to: Optional[int] = typer.Option(None, help="Upper time bound."),
    show_context: bool = typer.Option(
        False, "--show-context", help="Display the retrieved graph context and sources."
    ),
) -> None:
    settings = load_settings()
    store = Neo4jStore(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        result = ask_hybrid_question(
            store=store,
            settings=settings,
            question=question,
            top_k=top_k,
            year_from=year_from,
            year_to=year_to,
            return_context=show_context,
        )
        typer.echo(result.answer)
        if show_context and result.retriever_result:
            typer.echo("\n=== Context Used ===")
            for index, item in enumerate(result.retriever_result.items, start=1):
                typer.echo(f"\n[{index}]")
                typer.echo(str(item.content))
    finally:
        store.close()


@app.command("subgraph")
def subgraph(
    year_from: Optional[int] = typer.Option(None, help="Lower time bound."),
    year_to: Optional[int] = typer.Option(None, help="Upper time bound."),
    limit: int = typer.Option(50, help="Maximum number of rows."),
) -> None:
    settings = load_settings()
    store = Neo4jStore(
        settings.neo4j_uri,
        settings.neo4j_username,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        result = store.get_period_subgraph(
            year_from=year_from,
            year_to=year_to,
            limit=limit,
        )
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        store.close()


if __name__ == "__main__":
    app()
