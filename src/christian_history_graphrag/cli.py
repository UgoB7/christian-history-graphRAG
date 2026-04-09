from __future__ import annotations

import json
from typing import Optional

import typer

from christian_history_graphrag.config import load_settings
from christian_history_graphrag.ingest import (
    build_records,
    persist_records,
    populate_wikipedia_passages,
)
from christian_history_graphrag.kg_builder import run_kg_builder_enrichment
from christian_history_graphrag.neo4j_store import Neo4jStore
from christian_history_graphrag.rag import (
    ask_hybrid_question,
    ask_llm_only,
    ask_question,
    embed_passages,
)

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
        typer.echo("Step 1/3: Fetching Wikidata graph...")
        records = build_records(
            seed_qids=seed_qid,
            settings=settings,
            max_depth=depth,
            fetch_wikipedia=False,
        )
        typer.echo(f"  Wikidata graph loaded: {len(records)} entities")
        if wikipedia:
            with typer.progressbar(
                length=len(records),
                label="Step 2/3: Fetching Wikipedia passages",
            ) as progress:
                populate_wikipedia_passages(
                    records=records,
                    settings=settings,
                    progress=progress.update,
                )
        total_passages = sum(len(record.passages) for record in records.values())
        total_relations = sum(max(len(record.relations), 1) for record in records.values())
        with typer.progressbar(
            length=len(records),
            label="Step 3/3: Writing entities",
        ) as entity_bar:
            with typer.progressbar(
                length=max(total_passages, len(records)),
                label="           Writing passages",
            ) as passage_bar:
                with typer.progressbar(
                    length=total_relations,
                    label="           Writing relations",
                ) as relation_bar:
                    persist_records(
                        store,
                        records.values(),
                        entity_progress=entity_bar.update,
                        passage_progress=passage_bar.update,
                        relation_progress=relation_bar.update,
                    )
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
        result = store.driver.execute_query(
            """
            MATCH (p:Passage)
            WHERE NOT 'embedding' IN keys(p) OR p.embedding IS NULL
            RETURN count(p) AS total
            """,
            database_=store.database,
        )
        total = result.records[0]["total"] if result.records else 0
        with typer.progressbar(
            length=total,
            label="Embedding passages",
        ) as progress:
            embed_passages(store, settings, rebuild=rebuild, progress=progress.update)
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
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        help="Show the current Wikipedia page, estimated chunks, and extraction failures.",
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
        candidates = store.list_entities_for_kg_enrichment(
            qids=qid or None,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
        )
        with typer.progressbar(
            length=len(candidates),
            label="KG Builder enriching entities",
        ) as progress:
            enriched = run_kg_builder_enrichment(
                store=store,
                settings=settings,
                qids=qid or None,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                replace_existing=replace_existing,
                progress=progress.update,
                reporter=typer.echo if verbose else None,
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
    compare_llm_only: bool = typer.Option(
        False,
        "--compare-llm-only",
        help="Also show an answer from the same LLM without any GraphRAG context.",
    ),
    show_retrieval_only: bool = typer.Option(
        False,
        "--show-retrieval-only",
        help="Also show the retrieved graph/text evidence without any LLM generation.",
    ),
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
        need_context = show_context or show_retrieval_only
        result = ask_hybrid_question(
            store=store,
            settings=settings,
            question=question,
            top_k=top_k,
            year_from=year_from,
            year_to=year_to,
            return_context=need_context,
        )
        typer.echo("=== Hybrid GraphRAG Answer ===")
        typer.echo(result.answer)
        if compare_llm_only:
            typer.echo("\n=== LLM Only Answer ===")
            typer.echo(ask_llm_only(settings=settings, question=question))
        if show_retrieval_only and result.retriever_result:
            typer.echo("\n=== Retrieval Only ===")
            for index, item in enumerate(result.retriever_result.items, start=1):
                typer.echo(f"\n[{index}]")
                typer.echo(str(item.content))
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
