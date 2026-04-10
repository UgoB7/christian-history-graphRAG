# christian-history-graphRAG

Starter kit GraphRAG pour histoire chrétienne en mode 100% open source et self-hosted:

- Neo4j pour le graphe
- Wikidata pour les entités, relations, dates et coordonnées
- Wikipedia pour les passages texte
- KG Builder Neo4j pour extraire un sous-graphe sémantique depuis le texte Wikipedia
- `sentence-transformers` ou `Ollama` pour les embeddings
- `Ollama` pour le LLM de réponse
- `neo4j-graphrag` pour l’index vectoriel, le retriever et la génération RAG

## Ce qui a changé

Le projet ne dépend plus d’OpenAI ni d’API payante.

La pile par défaut est maintenant:

- LLM: `Ollama`
- Embeddings: `sentence-transformers`
- Données: Wikidata + Wikipedia + enrichissement KG Builder optionnel

Le projet repart aussi proprement de zéro pour les données et embeddings grâce à:

- `christian-history-graphrag reset-db`
- `christian-history-graphrag ingest --reset-db ...`
- `christian-history-graphrag embed --rebuild`

## Architecture

1. on choisit des `QID` Wikidata comme graines;
2. on récupère un sous-graphe structuré depuis Wikidata;
3. on extrait plusieurs paragraphes de Wikipedia;
4. on découpe ces paragraphes en chunks;
5. on écrit `:Entity` et `:Passage` dans Neo4j;
6. on peut enrichir les articles Wikipedia avec `SimpleKGPipeline` pour créer `:KgDocument`, `:KgChunk` et un graphe extrait par LLM;
7. on calcule les embeddings localement;
8. on interroge soit le graphe de base, soit le graphe hybride avec `VectorCypherRetriever` et des filtres temporels.

## Modèle de données

### Noeuds

- `(:Entity {wikidata_id, name, description, entity_kind, time_start_year, time_end_year, latitude, longitude, wikipedia_url})`
- `(:Passage {id, title, chunk_index, text, source, language, time_start_year, time_end_year, embedding})`
- `(:KgDocument {path, wikidata_id, wikipedia_title, wikipedia_url, source})`
- `(:KgChunk {text, index, embedding})`

### Relations

- `(:Entity)-[:HAS_PASSAGE]->(:Passage)`
- `(:Entity)-[:HAS_KG_DOCUMENT]->(:KgDocument)`
- `(:KgChunk)-[:KG_FROM_DOCUMENT]->(:KgDocument)`
- `(:ExtractedNode)-[:KG_FROM_CHUNK]->(:KgChunk)`
- relations Wikidata normalisées comme `:PART_OF`, `:HAS_PARTICIPANT`, `:LOCATED_IN`, `:INFLUENCED_BY`

## Installation

### 1. Lancer Neo4j

```bash
docker compose up -d
```

Neo4j Browser:

- http://localhost:7474

### 2. Environnement Python

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Installer Ollama

Le code attend un serveur Ollama sur `http://localhost:11434` par défaut.

Ensuite récupère un modèle:

```bash
ollama pull gemma4:e2b
```

### 4. Configurer le projet

```bash
cp .env.example .env
```

Variables importantes:

- `LLM_PROVIDER=ollama`
- `LLM_MODEL=gemma4:e2b`
- `KG_BUILDER_LLM_MODEL=qwen2.5:3b`
- `LLM_BASE_URL=http://localhost:11434`
- `EMBEDDING_PROVIDER=sentence-transformers`
- `EMBEDDING_MODEL=BAAI/bge-m3`
- `EMBEDDING_DEVICE=cpu` ou `cuda`

Recommandation pratique sur Mac:

- `LLM_MODEL=gemma4:e2b` pour les reponses si Ollama arrive a le charger
- `KG_BUILDER_LLM_MODEL=qwen2.5:3b` pour `kg-enrich`, plus stable et leger

## Repartir de zéro

```bash
christian-history-graphrag reset-db
```

Cette commande:

- supprime l’index vectoriel
- supprime tous les noeuds et relations Neo4j

Si tu préfères tout faire en une seule fois pendant l’ingestion:

```bash
christian-history-graphrag ingest \
  --seed-qid Q302 \
  --seed-qid Q8341 \
  --seed-qid Q43048 \
  --depth 1 \
  --reset-db
```

## Ingestion

Exemple:

```bash
christian-history-graphrag ingest \
  --seed-qid Q302 \
  --seed-qid Q8341 \
  --seed-qid Q43048 \
  --depth 1 \
  --reset-db
```

Exemples de graines:

- `Q302`: Jesus
- `Q8341`: Augustine of Hippo
- `Q43048`: First Council of Nicaea

Cette commande:

- récupère les entités Wikidata
- suit les relations utiles
- télécharge plusieurs paragraphes Wikipedia
- les découpe en plusieurs chunks
- écrit tout dans Neo4j

## Embeddings

```bash
christian-history-graphrag embed --rebuild
```

Cette commande:

- enlève les anciens embeddings
- recalcule tous les embeddings localement
- recrée l’index vectoriel Neo4j si besoin

## Enrichissement hybride avec KG Builder

Cette partie suit la logique de la doc officielle `SimpleKGPipeline`, mais en complément de Wikidata au lieu de remplacer Wikidata.

Exemple:

```bash
christian-history-graphrag kg-enrich --limit 15 --replace-existing
```

Ou pour une entité précise déjà présente dans Neo4j:

```bash
christian-history-graphrag kg-enrich --qid Q8341
```

Cette commande:

- lit des entités déjà ingérées depuis Wikidata
- recharge le texte Wikipedia complet pour ces entités
- lance `SimpleKGPipeline(from_pdf=False, text=...)`
- crée un sous-graphe lexical `KgDocument -> KgChunk`
- extrait des entités et relations supplémentaires depuis le texte
- relie le `KgDocument` à ton noeud `:Entity` existant
- crée un index vectoriel sur `:KgChunk.embedding`

## Poser une question

```bash
christian-history-graphrag ask \
  "Quels acteurs ont façonné la théologie chrétienne au IVe siècle ?" \
  --year-from 300 \
  --year-to 399 \
  --top-k 5 \
  --show-context
```

`--show-context` affiche:

- les passages exacts utilisés
- l’entité source
- le score
- les voisins du graphe
- l’URL Wikipedia utilisée

Pour exploiter le graphe hybride créé par `KG Builder`:

```bash
christian-history-graphrag ask-hybrid \
  "Quels acteurs ont façonné la théologie chrétienne au IVe siècle ?" \
  --year-from 300 \
  --year-to 399 \
  --top-k 5 \
  --show-context
```

`ask-hybrid` récupère des `KgChunk` puis traverse:

- `KgChunk -> KgDocument`
- `Entity -> KgDocument`
- noeuds et relations extraits par `KG Builder`

## Voir le graphe

Dans Neo4j Browser:

```cypher
MATCH (e:Entity)
WHERE coalesce(e.time_end_year, e.time_start_year, 999999) >= 300
  AND coalesce(e.time_start_year, e.time_end_year, -999999) <= 399
OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
RETURN e, r, neighbor
LIMIT 100
```

Ou exporter un sous-graphe:

```bash
christian-history-graphrag subgraph --year-from 300 --year-to 399 --limit 100
```

## Options open source recommandées

### Option 1. Bon défaut local

- LLM: `qwen3:8b` via Ollama
- Embeddings: `BAAI/bge-m3`
- usage: laptop costaud ou petite VM GPU

Avantages:

- simple
- très bon multilingue
- bonne qualité générale

### Option 2. Meilleur rapport qualité / coût sur Runpod

- LLM: `Qwen3-30B-A3B` ou `Qwen3-32B`
- Embeddings: `BAAI/bge-m3`
- device embeddings: `cuda`

Pourquoi:

- très bonne qualité open weights
- très bon multilingue
- adapté à un vrai usage RAG de production

### Option 3. Qualité maximale raisonnable en self-hosted

- LLM: `Llama-3.3-70B-Instruct` ou un grand modèle Qwen3
- Embeddings: `BAAI/bge-m3`
- ajout recommandé ensuite: re-ranker open source

Pourquoi:

- meilleure qualité de synthèse et de réponse finale
- très adapté si tu vas servir le modèle sur gros GPU Runpod

## Recommandation perso pour ton cas

Si ton objectif est la meilleure qualité possible sur Runpod:

1. garde ce code comme base;
2. mets `EMBEDDING_MODEL=BAAI/bge-m3` sur GPU;
3. commence avec un LLM Qwen3 moyen ou grand;
4. active le flux hybride `kg-enrich` pour extraire doctrine, influence, controverse, offices et groupes;
5. ajoute ensuite un reranker open source;
6. plus tard, fine-tune l’embedding model ou le reranker sur tes données d’histoire chrétienne.

Le plus gros gain qualité pour le RAG ne viendra pas seulement du LLM:

- il viendra beaucoup de la qualité des embeddings
- du chunking
- du reranking
- et d’un bon corpus historique ciblé

## Fichiers importants

- [src/christian_history_graphrag/cli.py](/Users/bellanca/Documents/christian-history-graphRAG/src/christian_history_graphrag/cli.py)
- [src/christian_history_graphrag/rag.py](/Users/bellanca/Documents/christian-history-graphRAG/src/christian_history_graphrag/rag.py)
- [src/christian_history_graphrag/providers.py](/Users/bellanca/Documents/christian-history-graphRAG/src/christian_history_graphrag/providers.py)
- [src/christian_history_graphrag/local_embeddings.py](/Users/bellanca/Documents/christian-history-graphRAG/src/christian_history_graphrag/local_embeddings.py)
- [src/christian_history_graphrag/kg_builder.py](/Users/bellanca/Documents/christian-history-graphRAG/src/christian_history_graphrag/kg_builder.py)
- [src/christian_history_graphrag/wikidata.py](/Users/bellanca/Documents/christian-history-graphRAG/src/christian_history_graphrag/wikidata.py)
- [src/christian_history_graphrag/wikipedia.py](/Users/bellanca/Documents/christian-history-graphRAG/src/christian_history_graphrag/wikipedia.py)

## Limites actuelles

- le graphe principal vient de Wikidata, pas d’extraction sémantique complète depuis tout le texte
- le LLM de réponse passe aujourd’hui par Ollama uniquement
- on n’a pas encore de reranker dédié
- on n’a pas encore de front carte + timeline

## Suite logique

Les meilleurs upgrades possibles ensuite:

1. ajouter un reranker open source
2. servir un plus gros LLM sur Runpod
3. fine-tuner les embeddings sur ton corpus
4. ajouter un front web avec graphe + carte + timeline
