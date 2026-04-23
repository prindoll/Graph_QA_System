# GraphRAG

LangChain + Neo4j GraphRAG service for indexing PDFs into a Microsoft GraphRAG-style knowledge model and answering questions with graph-aware retrieval.

## Architecture

The current pipeline stores everything in Neo4j:

```text
PDF / documents
  -> Document
  -> TextUnit chunks
  -> Entity + Relationship extraction
  -> Community detection
  -> CommunityReport generation
  -> TextUnit / Entity / CommunityReport embeddings
  -> Neo4j graph + Neo4j vector indexes
```

Main retrieval modes:

- `basic`: vector search over `TextUnit` nodes.
- `local`: entity vector search, multi-hop graph expansion, mapped text units and community reports.
- `global`: map-reduce style search over `CommunityReport` nodes.
- `drift`: global primer plus local follow-up searches.
- `auto`: routes each query to the best mode.

## Requirements

- Python 3.10+
- Neo4j 5.x
- OpenAI-compatible chat model credentials for extraction, report generation, routing, and answering
- Local MiniLM embeddings by default: `sentence-transformers/all-MiniLM-L6-v2`

Start Neo4j:

```powershell
docker compose up -d neo4j
```

Neo4j Browser: `http://localhost:7474`

Default credentials from `docker-compose.yaml`:

```text
user: neo4j
password: password
bolt: bolt://localhost:7687
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env`:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

GRAPH_DB_TYPE=neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

RETRIEVAL_MODE_DEFAULT=auto
MAX_HOPS=2
COMMUNITY_ALGORITHM=leiden
DEBUG=False
```

If `python-igraph` or `leidenalg` is unavailable, indexing falls back to NetworkX Louvain community detection.

## CLI Usage

Index a PDF and clear old data first:

```powershell
python main.py index uploads\document.pdf --clear
```

Index a page range:

```powershell
python main.py extract-pages uploads\document.pdf 0 10 --clear
```

Query with auto routing:

```powershell
python main.py query "What are the main themes in this document?" --mode auto --top-k 5 --max-hops 2
```

Use a specific retrieval mode:

```powershell
python main.py query "How is Alpha related to Beta?" --mode drift --max-hops 3
```

Show stats:

```powershell
python main.py stats
```

Clear graph data:

```powershell
python main.py clear
```

## API Usage

Run the FastAPI server:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open the UI:

```text
http://localhost:8000/static/index.html
```

Chat request shape:

```json
{
  "message": "Summarize the core ideas",
  "top_k": 5,
  "use_graph": true,
  "response_style": "detailed",
  "retrieval_mode": "auto",
  "max_hops": 2,
  "include_sources": true
}
```

Upload endpoint:

```text
POST /api/upload
form-data:
  file=<pdf>
  start_page=0
  end_page=-1
  clear_existing=true
```

Upload response includes counts for `documents`, `text_units`, `entities`, `relationships`, `communities`, `community_reports`, persisted nodes/edges, and vector index setup.

## Testing

Run unit tests:

```powershell
python -m pytest -q
```

Compile check:

```powershell
python -m compileall config src api.py main.py tests
```

Current tests cover stable IDs, text-unit splitting, graph normalization, routing, and duplicate-safe Neo4j `MERGE` writes. Live Neo4j retrieval quality still requires an indexed corpus.

## Quality Evaluation

Recommended retrieval metrics:

- `Recall@5`
- `MRR@10`
- `nDCG@10`
- entity/path hit rate for multi-hop questions

Recommended QA metrics:

- exact match / token F1 for benchmark datasets
- groundedness against retrieved sources
- citation precision
- hallucination rate

Before evaluating, ensure Neo4j is running and data has been reindexed with the new schema:

```powershell
docker compose up -d neo4j
python main.py index uploads\document.pdf --clear
python main.py stats
```

## Notes

- The new schema is not migrated from the previous graph format. Use `--clear` and reindex.
- All graph writes use `MERGE` to reduce duplicate nodes and relationships.
- Neo4j vector indexes are created for `TextUnit.text_embedding`, `Entity.description_embedding`, and `CommunityReport.content_embedding`.
- The default embedding model is local and cheap to run, but higher-quality embedding providers may improve semantic retrieval.
