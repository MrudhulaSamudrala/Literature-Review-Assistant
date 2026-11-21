# Smart Literature Review Assistant

Web application for comparing three retrieval-augmented generation architectures on scientific literature:

1. **VectorRAG** — semantic chunk retrieval over a Neo4j vector index
2. **GraphRAG** — property-graph retrieval over Neo4j
3. **HybridRAG** — fused vector and graph retrieval

The design follows the methodology in *Enhancing Scientific Literature Chatbots with Retrieval-Augmented Generation: A Performance Evaluation of Vector and Graph-Based Systems*.

This repository currently contains **project setup only**. RAG indexing, retrieval, and generation are stubbed and will be implemented next.

## Technology stack

| Layer | Choice |
| --- | --- |
| Frontend | React + Vite |
| Backend | Python, FastAPI, Pydantic, python-dotenv |
| Indexing / retrieval | LlamaIndex (not LangChain) |
| Graph / vector store | Neo4j |
| Embeddings | OpenAI `text-embedding-ada-002` |
| LLM | OpenAI `gpt-4o-mini` |
| Chunking | Semantic chunking (to be implemented with LlamaIndex) |

## Folder structure

```text
backend/
    app/
        main.py
        config.py
        api/
        core/
        models/
        services/
        rag/
            vector_rag/
            graph_rag/
            hybrid_rag/
        ingestion/
        retrieval/
        evaluation/
        utils/
    data/
        papers/
        processed/
        chunks/
    tests/
    requirements.txt
    .env.example
frontend/
    src/
        components/
        pages/
        services/
        types/
        App.jsx
        main.jsx
```

### What each major folder is for

- **`backend/app/api`** — FastAPI routers: health, PDF upload, and query.
- **`backend/app/core`** — logging and shared exceptions.
- **`backend/app/models`** — Pydantic schemas for requests and responses.
- **`backend/app/services`** — orchestration, including routing a question to the selected RAG pipeline.
- **`backend/app/rag/vector_rag`** — VectorRAG pipeline module (stub).
- **`backend/app/rag/graph_rag`** — GraphRAG pipeline module (stub).
- **`backend/app/rag/hybrid_rag`** — HybridRAG pipeline module (stub).
- **`backend/app/ingestion`** — PDF page extraction and LlamaIndex semantic chunking. Vector and graph indexing are not implemented yet.
- **`backend/app/retrieval`** — LlamaIndex / Neo4j retriever adapters (not implemented yet).
- **`backend/app/evaluation`** — comparison metrics for the three architectures (not implemented yet).
- **`backend/data/papers`** — uploaded source PDFs.
- **`backend/data/processed`** — extracted document text after ingestion.
- **`backend/data/chunks`** — semantic chunks used for indexing.
- **`frontend/src/components`** — PDF upload, mode selector, question form, answer, citations.
- **`frontend/src/pages`** — page-level composition.
- **`frontend/src/services`** — HTTP client for the FastAPI backend.
- **`frontend/src/types`** — RAG mode definitions shared by the UI.

## Environment variables

Copy `backend/.env.example` to `backend/.env` and fill in secrets. Keys are never hard-coded.

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_DATABASE=neo4j
```

Configuration is loaded by `backend/app/config.py` (`Settings` / `get_settings()`).

The API will start without OpenAI or Neo4j credentials. Semantic chunking uses LlamaIndex `SemanticSplitterNodeParser` with `buffer_size=1` and `breakpoint_percentile_threshold=95`. When `OPENAI_API_KEY` is set, breakpoint detection calls the configured `OPENAI_EMBEDDING_MODEL` (`text-embedding-ada-002`) and does not store those breakpoint vectors. When the key is empty, ingestion uses a clearly logged development fallback whose chunk boundaries are not equivalent to that setup. `EVALUATION_MODE=true` refuses the fallback. VectorRAG indexing and retrieval use Neo4j. `EMBEDDING_PROVIDER=local` embeds with the local Sentence Transformers model and does not call OpenAI embeddings. `EMBEDDING_PROVIDER=openai` keeps `text-embedding-ada-002`. `LLM_PROVIDER=openai` still uses `OPENAI_LLM_MODEL=gpt-4o-mini`. `LLM_PROVIDER=local` uses `LOCAL_LLM_MODEL` when that model is already available, and otherwise returns a clearly marked development fallback built from retrieved evidence. That fallback is not equivalent to `gpt-4o-mini`. The ingestion fallback is not used for VectorRAG. GraphRAG and HybridRAG are not implemented.

## Run the backend

From the repository root, in PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)  
OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Run the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The Vite app is served at [http://127.0.0.1:5173](http://127.0.0.1:5173) and proxies `/api` to the FastAPI backend.
