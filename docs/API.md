# RepoRAG Local API

The FastAPI layer exposes RepoRAG's existing single-question repository Q&A service to a future local frontend. It validates HTTP input, loads compatible prebuilt runtime artifacts, calls the existing Q&A service, and serializes its result. It does not implement retrieval, generation, ingestion, or index building.

## Local use

The API is intended for local development. Start it from the repository root on loopback only:

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m uvicorn reporag.api.app:app --reload --host 127.0.0.1 --port 8000
```

Only `http://localhost:3000` and `http://127.0.0.1:3000` are allowed as browser CORS origins.

## Health

`GET /health`

```json
{"status":"ok","service":"RepoRAG"}
```

## Repository Q&A

`POST /api/qa` requires an existing local repository and compatible indexes under `data/embeddings/<repository>/` and `data/bm25/<repository>/`. There is no HTTP indexing endpoint in Phase 1.

```json
{
  "repository": "D:\\Masters-Projects\\RealTimeCollab",
  "question": "Where is JWT authentication enforced?",
  "retrieval_method": "hybrid",
  "evidence_scope": "all",
  "provider": "deepseek",
  "context_k": 5,
  "max_output_tokens": 2000
}
```

Optional fields use the existing Q&A defaults. Retrieval methods are `vector`, `bm25`, `hybrid`, and `reranked`; evidence scopes are `all`, `code`, and `docs`; providers are `deepseek`, `gemini`, and `openai`.

Successful responses contain the repository name, exact question and answer, selected method/scope/provider/model, citation status, cited source IDs, provider usage and latency when returned, and source blocks populated directly from existing Q&A evidence metadata. Source paths remain repository-relative.

```json
{
  "question": "Where is JWT authentication enforced?",
  "repository": "RealTimeCollab",
  "retrieval_method": "hybrid",
  "evidence_scope": "all",
  "provider": "deepseek",
  "model": "deepseek-flash",
  "answer": "JWT authentication is enforced by the authentication middleware [S1].",
  "citations_valid": true,
  "cited_source_ids": ["S1"],
  "generation_latency_seconds": 1.25,
  "usage": {
    "input_tokens": 100,
    "output_tokens": 20,
    "reasoning_tokens": 5,
    "total_tokens": 120
  },
  "sources": [
    {
      "id": "S1",
      "file_path": "src/auth.ts",
      "symbol": "authenticate",
      "parent_symbol": null,
      "start_line": 12,
      "end_line": 24,
      "language": "typescript",
      "chunk_type": "function",
      "retrieval_rank": 1,
      "content": "export function authenticate(token: string) { ... }"
    }
  ]
}
```

Provider usage and generation latency are `null` when the provider does not return them. Individual token fields are included only when supplied by the provider SDK.

## Errors

Validation failures return HTTP 422. Missing repositories or scoped evidence return 404. Missing, stale, or incompatible indexes return 409. Missing provider configuration returns 503, provider failures return 502, and a configured generation output limit returns 422 with retry guidance. Error responses contain only a safe `detail` and stable `code`; credentials, provider exception bodies, headers, and stack traces are not returned.
