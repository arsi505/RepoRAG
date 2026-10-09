# Hybrid Retrieval

Day 6 implements Method C: deterministic Reciprocal Rank Fusion (RRF) over the unchanged Method A Vector ranking and Method B BM25 ranking. Hybrid retrieval exists to combine semantic and lexical evidence without comparing their incompatible raw score scales. A cosine similarity such as `0.55` and a BM25 score such as `16.27` are therefore never added, normalized, or otherwise fused directly.

## Frozen method

For chunk `d`, RepoRAG computes:

```text
RRF_score(d) = sum(1 / (k + rank_i(d)))
```

The sum contains one contribution for each retriever in which the chunk occurs. Ranks are 1-based. The frozen configuration is:

- method name: `hybrid_rrf`
- RRF version: `reporag-rrf-v1`
- RRF rank-offset constant: `k = 60`
- Vector method identity: `reporag-exact-cosine-v1`
- BM25 method identity: `reporag-bm25-v1`

The RRF `k` is a rank-offset constant, not a retrieval depth, and is not tuned using sanity queries.

## Full-ranking fusion

Each original query is sent unchanged and independently to both existing retrievers. Vector ranks all indexed chunks by exact cosine similarity. BM25 ranks every positive-scoring chunk. RepoRAG fuses those complete available lists before selecting the requested final top-k; it does not introduce a top-20 or top-50 candidate cutoff.

Exact `chunk_id` is the only fusion identity. A chunk missing from one list remains eligible and receives only the other retriever's contribution. In particular, a zero-score BM25 chunk receives only its Vector contribution.

Hybrid results sort by RRF score descending. Scores equal at 12 decimal places are treated as effectively tied and sorted by `chunk_id` ascending. Raw cosine and BM25 scores are retained for inspection but are never undocumented tie-breakers.

## Compatibility and fingerprinting

Before search, RepoRAG requires both indexes to have the same repository, commit, chunk count, ordered chunk IDs, and ordered Day 3 corpus fingerprint derived from repository, commit, and each chunk's ID and content hash. It also requires the frozen Method A and Method B configurations. Incompatible indexes fail rather than being fused.

The deterministic hybrid SHA-256 fingerprint covers repository, commit, corpus fingerprint, Vector source/config fingerprint, BM25 source/config fingerprint, method name, RRF version, RRF `k`, and both method-version identities. It excludes timestamps and machine paths.

Hybrid requires no heavy new index. The CLI writes deterministic runtime metadata to `data/hybrid/<repository>/metadata.json`; that directory is gitignored, while `data/hybrid/.gitkeep` is tracked. Query-specific results are not persisted as index state.

## Usage

From the repository root:

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m reporag.retrieval.hybrid_cli `
  data\embeddings\realtimecollab `
  data\bm25\realtimecollab `
  "Where is a realtime client connection handled?" `
  --top-k 5 --verbose
```

The pinned embedding snapshot must already exist in `data/model_cache`; hybrid search does not download models or rebuild either source index.

## Current limitations

Method C performs exhaustive ranking of both local indexes and has no candidate cutoff, so latency grows with corpus size. BM25 contributes only positive-scoring results. RRF uses ranks rather than score magnitude and does not learn query-specific weights. Day 6 includes no reranker, query expansion, generated answers, evaluation labels, or research metrics.
