# Cross-Encoder Reranking

Day 7 implements Method D, a genuine two-stage retrieval pipeline. Frozen Method C first performs full-ranking Vector/BM25 Reciprocal Rank Fusion and returns its top 50 candidates. A cross-encoder then independently scores each `(original query, formatted candidate document)` pair and reorders only that fixed candidate set.

## Frozen configuration

- Method: `hybrid_reranked`
- Method version: `reporag-hybrid-reranker-v1`
- Candidate generator: `hybrid_rrf`
- Candidate depth: `candidate_k = 50`
- Model: `BAAI/bge-reranker-v2-m3`
- Immutable model revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- License: Apache-2.0
- Architecture-supported sequence length: 8,192 usable tokens
- Configured reranking sequence length: `max_length = 1024`
- Document formatter: `reporag-reranker-document-v1`
- Batch size: 4

The [official model metadata](https://huggingface.co/BAAI/bge-reranker-v2-m3) declares Apache-2.0. The architecture has 8,194 positional embeddings, corresponding to 8,192 usable tokens after special tokens. The model maintainers recommend `max_length=1024` because that is the reranker fine-tuning length, so Method D freezes 1,024 rather than silently using a library default.

## Candidate generation and scoring

Method D calls Method C with `top_k=50`; it does not reproduce or alter Vector, BM25, or RRF. Method C still ranks every Vector chunk and every positive-scoring BM25 chunk, performs full RRF fusion, and only then returns the Hybrid top 50. The exact original query is used at every stage without rewriting, expansion, routing, or generated text.

The cross-encoder sees one query/document pair per candidate. It returns a native raw relevance logit. Method D sorts only by that raw logit; it does not add, normalize, or weight the RRF score. A sigmoid could monotonically map logits into `[0, 1]`, but Method D does not need or store that transformation.

Scores equal at 12 decimal places are treated as effectively tied. Ties use original Hybrid rank ascending, then `chunk_id` ascending. This uses Hybrid rank only for deterministic tie resolution, not score mixing.

## Deterministic document representation

The exact Version 1 template is:

```text
File: {repository-relative file_path}
Language: {language}
Type: {chunk_type}
Symbol: {symbol_name or empty}
Parent: {parent_symbol or empty}
Code:
{exact chunk content}
```

It excludes absolute paths, timestamps, commit hashes, ranks, and Vector/BM25/RRF scores.

## Truncation and diagnostics

Pair token lengths are measured before scoring. A pair above 1,024 tokens is counted as both over the configured limit and deterministically truncated by the CrossEncoder tokenizer during inference. Source chunks are never changed. Each query reports candidate count, over-limit count, and truncation count.

Inference uses deterministic batches of four. CPU execution uses the model's safe float32 parameters; FP16 is not forced. CUDA may be used when available without changing ranking rules.

## Fingerprint and runtime metadata

The deterministic Method D SHA-256 fingerprint covers repository, commit, frozen corpus fingerprint, Hybrid fingerprint, candidate method/depth, model name and immutable revision, configured length, formatter version, Method D version, and tie policy. It excludes query text, timestamps, machine paths, and hardware.

Lightweight metadata is written to `data/reranking/<repository>/metadata.json`. Runtime metadata and the shared model cache are gitignored; model binaries and query results are not committed.

## Usage

After the pinned model snapshot has been cached, run:

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m reporag.reranking.cli `
  data\embeddings\realtimecollab `
  data\bm25\realtimecollab `
  "Where is a realtime client connection handled?" `
  --top-k 5 --verbose
```

Use `--allow-download` only for the initial pinned-revision download.

## Limitations

The reranker cannot recover a relevant chunk outside Hybrid top 50. Cross-encoder inference is substantially more expensive than first-stage retrieval, particularly on CPU. Inputs over 1,024 tokens lose tail content during deterministic truncation. Documentation remains in the frozen corpus and may move up or down. Day 7 adds no generation, answer assembly, labels, evaluation metrics, or statistical analysis.
