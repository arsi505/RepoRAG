# Vector Retrieval

Day 4 implements Method A, the vector-only baseline. It embeds the frozen Day 3 chunks and ranks every chunk by exact cosine similarity to a natural-language query. It does not generate answers or implement lexical, hybrid, reranked, or approximate retrieval.

## Frozen model

RepoRAG uses `jinaai/jina-embeddings-v2-base-code`, a code-focused embedding model that maps code and natural-language descriptions into the same 768-dimensional space. The immutable Hugging Face revision is:

```text
516f4baf13dec4ddddda8631e019b5737c8bc250
```

The model license is Apache-2.0. Loading uses Sentence Transformers 5.7.0 with `trust_remote_code=True`, as required by the model card. The model revision is always supplied to the loader. Its external JinaBert trusted-code dependency is separately pinned to revision `3baf9e3ac750e76e8edd3019170176884695fb94`. Transformers is pinned to the compatible 4.57.6 release because Transformers 5 removed an API imported by that custom implementation. The model card defines no separate query/document instruction prefixes; documents use the template below and queries are passed unchanged.

The model was trained using sequence length 512 and supports sequence lengths up to 8192 using its ALiBi-based architecture. RepoRAG explicitly configures both document and query inference to the model-supported maximum of 8,192 tokens. This avoids unnecessarily discarding content from the frozen Day 3 chunks; it does not imply that the model was trained at 8,192 tokens.

Before document encoding, RepoRAG counts tokenizer input IDs with truncation disabled. Index metadata records the maximum observed document length, the number above the configured limit, the number deterministically truncated, and the affected chunk IDs. The frozen RealTimeCollab corpus has no document above 8,192 tokens, so its rebuilt index records zero truncations.

## Frozen embedding text

Formatter version `reporag-embedding-text-v1` uses exactly this template for every chunk:

```text
Repository: {repository}
File: {repository-relative file_path}
Language: {language}
Type: {chunk_type}
Symbol: {symbol_name or empty string}
Parent: {parent_symbol or empty string}
Code:
{exact chunk content}
```

It deliberately excludes commit hashes, timestamps, absolute paths, and generated instructions. The exact Day 3 chunk content is not altered.

## Normalization and ranking

Document and query embeddings are L2-normalized. Cosine similarity is therefore the matrix-vector dot product. Search exhaustively scores every indexed chunk with NumPy, sorts by similarity descending, and returns configurable top-k results. Version 1 uses exact search because current research repositories fit in memory and a transparent baseline is preferable to approximate-nearest-neighbor tuning.

Scores equal after rounding to 12 decimal places are treated as effectively tied and ordered by `chunk_id` ascending. This makes ranking deterministic for the same index and query.

## Persistence and consistency

Each index is stored under `data/embeddings/<repository>/`:

```text
embeddings.npy  normalized float32 matrix; row N belongs to chunk N
metadata.json   snapshot, model/length configuration, token audit, ordered IDs, and fingerprint
chunks.jsonl    ordered Day 3 chunk records used by search results
```

NumPy arrays are saved without pickle. Loading validates repository, commit, chunk count and order, source fingerprint, model name and revision, dimension, normalization policy, format version, matrix shape, finite values, and row norms. An expected current snapshot/configuration can also be supplied; mismatches fail with a regeneration message rather than silently using a stale index.

The deterministic source-input fingerprint is SHA-256 over a header containing format version, model identity/revisions, maximum sequence length, and normalization policy, followed by each ordered `(chunk_id, content_hash)` pair. This proves which frozen source input and embedding configuration an index represents. It does not claim that floating-point model output is bit-identical across all CPU, operating-system, and library environments.

## Documentation as repository content

Markdown and other repository documentation remain in the frozen searchable corpus as realistic content and potential distractors. For future implementation-oriented benchmark questions, README or documentation chunks must not automatically count as ground-truth implementation evidence when the requested behavior is actually implemented in source, test, or configuration files. Day 4 does not implement benchmark or ground-truth logic.

## Cache behavior

Model files are downloaded into ignored `data/model_cache/`; generated indexes are in ignored `data/embeddings/`. Index building may download a missing pinned snapshot, while search loads the completed cache in strict offline mode. Only `data/embeddings/.gitkeep` is tracked. Model binaries and generated index files are runtime artifacts, while future final evaluation CSV/JSON evidence remains trackable.

## Commands

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python -m reporag.embeddings.cli build "D:\path\to\repository"
.venv\Scripts\python -m reporag.retrieval.cli data\embeddings\repository "Where is authentication enforced?" --top-k 5
.venv\Scripts\python -m reporag.retrieval.cli data\embeddings\repository "Where is authentication enforced?" --top-k 5 --show-content
```

The build command re-runs ingestion and deterministic Day 3 chunking. If an existing frozen JSONL dataset differs from the current repository snapshot, it stops instead of re-embedding changed chunks under the old dataset identity.

## Current limitations

- The entire float32 matrix is loaded into memory and scored for every query.
- First use requires downloading the pinned model and trusted model code.
- Inputs beyond the explicit 8,192-token model limit are deterministically truncated and reported in index metadata.
- CPU inference is substantially slower than a GPU, especially during index construction.
- Search returns source chunks only; it does not synthesize or judge answers.
