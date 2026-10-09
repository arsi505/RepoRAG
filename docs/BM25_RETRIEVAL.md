# BM25 Lexical Retrieval

Day 5 implements Method B, the BM25-only lexical baseline. It searches exactly the same frozen Day 3 chunks as vector retrieval, without embeddings, query rewriting, stemming, hybrid fusion, reranking, or answer generation.

## Frozen configuration

- Algorithm: exhaustive BM25
- `k1 = 1.5`, controlling term-frequency saturation
- `b = 0.75`, controlling document-length normalization
- Tokenizer: `reporag-code-tokenizer-v1`
- Document formatter: `reporag-bm25-document-v1`

These values are not tuned against sanity or benchmark queries.

## Formula

For query tokens `Q` and document `D`, RepoRAG computes:

```text
score(D, Q) = sum over q in Q:
  IDF(q) * TF(q,D) * (k1 + 1)
  / (TF(q,D) + k1 * (1 - b + b * |D| / avgdl))
```

The inverse document frequency is the always-positive Robertson-style variant:

```text
IDF(q) = ln(1 + (N - DF(q) + 0.5) / (DF(q) + 0.5))
```

`N` is the number of documents and `DF(q)` is the number containing the term. The added `1` keeps terms occurring in many or all documents finite and positive. Terms absent from the corpus contribute zero. Scores are checked for NaN and infinity.

## Code-aware tokenization

Tokenization is deterministic and case-insensitive. It retains the lowercase complete identifier and adds components for camelCase, PascalCase, snake_case, kebab-case, acronym boundaries, and numeric components. Thus `handleConnection` yields `handleconnection`, `handle`, and `connection`; `JWTAuthGuard` yields `jwtauthguard`, `jwt`, `auth`, and `guard`. Paths are separated at slashes, dots, and punctuation, while hyphenated filename identifiers retain their exact form and components.

No English stopword list, stemming library, model, or network service is used. Only empty and punctuation-only spans disappear.

## Document representation

Formatter version `reporag-bm25-document-v1` uses:

```text
File: {repository-relative path}
Language: {language}
Type: {chunk type}
Symbol: {symbol name or empty}
Parent: {parent symbol or empty}
Code:
{exact chunk content}
```

It excludes absolute paths, timestamps, random data, and commit hashes. Markdown/docs remain realistic searchable repository content and potential distractors; future implementation-ground-truth policy remains separate from retrieval.

## Persistence and validation

Generated indexes live under `data/bm25/<repository>/`:

```text
metadata.json  snapshot, versions, parameters, fingerprint, corpus statistics
chunks.jsonl   ordered frozen chunk records
corpus.jsonl   ordered chunk IDs and lexical token arrays
```

No pickle is used. Loading validates repository, commit, counts, ordered IDs, fingerprint, tokenizer and formatter versions, `k1`, `b`, token alignment, and recomputed corpus statistics.

The SHA-256 source/config fingerprint covers repository, commit, tokenizer version, formatter version, `k1`, `b`, and every ordered `(chunk_id, content_hash)` pair.

## Ranking behavior

Every document is scored exhaustively. Results sort by raw BM25 score descending. Scores equal at 12 decimal places use `chunk_id` ascending. Interactive search returns only positive-scoring results; it does not fill top-k with unrelated zero-score chunks.

## Commands

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m reporag.lexical.cli build "D:\path\to\repository"
.venv\Scripts\python.exe -m reporag.lexical.cli search data\bm25\repository "handleConnection" --top-k 5
.venv\Scripts\python.exe -m reporag.lexical.cli search data\bm25\repository "handleConnection" --top-k 5 --show-content
```

## Limitations

- There is no stemming, lemmatization, synonym expansion, or typo correction.
- Query term frequency is preserved, so repeated query tokens contribute repeatedly.
- The implementation scans all documents and recomputes document term counts per query.
- Raw BM25 and cosine scores use different scales and must not be compared as normalized values.
- Documentation may outrank source when it shares stronger lexical evidence.
