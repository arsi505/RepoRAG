# RepoRAG

Working title:  
RepoRAG: An Empirical Evaluation of Retrieval Strategies for Repository-Level Software Engineering Tasks

RepoRAG is a repository-aware retrieval-augmented generation research project for empirically evaluating retrieval strategies on repository-level software-engineering questions. Retrieval evaluation—not generic chatbot development—is the primary contribution.

## Research Questions

**Primary research question:** How do different retrieval strategies affect retrieval effectiveness for repository-level software-engineering questions?

**Secondary research question:** What retrieval-quality versus latency trade-offs exist between the four retrieval strategies?

## Retrieval Methods

1. Vector retrieval
2. BM25 retrieval
3. Hybrid retrieval
4. Hybrid retrieval + reranking

## Evaluation Metrics

**Primary metric:** Recall@5

**Supporting metrics:** Recall@1, Recall@3, Hit@k, MRR, and latency

## Status

**Day 5 — Vector and BM25 baselines implemented**

The research design and Day 3 chunk dataset remain frozen. Method A provides exact cosine vector search and Method B provides deterministic BM25 lexical search over the same chunks. Hybrid retrieval, reranking, LLM answer generation, and evaluation remain future work.

## Day 2 / Development

RepoRAG ingestion accepts an existing local directory or a public GitHub repository URL. It records file and Git snapshot metadata in `data/manifests/` without storing source contents.

From PowerShell at the repository root, expose the `src` package and run the CLI:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m reporag.ingestion.cli "D:\path\to\repository"
python -m reporag.ingestion.cli "https://github.com/owner/repository.git"
```

Run the dependency-free automated tests with:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

See [Repository Ingestion](docs/INGESTION.md) for filtering rules, metadata, reproducibility behavior, and current limitations.

## Day 3 / Development

Install the project and its pinned Tree-sitter dependency from `pyproject.toml`:

```powershell
python -m pip install -e .
```

With the established source-layout workflow, generate chunks for a local or public GitHub repository using:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m reporag.chunking.cli "D:\path\to\repository"
```

The first use of a structural language may download its grammar into the ignored `data/tree_sitter_cache/` directory. Runtime chunks are written to `data/chunks/<repository>.jsonl`.

See [Code-Aware Chunking](docs/CHUNKING.md) for supported languages, metadata, fallback behavior, stable IDs, and limitations.

## Day 4 / Vector Retrieval

Install the pinned dependencies into an isolated environment, then build and search an index:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python -m reporag.embeddings.cli build "D:\path\to\repository"
.venv\Scripts\python -m reporag.retrieval.cli data\embeddings\repository "Where is realtime connection handling implemented?" --top-k 5
```

Add `--show-content` to the search command to print exact chunk source. Generated model and index artifacts remain local and gitignored. See [Vector Retrieval](docs/VECTOR_RETRIEVAL.md) for the frozen model revision, input template, normalization, persistence, stale-index validation, and limitations.

The model was trained at a sequence length of 512 and uses its documented ALiBi-based support for an explicit 8,192-token inference maximum. Index metadata reports pre-truncation token lengths and any documents exceeding that limit.

## Day 5 / BM25 Retrieval

Build and search the independent lexical index:

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m reporag.lexical.cli build "D:\path\to\repository"
.venv\Scripts\python.exe -m reporag.lexical.cli search data\bm25\repository "Where is authentication enforced?" --top-k 5
```

Add `--show-content` to print exact source. The frozen BM25 configuration uses `k1=1.5`, `b=0.75`, code-aware identifier/path tokenization, exhaustive scoring, and positive-score-only results. See [BM25 Lexical Retrieval](docs/BM25_RETRIEVAL.md).
