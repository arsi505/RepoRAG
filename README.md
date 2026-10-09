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

**Day 3 — Code-aware parsing and chunking implemented**

The research design remains frozen. Repository ingestion and deterministic code-aware chunk generation are implemented; retrieval implementation is not included.

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
