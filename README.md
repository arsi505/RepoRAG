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

**Day 2 — Repository ingestion implemented**

The research design remains frozen. Day 2 adds repository discovery and reproducible JSON manifests only; retrieval implementation is not included.

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
