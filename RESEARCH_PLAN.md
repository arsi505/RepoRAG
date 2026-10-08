# RepoRAG — Research Plan

**Project:** RepoRAG — Repository-Aware RAG for Software Engineering  
**Status:** Frozen Research Scope — Version 1  
**Date:** October 2026

## 1. Project Goal

RepoRAG is a repository-aware retrieval-augmented generation system designed for software-engineering questions about source-code repositories.

The primary research goal is not to build another generic code chatbot.

The primary goal is to empirically evaluate how different retrieval strategies perform when answering repository-level software-engineering questions.

The system will later include a minimal repository Q&A interface, but retrieval evaluation remains the main research contribution.

---

## 2. Primary Research Question

**RQ1: How do different retrieval strategies affect retrieval effectiveness for repository-level software-engineering questions?**

The following retrieval strategies will be compared:

A. Vector-only retrieval  
B. BM25-only retrieval  
C. Hybrid retrieval  
D. Hybrid retrieval + reranking

---

## 3. Secondary Research Question

**RQ2: What retrieval-quality versus latency trade-offs exist between the four retrieval strategies?**

This allows the study to evaluate not only which method retrieves more relevant repository evidence, but also what additional computational cost is required.

---

## 4. Research Position

This study is exploratory.

The experiment does not assume that Hybrid + Reranking will necessarily be the best method.

All four approaches will be implemented and evaluated using the same repositories, questions, chunks, evaluation metrics, and experimental conditions.

Results will be reported even if they do not match initial expectations.

---

## 5. Retrieval Methods

### Method A — Vector Retrieval

Repository chunks will be converted into embeddings.

The user question will also be converted into an embedding.

Similarity between the question and repository chunks will be calculated using vector similarity.

The most similar chunks will be returned.

---

### Method B — BM25 Retrieval

BM25 will retrieve repository chunks using lexical or keyword similarity.

This method does not depend on semantic embeddings.

It provides a lexical-search baseline against which semantic retrieval can be compared.

---

### Method C — Hybrid Retrieval

Vector retrieval and BM25 retrieval will both generate ranked candidate lists.

Their rankings will be combined using a fixed rank-fusion method.

The initial implementation will use Reciprocal Rank Fusion (RRF).

The same fusion configuration must be used for every evaluation question.

---

### Method D — Hybrid Retrieval + Reranking

Hybrid retrieval will first generate candidate chunks.

A reranking model will then rescore and reorder these candidates.

Only the ranking stage will differ from Method C.

---

## 6. Repository-Aware Processing

RepoRAG must preserve repository structure rather than treating source code like ordinary text.

Chunks should contain metadata including:

- repository name
- file path
- programming language
- chunk type
- function name when available
- class name when available
- starting line
- ending line

Chunking should attempt to respect software structures such as files, functions, classes, and methods whenever technically practical.

---

## 7. Initial Evaluation Repositories

Version 1 of the study will use repositories whose architecture can be manually inspected and understood well enough to create reliable ground truth.

Initial repositories:

**Reloop**  
Reliable distributed job-processing and workflow system.

**RealTimeCollab**  
Real-time collaborative backend system.

**ArsiShell**  
C++17 command shell and systems-programming project.

Using these repositories makes manual verification of relevant files and implementation locations practical during the initial research sprint.

A later extension may evaluate additional external open-source repositories to improve external validity.

---

## 8. Repository File Policy

Relevant source-code, test, and configuration files may be indexed.

Generated or dependency files must be excluded.

Examples of excluded content include:

node_modules  
dist  
build outputs  
compiled binaries  
vendor dependencies  
minified files  
generated code  
temporary files  
Git metadata

Documentation such as README files should not be used as a substitute for retrieving actual implementation evidence during the primary source-code retrieval experiment.

---

## 9. Evaluation Dataset

Approximately **40 repository-level software-engineering questions** will be created.

Questions will be distributed across the selected repositories.

Each question will contain:

- question ID
- repository
- natural-language question
- manually verified relevant file or files
- relevant function/class when useful
- verification notes

Example questions include:

Where is authentication enforced?

Which files implement retry behavior?

Where is stale-work recovery handled?

Which component controls concurrency?

Which tests verify authorization?

Where is job failure recovery implemented?

Which file contains tenant-isolation logic?

Where is command parsing performed?

---

## 10. Ground Truth

Ground truth must be manually verified.

For every evaluation question, the relevant repository file or files must be identified before comparing retrieval methods.

Ground-truth labels must not be changed simply because a retrieval method performs poorly.

Ambiguous questions should be corrected or removed before the final benchmark.

---

## 11. Evaluation Level

Retrieval systems return chunks, but the main experiment will evaluate repository evidence primarily at the **file level**.

If several retrieved chunks come from the same file, the file will be represented by its highest-ranked chunk when calculating file-ranking metrics.

This prevents a single file containing many retrieved chunks from unfairly dominating the evaluation.

---

## 12. Primary Metrics

### Recall@1

Measures how much of the relevant repository evidence appears within the first retrieved result.

### Recall@3

Measures relevant evidence appearing within the first three results.

### Recall@5

Measures relevant evidence appearing within the first five results.

### Hit@k / Correct-File Retrieval

Measures whether at least one correct file appears within the top-k results.

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant result is ranked.

A correct file appearing at rank 1 receives a better score than one appearing at rank 5.

### Query Latency

Measures how long each retrieval method takes to return results.

Latency comparisons must be performed under approximately equivalent machine and runtime conditions.

---

## 13. Primary Evaluation Metric

**Recall@5** will be treated as the main retrieval-effectiveness metric.

MRR and Hit@k will provide additional ranking information.

Latency will be treated as a system-performance metric rather than an accuracy metric.

---

## 14. Experimental Controls

Every retrieval method must use the same:

- repositories
- repository snapshot/version
- indexed source files
- code chunks
- chunk metadata
- evaluation questions
- ground-truth labels
- k values
- machine where practical
- benchmark script

Retrieval strategies must not receive different questions or easier datasets.

The experiment must be reproducible from saved configuration files.

---

## 15. Separation of Retrieval and Generation

The primary research evaluation focuses on retrieval.

LLM answer generation will not determine the primary retrieval results.

This prevents generation quality from hiding retrieval failures.

For example:

If the retriever finds the wrong files but an LLM guesses the correct answer, the retrieval experiment should still consider the retrieval unsuccessful.

Likewise, if the retriever finds the correct evidence but the LLM produces a poor answer, the retrieval system should still receive credit for successful retrieval.

---

## 16. Repository Q&A

RepoRAG will also contain a minimal repository-question-answering capability.

The pipeline will be:

Repository  
→ File Discovery  
→ Filtering  
→ Code-Aware Chunking  
→ Metadata  
→ Indexing  
→ Retrieval  
→ Optional Reranking  
→ Relevant Context  
→ LLM  
→ Answer

This Q&A component demonstrates the practical usefulness of the retrieval system.

It is not the primary empirical measurement.

---

## 17. Out of Scope for Version 1

The following are intentionally excluded from the 14-day research sprint:

- user accounts
- authentication system
- payment system
- subscriptions
- SaaS billing
- complex frontend
- mobile application
- multi-agent architecture
- MCP integration
- fine-tuning
- custom model training
- production cloud infrastructure
- enterprise permissions
- multi-user collaboration
- many vector databases
- advanced support for every programming language

These features may be considered after the research experiment is complete.

---

## 18. Technology Direction

The research pipeline will primarily use Python because Python provides a mature ecosystem for information retrieval, embeddings, reranking, evaluation, and scientific analysis.

The initial product should work locally.

Expensive infrastructure should not be required.

A simple CLI or minimal interface is sufficient during the research phase.

---

## 19. Experimental Dataset Size

Target:

**Approximately 40 manually verified questions**

With four retrieval methods:

40 questions × 4 methods = **160 primary retrieval evaluations**

Additional Recall@k measurements and repeated latency measurements may produce a larger number of individual observations.

---

## 20. Expected Research Outputs

The completed research project should produce:

1. Working RepoRAG retrieval system
2. Repository ingestion pipeline
3. Code-aware chunking system
4. Vector retrieval baseline
5. BM25 baseline
6. Hybrid retrieval
7. Hybrid + reranking retrieval
8. Ground-truth evaluation dataset
9. Automated benchmark
10. Machine-readable benchmark results
11. Result tables
12. Result visualizations
13. Statistical summaries
14. Technical README
15. Reproducibility instructions
16. Research manuscript draft

---

## 21. Planned Manuscript

Working title:

**RepoRAG: An Empirical Evaluation of Retrieval Strategies for Repository-Level Software Engineering Tasks**

Proposed sections:

Abstract  
Introduction  
Research Questions  
Related Background  
RepoRAG System Design  
Experimental Methodology  
Repository Dataset  
Evaluation Questions and Ground Truth  
Retrieval Methods  
Evaluation Metrics  
Results  
Discussion  
Threats to Validity  
Limitations  
Conclusion  
References

---

## 22. Threats to Validity Already Recognized

The initial study uses a relatively small number of repositories.

The repository selection may not represent all software projects.

The project author is familiar with some of the evaluated repositories.

Manual ground-truth labeling can contain human judgment.

Different embedding or reranking models could produce different results.

Different programming languages may respond differently to retrieval strategies.

These limitations must be reported rather than hidden.

---

## 23. Future Extensions

After Version 1 is completed, possible follow-up work includes:

external open-source repositories  
larger evaluation datasets  
additional programming languages  
different embedding models  
different reranking models  
chunking-strategy experiments  
answer-quality evaluation  
repository graph information  
dependency-aware retrieval  
symbol-aware retrieval  
call-graph-aware retrieval  
commercial SaaS interface

These are extensions, not requirements for the initial experiment.

---

## 24. Frozen Scope Rule

Once benchmarking begins:

**Do not change repositories, ground truth, evaluation questions, retrieval definitions, or evaluation metrics because one method performs unexpectedly poorly.**

Any major methodological change requires creating a new experiment version and documenting the change.

---

## 25. Day 14 Success Condition

RepoRAG Version 1 will be considered successful when:

- all four retrieval strategies work
- the same evaluation dataset can be tested automatically
- ground truth is documented
- metrics are calculated automatically
- results can be reproduced
- differences between retrieval strategies can be analyzed
- limitations are documented
- the repository is clean enough to show a professor
- Manuscript Version 1 has been written

The goal is not to prove that RepoRAG is superior.

The goal is to conduct a controlled, reproducible, and honest empirical software-engineering experiment.
