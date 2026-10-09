# Repository Question Answering

Day 8 adds a single-turn product and demonstration layer over RepoRAG's frozen retrieval methods. It retrieves repository chunks, assigns deterministic evidence IDs, sends the exact evidence to a generator, and validates source markers in the returned answer. It does not change the retrieval research methods or add an evaluation benchmark.

## Frozen Q&A configuration

- Q&A version: `reporag-qa-v1`
- Prompt version: `reporag-qa-prompt-v1`
- Context formatter: `reporag-qa-context-v1`
- Default retrieval method: `hybrid`
- Default context depth: `context_k = 5`
- Provider: `openai`
- Model snapshot: `gpt-5.4-mini-2026-03-17`
- Maximum output: 1,200 tokens
- OpenAI SDK: `openai==3.26.1`

Hybrid is the interactive default because it is substantially faster than CPU reranking and combines the existing semantic and lexical rankings. This is a practical product default, not a claim that Hybrid is scientifically superior.

Users may select `vector`, `bm25`, `hybrid`, or `reranked`. The original question is passed unchanged to exactly one retrieval call. No query rewriting, expansion, multi-query retrieval, agent loop, or conversational history is used.

## Evidence and prompting

Retrieval order is preserved. After deterministic duplicate removal, evidence receives IDs `S1`, `S2`, and so on. Each block contains the repository-relative path, language, chunk type, symbol and parent, line range, retrieval rank, and exact frozen chunk content. Absolute local paths, timestamps, retrieval scores, secrets, and model-cache details are excluded.

The provider receives one instruction template and one user input containing the exact question and source blocks. Instructions require repository-only answers, inline `[S1]` citations, cautious inference, and an explicit insufficient-evidence response rather than guessing.

After generation, RepoRAG extracts markers matching `[S1]`, `[S2]`, and so on. Repeated markers are deduplicated in first-use order. Every cited ID must exist in the supplied context. Unknown IDs are reported and make citation validation fail; an evidence-based answer with no source markers also fails validation. IDs are mapped back to path, symbol, and lines for human-readable CLI output.

If retrieval returns no evidence, generation is not called and RepoRAG returns `No repository evidence was retrieved for this question.`

## OpenAI provider and security

The provider uses the official Responses API through `openai==3.26.1`, with `instructions`, `input`, `max_output_tokens=1200`, `tools=[]`, and `store=False`. It exposes `response.output_text`, response identity, returned model, usage, and generation latency. No web search, file search, code interpreter, MCP, or other tool is enabled.

The provider reads `OPENAI_API_KEY` only when a live generation call is required. Keys are never logged, persisted, or returned in result objects. `.env` remains gitignored; `.env.example` contains only an empty variable declaration.

The SDK is configured for a 60-second timeout and its bounded two retries. RepoRAG adds no custom retry loop. Authentication, timeout, rate-limit, connection, API, and empty-response failures become concise CLI errors; it never silently falls back to fake generation.

## Dry run

Dry run executes retrieval and context assembly but never calls a generator:

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m reporag.qa.cli `
  data\embeddings\realtimecollab `
  data\bm25\realtimecollab `
  "Where is a realtime client connection handled?" `
  --method hybrid --context-k 5 --dry-run
```

`--show-context` prints the exact context for a live request as well. Questions, prompts, and answers are not persisted by default.

For live generation, set `OPENAI_API_KEY` in the process environment and omit `--dry-run`:

```powershell
$env:OPENAI_API_KEY = "..."
.venv\Scripts\python.exe -m reporag.qa.cli `
  data\embeddings\realtimecollab `
  data\bm25\realtimecollab `
  "Where is JWT authentication enforced?" `
  --method hybrid --context-k 5
```

## Fingerprint and limitations

The deterministic Q&A configuration fingerprint covers Q&A, prompt, and context-format versions; selected retrieval method and its frozen fingerprint; `context_k`; provider; model snapshot; and maximum output tokens. It excludes questions, answers, response IDs, timestamps, API keys, hardware, and machine paths.

Answers are limited by the selected retrieval method and context depth. Citation validation proves that markers name supplied sources, not that every claim is semantically supported. Documentation remains valid frozen corpus content. Context is not compressed, so large chunks consume model input. Version 1 is single-turn and has no conversation memory, external tools, frontend, benchmark, ground truth, or evaluation metrics.
