# Repository Question Answering

Day 8 adds a single-turn product and demonstration layer over RepoRAG's frozen retrieval methods. It retrieves repository chunks, assigns deterministic evidence IDs, sends the exact evidence to a generator, and validates source markers in the returned answer. It does not change the retrieval research methods or add an evaluation benchmark.

## Frozen Q&A configuration

- Q&A version: `reporag-qa-v1`
- Prompt version: `reporag-qa-prompt-v1`
- Context formatter: `reporag-qa-context-v1`
- Default retrieval method: `hybrid`
- Default context depth: `context_k = 5`
- Default evidence scope: `all`
- Default provider: `deepseek`
- Default model: `deepseek-flash`
- Alternative provider/models: `gemini` / `gemini-3.8-flash`; `openai` / `gpt-5.4-mini-2026-03-17`
- Default maximum output: 2,000 tokens (configurable with `--max-output-tokens`)
- Google Gen AI SDK: `google-genai==2.29.0`
- OpenAI SDK: `openai==3.26.1`

Hybrid is the interactive default because it is substantially faster than CPU reranking and combines the existing semantic and lexical rankings. This is a practical product default, not a claim that Hybrid is scientifically superior.

Users may select `vector`, `bm25`, `hybrid`, or `reranked`. The original question is passed unchanged to exactly one retrieval call. No query rewriting, expansion, multi-query retrieval, agent loop, or conversational history is used.

## Evidence and prompting

Retrieval order is preserved. The Q&A product layer supports `--evidence-scope all`, `code`, or `docs`. `all` retains the current behavior. For `code` and `docs`, Q&A requests a deterministic candidate pool of `max(50, context_k * 10)`, preserves its ranking, filters Markdown documentation from or into the context as requested, and takes the first `context_k` matches. It never falls back between scopes. This filtering does not change Methods A-D.

After scope filtering and deterministic duplicate removal, evidence receives IDs `S1`, `S2`, and so on. Each block contains the repository-relative path, language, chunk type, symbol and parent, line range, retrieval rank, and exact frozen chunk content. Absolute local paths, timestamps, retrieval scores, secrets, and model-cache details are excluded.

The provider receives one instruction template and one user input containing the exact question and source blocks. Instructions require repository-only answers, inline `[S1]` citations, cautious inference, and an explicit insufficient-evidence response rather than guessing.

After generation, RepoRAG extracts markers matching `[S1]`, `[S2]`, and so on. Repeated markers are deduplicated in first-use order. Every cited ID must exist in the supplied context. Unknown IDs are reported and make citation validation fail; an evidence-based answer with no source markers also fails validation. IDs are mapped back to path, symbol, and lines for human-readable CLI output.

If retrieval returns no evidence, generation is not called and RepoRAG returns `No repository evidence was retrieved for this question.`

## Generation providers and security

DeepSeek is the recommended practical default. Its provider uses the existing `openai==3.26.1` SDK with the Responses API at `https://api.deepseek.com`, the `deepseek-flash` model, shared grounding instructions, unchanged question-plus-evidence input, `max_output_tokens=2000` by default, `tools=[]`, and `store=False`.

Gemini remains supported with `--provider gemini`. Its provider uses the official `google-genai==2.29.0` SDK and `models.generate_content`, with the same system instruction, user content, and output limit. The model is `gemini-3.8-flash`.

OpenAI remains supported with `--provider openai`. Its provider uses the official Responses API through `openai==3.26.1`, with `instructions`, `input`, the shared configurable output limit, `tools=[]`, and `store=False`. All providers expose text, provider/model identity, response identity when available, usage when returned, and generation latency. No provider enables web search, URL retrieval, external retrieval, code execution, function calling, browsing, or other tools.

Providers read `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` only when a live generation call is required. For local CLI use, `python-dotenv==1.2.4` loads the gitignored `.env` without overriding environment variables already set by the process. Keys are never logged, persisted, or returned in result objects. `.env.example` contains empty placeholders only; API keys must never be committed.

Authentication, timeout, rate-limit, connection, provider, and empty-response failures become concise CLI errors; RepoRAG never silently falls back to fake generation. The OpenAI SDK retains its 60-second timeout and bounded two retries. RepoRAG adds no custom retry loop.

## Dry run

Dry run executes retrieval and context assembly but never calls a generator:

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m reporag.qa.cli `
  data\embeddings\realtimecollab `
  data\bm25\realtimecollab `
  "Where is a realtime client connection handled?" `
  --method hybrid --context-k 5 --evidence-scope code --dry-run
```

`--show-context` prints the exact context for a live request as well. Questions, prompts, and answers are not persisted by default.

For live DeepSeek generation, place `DEEPSEEK_API_KEY` in `.env` or set it in the process environment, then omit `--dry-run`:

```powershell
.venv\Scripts\python.exe -m reporag.qa.cli `
  data\embeddings\realtimecollab `
  data\bm25\realtimecollab `
  "Where is JWT authentication enforced?" `
  --provider deepseek --method hybrid --context-k 5 --max-output-tokens 2000
```

Choose Gemini with `--provider gemini` and `GEMINI_API_KEY`, or OpenAI with `--provider openai` and `OPENAI_API_KEY`. The provider defaults to `deepseek` when the option is omitted.

## Fingerprint and limitations

The deterministic Q&A configuration fingerprint covers Q&A, prompt, and context-format versions; selected retrieval method and its frozen fingerprint; `context_k`; evidence scope; selected provider; selected model snapshot; and maximum output tokens. Switching scope or generation provider therefore changes the fingerprint. It excludes questions, answers, response IDs, timestamps, API keys, hardware, and machine paths.

Answers are limited by the selected retrieval method and context depth. Citation validation proves that markers name supplied sources, not that every claim is semantically supported. Documentation remains valid frozen corpus content. Context is not compressed, so large chunks consume model input. Version 1 is single-turn and has no conversation memory, external tools, benchmark, ground truth, or evaluation metrics. Its API and web interface are local-only and require prebuilt indexes.
