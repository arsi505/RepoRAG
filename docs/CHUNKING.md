# Code-Aware Chunking

RepoRAG uses code-aware chunking so retrieval units follow meaningful software structures instead of arbitrary character windows. Day 3 consumes only files accepted by the existing ingestion pipeline; it does not rescan repositories with a separate filtering policy.

## Tree-sitter

`tree-sitter-language-pack` provides the grammar binaries, while `reporag.chunking.parser.TreeSitterParser` isolates the rest of the project from its API. The adapter converts native syntax nodes into immutable Python snapshots before returning them. Parser failures and unusable syntax trees fall back safely to line chunks.

For reproducibility, `tree-sitter-language-pack` is pinned to 1.21.0 and the native `tree-sitter` binding is pinned to 0.25.2 for Python 3.14 stability. Required canonical grammars are declared in `language-pack.toml`: `python`, `typescript`, `tsx`, `javascript`, `c`, and `cpp`. The package does not expose a separate `jsx` grammar, so RepoRAG deliberately maps JSX to the canonical `javascript` grammar. Grammars may require a first-use download or explicit prefetch. Downloaded binaries live in the ignored project-owned `data/tree_sitter_cache/` runtime cache and are not committed or vendored.

Structural parsing currently supports:

- Python
- TypeScript and TSX
- JavaScript and JSX (both use the JavaScript grammar)
- C
- C++

Language mapping is centralized in `src/reporag/chunking/languages.py`. C/C++ header ambiguity is not inferred in Version 1: `.h` uses C and `.hpp` uses C++.

## Structural Chunks

The extractors recognize functions, methods, constructors, classes without extracted methods, TypeScript interfaces/type declarations, and enums where the grammar exposes them reliably. Methods retain their parent class. Large classes are not duplicated when method chunks already represent their behavior.

Source outside selected structures is retained in deterministic `file` chunks. Chunk content is decoded from exact UTF-8 source byte slices; RepoRAG does not rewrite identifiers or normalize line endings.

## Fallback Policy

Unsupported languages, configuration/text files, parser failures, syntax trees without useful structures, and files containing no symbols use line-based chunks. Defaults are 80 lines per chunk with 10 lines of overlap. Empty accepted files still produce a chunk. Defaults are centralized in `fallback.py` and can be changed through CLI options.

## Oversized Structures

Functions, methods, and other structural units are capped at 200 lines by default. Oversized units are divided into consecutive non-overlapping parts while preserving symbol and parent metadata. Split parts use one-based `part_index` and a shared `part_count`; ordinary structures use a null `part_index` and `part_count` of 1.

## Stable Identity and Hashing

Each chunk has a SHA-256 content hash over the exact UTF-8 chunk content. Its stable SHA-256 chunk ID is calculated from:

```text
repository, commit hash, relative file path, chunk type,
symbol name, parent symbol, line range, part index/count, content hash
```

No absolute machine path is included. Line numbers are one-based. Final ordering is deterministic by relative file path, start line, end line, and chunk ID.

## Output

The CLI writes one compact JSON object per line to:

```text
data/chunks/<repository-name>.jsonl
```

Runtime JSONL files are gitignored because they contain repository source. `data/chunks/.gitkeep` remains tracked; JSONL is not ignored globally.

## Current Limitations

- Only UTF-8 files accepted by ingestion are chunked.
- Structural extraction is limited to the languages listed above.
- Preprocessor-heavy C/C++ and severely malformed files may use fallback chunks.
- Out-of-class C++ method definitions are functions unless their class relationship is explicit in the selected AST context.
- `.h` files are parsed as C in Version 1.
- Tree-sitter grammars require a one-time download unless the local cache is already populated.
- No embeddings, indexing, retrieval, reranking, generation, or evaluation is implemented on Day 3.
