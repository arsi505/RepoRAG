# Repository Ingestion

Day 2 ingestion discovers source, test, configuration, and selected project files in either an existing local directory or a public GitHub repository. It writes metadata only; source contents are not stored in the manifest.

## Accepted Sources

- Local directories are inspected in place and are never modified.
- Public `https://github.com/owner/repository[.git]` URLs are cloned with the installed Git executable into `data/repository_cache/`. A matching clean cached clone is reused. Dirty, invalid, or mismatched cache entries are rejected rather than overwritten.

## Filtering and Safety

The centralized policy in `src/reporag/ingestion/filters.py` allows common source, test, configuration, markup, and documentation extensions plus `Dockerfile` and `Makefile`. Dependency, VCS, build, IDE, virtual-environment, cache, and temporary directories are pruned before traversal.

The default maximum accepted file size is 2 MiB and can be changed with `--max-file-size`. Unsupported extensions, symbolic links, oversized files, binary content, non-UTF-8 content, and unreadable files are skipped with a recorded reason. Files beneath pruned directories are not enumerated, so they do not inflate skipped-file counts.

## Collected Metadata

Each accepted file records its repository-relative POSIX path, extension, recognized language, and byte size. Repository metadata records the name, original source, source type, local ingestion path, current Git commit and branch when available, accepted/skipped counts, and a UTC timestamp.

Generated JSON contains `repository`, `files`, and `skipped_files` sections. Entries are ordered deterministically by relative path. Source contents are intentionally excluded until the later parsing/chunking phase.

## Git Snapshots and Reproducibility

Git metadata is read with the installed `git` command. A current commit hash identifies the ingested snapshot. Non-Git directories remain valid local sources and receive `null` commit and branch values. Detached Git checkouts have a commit hash and a `null` branch.

Cached GitHub clones are reused as their existing clean snapshot; ingestion does not silently update them. Re-cloning or updating a cached snapshot should be an explicit action so the recorded commit remains understandable.

## Current Limitations

- Only public GitHub HTTPS URLs are accepted as remote sources.
- UTF-8 is the only accepted text encoding.
- Language detection is extension/name based.
- Ignored-directory rules are explicit; `.gitignore` patterns are not interpreted.
- No parsing, chunking, embeddings, retrieval, or source-content persistence is implemented on Day 2.
