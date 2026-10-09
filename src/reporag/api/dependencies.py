"""Thin dependency boundary between HTTP handling and existing RepoRAG Q&A."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from reporag.chunking.writer import chunk_filename
from reporag.embeddings.index import IndexErrorBase, load_index as load_vector
from reporag.generation.environment import load_local_environment
from reporag.ingestion.repository import git_metadata
from reporag.lexical.index import BM25IndexError, load_index as load_bm25
from reporag.qa.cli import create_generator
from reporag.qa.config import DEFAULT_QA_CONFIG, model_for_provider
from reporag.qa.models import QAResult
from reporag.qa.retrieval import RepositoryRetriever
from reporag.qa.service import answer_question
from reporag.retrieval.hybrid import HybridCompatibilityError

from .schemas import QARequest


class APIRequestError(RuntimeError):
    """A safe, structured error suitable for an HTTP response."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class RepositoryQAExecutor:
    vector_root: Path = Path("data/embeddings")
    bm25_root: Path = Path("data/bm25")
    model_cache: Path = Path("data/model_cache")

    def execute(self, request: QARequest) -> QAResult:
        repository_path = Path(request.repository).expanduser().resolve()
        if not repository_path.is_dir():
            raise APIRequestError(
                404,
                "repository_not_found",
                "Repository directory does not exist.",
            )

        repository_key = Path(chunk_filename(repository_path.name)).stem
        vector_path = self.vector_root / repository_key
        bm25_path = self.bm25_root / repository_key
        if not vector_path.is_dir() or not bm25_path.is_dir():
            raise APIRequestError(
                409,
                "repository_not_indexed",
                "Compatible RepoRAG runtime artifacts are missing; index the repository first.",
            )

        try:
            vector = load_vector(vector_path)
            bm25 = load_bm25(bm25_path)
            retriever = RepositoryRetriever(
                vector,
                bm25,
                model_cache=self.model_cache,
            )
        except (IndexErrorBase, BM25IndexError, HybridCompatibilityError) as exc:
            raise APIRequestError(
                409,
                "incompatible_artifacts",
                "RepoRAG runtime artifacts are incompatible; index the repository again.",
            ) from exc

        if vector.metadata.repository != repository_path.name:
            raise APIRequestError(
                409,
                "incompatible_artifacts",
                "RepoRAG runtime artifacts do not match the requested repository.",
            )
        commit_hash, _branch = git_metadata(repository_path)
        if vector.metadata.commit_hash is not None and commit_hash is None:
            raise APIRequestError(
                422,
                "git_repository_required",
                "The requested path is not the Git repository used to build these artifacts.",
            )
        if commit_hash is not None and commit_hash != vector.metadata.commit_hash:
            raise APIRequestError(
                409,
                "stale_artifacts",
                "Repository commit does not match its RepoRAG artifacts; index it again.",
            )

        load_local_environment()
        model = model_for_provider(request.provider)
        config = replace(
            DEFAULT_QA_CONFIG,
            default_retrieval_method=request.retrieval_method,
            default_context_k=request.context_k,
            evidence_scope=request.evidence_scope,
            generation_provider=request.provider,
            generation_model=model,
            max_output_tokens=request.max_output_tokens,
        )
        generator = create_generator(
            request.provider,
            dry_run=False,
            max_output_tokens=request.max_output_tokens,
        )
        result = answer_question(
            request.question,
            retriever,
            generator,
            retrieval_method=request.retrieval_method,
            context_k=request.context_k,
            config=config,
            generation_provider=request.provider,
            generation_model=model,
        )
        if not result.evidence:
            raise APIRequestError(
                404,
                "no_evidence",
                "No repository evidence matched the request.",
            )
        return result


def get_qa_executor() -> RepositoryQAExecutor:
    return RepositoryQAExecutor()
