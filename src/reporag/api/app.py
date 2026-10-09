"""FastAPI application exposing the existing RepoRAG Q&A service."""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from reporag.qa.service import QAServiceError

from .dependencies import APIRequestError, RepositoryQAExecutor, get_qa_executor
from .schemas import (
    ErrorResponse,
    HealthResponse,
    QARequest,
    QAResponse,
    SourceResponse,
)


LOCAL_FRONTEND_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

app = FastAPI(title="RepoRAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(LOCAL_FRONTEND_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(APIRequestError)
async def api_request_error_handler(
    _request: Request, error: APIRequestError
) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": error.detail, "code": error.code},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    _request: Request, _error: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed.", "code": "invalid_request"},
    )


def _safe_generation_error(error: QAServiceError) -> APIRequestError:
    message = str(error).casefold()
    if "api_key is not set" in message:
        return APIRequestError(
            503,
            "provider_key_missing",
            "The selected generation provider is not configured.",
        )
    if "configured output-token limit" in message:
        return APIRequestError(
            422,
            "generation_output_limit",
            "Generation reached the configured output-token limit; retry with a larger max_output_tokens value.",
        )
    return APIRequestError(
        502,
        "generation_failed",
        "The generation provider could not complete the request.",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="RepoRAG")


@app.post(
    "/api/qa",
    response_model=QAResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def repository_qa(
    request: QARequest,
    executor: RepositoryQAExecutor = Depends(get_qa_executor),
) -> QAResponse:
    try:
        result = executor.execute(request)
    except QAServiceError as exc:
        raise _safe_generation_error(exc) from exc

    sources = [
        SourceResponse(
            id=item.source_id,
            file_path=item.file_path,
            symbol=item.symbol_name,
            parent_symbol=item.parent_symbol,
            start_line=item.start_line,
            end_line=item.end_line,
            language=item.language,
            chunk_type=item.chunk_type,
            retrieval_rank=item.retrieval_rank,
            content=item.content,
        )
        for item in result.evidence
    ]
    return QAResponse(
        question=result.question,
        repository=result.evidence[0].repository,
        retrieval_method=result.retrieval_method,
        evidence_scope=result.evidence_scope,
        provider=result.provider or request.provider,
        model=result.model or model_for_request(request.provider),
        answer=result.answer_text,
        citations_valid=result.citations_valid,
        cited_source_ids=list(result.cited_source_ids),
        generation_latency_seconds=result.generation_latency_seconds,
        usage=result.usage,
        sources=sources,
    )


def model_for_request(provider: str) -> str:
    """Resolve a validated provider without reflecting request data in errors."""

    from reporag.qa.config import model_for_provider

    return model_for_provider(provider)
