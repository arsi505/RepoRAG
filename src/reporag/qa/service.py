"""Single-retrieval, single-generation grounded Q&A orchestration."""

from __future__ import annotations

from time import perf_counter

from reporag.generation.protocol import GenerationError, GeneratorProvider

from .citations import validate_citations
from .config import DEFAULT_QA_CONFIG, QAConfig, qa_fingerprint
from .context import build_evidence, format_context
from .models import QAResult, QATimings
from .prompts import GROUNDING_INSTRUCTIONS, build_user_input
from .retrieval import QARetriever


EMPTY_RETRIEVAL_ANSWER = "No repository evidence was retrieved for this question."
DRY_RUN_ANSWER = "Dry run completed; generation was not called."


class QAServiceError(RuntimeError):
    """Raised for a concise Q&A orchestration failure."""


def answer_question(
    question: str,
    retriever: QARetriever,
    generator: GeneratorProvider,
    *,
    retrieval_method: str = "hybrid",
    context_k: int = 5,
    dry_run: bool = False,
    config: QAConfig = DEFAULT_QA_CONFIG,
    generation_provider: str | None = None,
    generation_model: str | None = None,
) -> QAResult:
    if not question.strip():
        raise ValueError("Question must not be empty")
    config.validate_request(retrieval_method, context_k)
    fingerprint = qa_fingerprint(
        retrieval_method=retrieval_method,
        retrieval_fingerprint=retriever.fingerprint(retrieval_method),
        context_k=context_k,
        config=config,
        generation_provider=generation_provider,
        generation_model=generation_model,
    )
    total_started = perf_counter()
    retrieval_started = perf_counter()
    results = retriever.retrieve(question, retrieval_method, context_k)
    retrieval_seconds = perf_counter() - retrieval_started

    context_started = perf_counter()
    evidence = build_evidence(results, context_k=context_k)
    context_text = format_context(evidence)
    user_input = build_user_input(question, context_text) if evidence else ""
    context_seconds = perf_counter() - context_started

    if not evidence:
        total_seconds = perf_counter() - total_started
        return QAResult(
            question=question,
            answer_text=EMPTY_RETRIEVAL_ANSWER,
            retrieval_method=retrieval_method,
            context_k=context_k,
            cited_source_ids=(),
            unknown_source_ids=(),
            citations_valid=False,
            evidence=(),
            context_text="",
            qa_fingerprint=fingerprint,
            timings=QATimings(
                retrieval_seconds, context_seconds, None, 0.0, total_seconds
            ),
        )

    if dry_run:
        total_seconds = perf_counter() - total_started
        return QAResult(
            question=question,
            answer_text=DRY_RUN_ANSWER,
            retrieval_method=retrieval_method,
            context_k=context_k,
            cited_source_ids=(),
            unknown_source_ids=(),
            citations_valid=False,
            evidence=evidence,
            context_text=context_text,
            qa_fingerprint=fingerprint,
            dry_run=True,
            timings=QATimings(
                retrieval_seconds, context_seconds, None, 0.0, total_seconds
            ),
        )

    try:
        generation = generator.generate(GROUNDING_INSTRUCTIONS, user_input)
    except GenerationError as exc:
        raise QAServiceError(f"Generation failed: {exc}") from exc
    if not generation.text.strip():
        raise QAServiceError("Generation failed: provider returned empty text")

    citation_started = perf_counter()
    validation = validate_citations(generation.text, evidence)
    citation_seconds = perf_counter() - citation_started
    total_seconds = perf_counter() - total_started
    return QAResult(
        question=question,
        answer_text=generation.text,
        retrieval_method=retrieval_method,
        context_k=context_k,
        cited_source_ids=validation.cited_source_ids,
        unknown_source_ids=validation.unknown_source_ids,
        citations_valid=validation.citations_valid,
        evidence=evidence,
        context_text=context_text,
        qa_fingerprint=fingerprint,
        provider=generation.provider,
        model=generation.model,
        response_id=generation.response_id,
        usage=generation.usage,
        generation_latency_seconds=generation.latency_seconds,
        timings=QATimings(
            retrieval_seconds,
            context_seconds,
            generation.latency_seconds,
            citation_seconds,
            total_seconds,
        ),
    )
