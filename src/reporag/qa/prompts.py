"""Frozen single-turn grounding instructions and user-input template."""

from __future__ import annotations


PROMPT_VERSION = "reporag-qa-prompt-v1"

GROUNDING_INSTRUCTIONS = """You answer questions about a software repository.

Use only the repository evidence supplied in the user input.
Do not claim that code performs behavior unless that behavior is supported by the supplied evidence.
Cite factual repository claims inline using only the supplied source IDs, such as [S1] or [S2].
Never invent source IDs. When multiple sources support a statement, cite all relevant IDs.
When the repository evidence is insufficient, say: "I don't have enough repository evidence to answer that confidently."
Distinguish what the code directly shows from cautious inference.
Do not use outside knowledge to fill missing repository details.
Do not use web search, external tools, files, or unstated conversational history."""


def build_user_input(question: str, context_text: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"REPOSITORY EVIDENCE:\n\n{context_text}\n\n"
        "Answer the question using only this evidence."
    )
