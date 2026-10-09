"""Explicit code-aware lexical tokenizer."""

from __future__ import annotations

import re


_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*")
_COMPONENT = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def _components(identifier: str) -> tuple[str, ...]:
    components: list[str] = []
    for separated in re.split(r"[_-]+", identifier):
        components.extend(match.group(0).lower() for match in _COMPONENT.finditer(separated))
    return tuple(component for component in components if component)


def tokenize(text: str) -> tuple[str, ...]:
    """Return lowercase exact identifier tokens followed by useful components.

    Punctuation and whitespace only separate tokens. No stopword removal,
    stemming, network service, or language model is involved.
    """

    tokens: list[str] = []
    for match in _IDENTIFIER.finditer(text):
        original = match.group(0).lower()
        tokens.append(original)
        parts = _components(match.group(0))
        if len(parts) > 1:
            tokens.extend(part for part in parts if part != original)
    return tuple(tokens)
