"""Generation-provider abstractions for repository Q&A."""

from .protocol import GenerationError, GenerationResult, GeneratorProvider

__all__ = ["GenerationError", "GenerationResult", "GeneratorProvider"]
