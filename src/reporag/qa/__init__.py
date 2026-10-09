"""Single-turn, evidence-grounded repository question answering."""

from .config import DEFAULT_QA_CONFIG, QAConfig
from .service import answer_question

__all__ = ["DEFAULT_QA_CONFIG", "QAConfig", "answer_question"]
