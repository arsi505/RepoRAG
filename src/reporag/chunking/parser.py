"""Small adapter around the third-party Tree-sitter language pack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ParserFailure(RuntimeError):
    """Raised when a grammar cannot parse a source file."""


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    node_type: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    has_errors: bool


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    nodes: tuple[NodeSnapshot, ...]
    has_errors: bool


def _get_parser(grammar: str, cache_directory: str) -> tuple[Any, Any]:
    from tree_sitter import Parser
    from tree_sitter_language_pack import PackConfig, configure, get_language

    configure(PackConfig(cache_dir=cache_directory))
    language = get_language(grammar)
    return Parser(language), language


class TreeSitterParser:
    """Language-independent parsing interface used by the chunker."""

    def __init__(self, cache_directory: Path = Path("data/tree_sitter_cache")) -> None:
        self.cache_directory = cache_directory.resolve()

    def parse(self, grammar: str, source: bytes) -> ParseOutcome:
        try:
            parser, _language = _get_parser(grammar, str(self.cache_directory))
            tree = parser.parse(source)
            root = tree.root_node
            has_errors = bool(root.has_error)
            snapshots: list[NodeSnapshot] = []
            stack = [root]
            while stack:
                node = stack.pop()
                snapshots.append(
                    NodeSnapshot(
                        node_type=node.type,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        start_line=node.start_point.row + 1,
                        end_line=node.end_point.row + 1,
                        has_errors=bool(node.has_error),
                    )
                )
                stack.extend(reversed(node.named_children))
        except Exception as exc:
            raise ParserFailure(f"Tree-sitter parser failed for {grammar}: {exc}") from exc
        return ParseOutcome(tuple(snapshots), has_errors)
