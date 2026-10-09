"""Language-specific extraction from safe, immutable Tree-sitter node snapshots."""

from __future__ import annotations

import re

from .models import StructuralUnit
from .parser import NodeSnapshot


def _text(node: NodeSnapshot, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _match_name(pattern: str, node: NodeSnapshot, source: bytes) -> str | None:
    match = re.search(pattern, _text(node, source))
    return match.group(1) if match else None


def _contains(outer: NodeSnapshot, inner: NodeSnapshot) -> bool:
    return (
        outer.start_byte <= inner.start_byte
        and inner.end_byte <= outer.end_byte
        and (outer.start_byte, outer.end_byte) != (inner.start_byte, inner.end_byte)
    )


def _nearest_container(
    node: NodeSnapshot, containers: list[tuple[NodeSnapshot, str | None]]
) -> tuple[NodeSnapshot, str | None] | None:
    matches = [item for item in containers if _contains(item[0], node)]
    if not matches:
        return None
    return min(matches, key=lambda item: item[0].end_byte - item[0].start_byte)


def _inside_any(node: NodeSnapshot, containers: list[NodeSnapshot]) -> bool:
    return any(_contains(container, node) for container in containers)


def _unit(
    node: NodeSnapshot,
    *,
    chunk_type: str,
    symbol_name: str | None,
    parent_symbol: str | None,
) -> StructuralUnit:
    return StructuralUnit(
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        start_line=node.start_line,
        end_line=node.end_line,
        chunk_type=chunk_type,
        symbol_name=symbol_name,
        parent_symbol=parent_symbol,
    )


def _python_name(node: NodeSnapshot, source: bytes) -> str | None:
    pattern = (
        r"\bclass\s+([A-Za-z_]\w*)"
        if node.node_type == "class_definition"
        else r"\bdef\s+([A-Za-z_]\w*)"
    )
    return _match_name(pattern, node, source)


def _javascript_name(node: NodeSnapshot, source: bytes) -> str | None:
    patterns = {
        "class_declaration": r"\bclass\s+([A-Za-z_$][\w$]*)",
        "class": r"\bclass\s+([A-Za-z_$][\w$]*)",
        "abstract_class_declaration": r"\bclass\s+([A-Za-z_$][\w$]*)",
        "function_declaration": r"\bfunction\s*\*?\s*([A-Za-z_$][\w$]*)",
        "generator_function_declaration": r"\bfunction\s*\*?\s*([A-Za-z_$][\w$]*)",
        "interface_declaration": r"\binterface\s+([A-Za-z_$][\w$]*)",
        "type_alias_declaration": r"\btype\s+([A-Za-z_$][\w$]*)",
        "enum_declaration": r"\benum\s+([A-Za-z_$][\w$]*)",
    }
    pattern = patterns.get(node.node_type)
    return _match_name(pattern, node, source) if pattern else None


def _javascript_method_name(node: NodeSnapshot, source: bytes) -> str | None:
    declaration = source[node.start_byte : node.end_byte].decode("utf-8")
    matches = re.findall(
        r"([#A-Za-z_$][\w$#]*)\s*(?:<[^{}()]*>)?\s*\(",
        declaration,
    )
    return matches[0] if matches else None


def _named_function_declarator(declaration: str) -> str | None:
    assignment = re.match(
        r"\s*([A-Za-z_$][\w$]*)\s*(?:\??\s*:\s*[^=]+)?=\s*(.+)\Z",
        declaration,
        re.DOTALL,
    )
    if not assignment:
        return None
    initializer = assignment.group(2).lstrip()
    function_expression = re.match(
        r"(?:async\s+)?function(?:\s+[A-Za-z_$][\w$]*)?\s*\(",
        initializer,
    )
    arrow_function = re.match(
        r"(?:async\s+)?(?:<[^>]+>\s*)?"
        r"(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*"
        r"(?::[^=]+)?=>",
        initializer,
        re.DOTALL,
    )
    if not function_expression and not arrow_function:
        return None
    return assignment.group(1)


def _c_family_name(node: NodeSnapshot, source: bytes) -> str | None:
    if node.node_type in {"class_specifier", "struct_specifier"}:
        return _match_name(r"\b(?:class|struct)\s+([A-Za-z_]\w*)", node, source)
    if node.node_type == "enum_specifier":
        return _match_name(r"\benum(?:\s+class)?\s+([A-Za-z_]\w*)", node, source)
    signature = _text(node, source).split("{", 1)[0]
    matches = re.findall(r"(~?[A-Za-z_]\w*)\s*\(", signature)
    return matches[-1] if matches else None


def _extract_python(nodes: tuple[NodeSnapshot, ...], source: bytes) -> list[StructuralUnit]:
    classes = [
        (node, _python_name(node, source))
        for node in nodes
        if node.node_type == "class_definition" and not node.has_errors
    ]
    functions = [
        node for node in nodes if node.node_type == "function_definition" and not node.has_errors
    ]
    units: list[StructuralUnit] = []
    method_classes: set[tuple[int, int]] = set()

    for function in functions:
        if _inside_any(function, [other for other in functions if other is not function]):
            continue
        parent = _nearest_container(function, classes)
        parent_name = parent[1] if parent else None
        symbol = _python_name(function, source)
        chunk_type = "function"
        if parent is not None:
            method_classes.add((parent[0].start_byte, parent[0].end_byte))
            chunk_type = "constructor" if symbol == "__init__" else "method"
        units.append(
            _unit(
                function,
                chunk_type=chunk_type,
                symbol_name=symbol,
                parent_symbol=parent_name,
            )
        )

    for class_node, class_name in classes:
        if (class_node.start_byte, class_node.end_byte) not in method_classes:
            parent = _nearest_container(class_node, classes)
            units.append(
                _unit(
                    class_node,
                    chunk_type="class",
                    symbol_name=class_name,
                    parent_symbol=parent[1] if parent else None,
                )
            )
    return units


def _extract_javascript_like(
    nodes: tuple[NodeSnapshot, ...], source: bytes, language: str
) -> list[StructuralUnit]:
    class_types = {"class_declaration", "class"}
    if language in {"typescript", "tsx"}:
        class_types.add("abstract_class_declaration")
    classes = [
        (node, _javascript_name(node, source))
        for node in nodes
        if node.node_type in class_types and not node.has_errors
    ]
    methods = [
        node for node in nodes if node.node_type == "method_definition" and not node.has_errors
    ]
    units: list[StructuralUnit] = []
    method_classes: set[tuple[int, int]] = set()

    for method in methods:
        parent = _nearest_container(method, classes)
        symbol = _javascript_method_name(method, source)
        if parent is not None:
            method_classes.add((parent[0].start_byte, parent[0].end_byte))
        units.append(
            _unit(
                method,
                chunk_type="constructor" if symbol == "constructor" else "method",
                symbol_name=symbol,
                parent_symbol=parent[1] if parent else None,
            )
        )

    function_types = {"function_declaration", "generator_function_declaration"}
    for function in nodes:
        if function.has_errors or function.node_type not in function_types:
            continue
        if _inside_any(function, methods):
            continue
        units.append(
            _unit(
                function,
                chunk_type="function",
                symbol_name=_javascript_name(function, source),
                parent_symbol=None,
            )
        )

    for declaration in nodes:
        if declaration.has_errors or declaration.node_type != "variable_declarator":
            continue
        if _inside_any(declaration, methods):
            continue
        text = _text(declaration, source)
        symbol_name = _named_function_declarator(text)
        if symbol_name is None:
            continue
        units.append(
            _unit(
                declaration,
                chunk_type="function",
                symbol_name=symbol_name,
                parent_symbol=None,
            )
        )

    for declaration in nodes:
        if declaration.has_errors:
            continue
        if declaration.node_type in {"interface_declaration", "type_alias_declaration"}:
            units.append(
                _unit(
                    declaration,
                    chunk_type=(
                        "interface" if declaration.node_type == "interface_declaration" else "type"
                    ),
                    symbol_name=_javascript_name(declaration, source),
                    parent_symbol=None,
                )
            )
        elif declaration.node_type == "enum_declaration":
            units.append(
                _unit(
                    declaration,
                    chunk_type="enum",
                    symbol_name=_javascript_name(declaration, source),
                    parent_symbol=None,
                )
            )

    for class_node, class_name in classes:
        if (class_node.start_byte, class_node.end_byte) not in method_classes:
            units.append(
                _unit(
                    class_node,
                    chunk_type="class",
                    symbol_name=class_name,
                    parent_symbol=None,
                )
            )
    return units


def _extract_c_family(
    nodes: tuple[NodeSnapshot, ...], source: bytes, language: str
) -> list[StructuralUnit]:
    class_types = {"class_specifier", "struct_specifier"} if language == "cpp" else set()
    classes = [
        (node, _c_family_name(node, source))
        for node in nodes
        if node.node_type in class_types and not node.has_errors
    ]
    functions = [
        node for node in nodes if node.node_type == "function_definition" and not node.has_errors
    ]
    units: list[StructuralUnit] = []
    method_classes: set[tuple[int, int]] = set()

    for function in functions:
        if _inside_any(function, [other for other in functions if other is not function]):
            continue
        parent = _nearest_container(function, classes)
        symbol = _c_family_name(function, source)
        chunk_type = "function"
        if parent is not None:
            method_classes.add((parent[0].start_byte, parent[0].end_byte))
            chunk_type = "constructor" if symbol == parent[1] else "method"
        units.append(
            _unit(
                function,
                chunk_type=chunk_type,
                symbol_name=symbol,
                parent_symbol=parent[1] if parent else None,
            )
        )

    for node in nodes:
        if node.node_type == "enum_specifier" and not node.has_errors:
            name = _c_family_name(node, source)
            if name:
                units.append(
                    _unit(
                        node,
                        chunk_type="enum",
                        symbol_name=name,
                        parent_symbol=None,
                    )
                )

    for class_node, class_name in classes:
        if (class_node.start_byte, class_node.end_byte) not in method_classes:
            units.append(
                _unit(
                    class_node,
                    chunk_type="class",
                    symbol_name=class_name,
                    parent_symbol=None,
                )
            )
    return units


def extract_structures(
    language: str,
    nodes: tuple[NodeSnapshot, ...],
    source: bytes,
) -> tuple[StructuralUnit, ...]:
    if language == "python":
        units = _extract_python(nodes, source)
    elif language in {"javascript", "typescript", "tsx"}:
        units = _extract_javascript_like(nodes, source, language)
    elif language in {"c", "cpp"}:
        units = _extract_c_family(nodes, source, language)
    else:
        units = []
    return tuple(
        sorted(units, key=lambda unit: (unit.start_byte, unit.end_byte, unit.chunk_type))
    )
