#!/usr/bin/env python3
"""AST audit for the deterministic Codex timeout regression test.

This audit deliberately inspects only the owned timeout test. It rejects
real-time primitives and wall-clock assertions so the behavioral test cannot
silently become timing-sensitive again.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path


_TARGET_CLASS = "TestCodexAuxiliaryAdapterTimeout"
_TARGET_TEST = "test_enforces_total_timeout_while_stream_keeps_emitting_events"
_REAL_TIME_ATTRIBUTES = {
    "clock_gettime",
    "clock_gettime_ns",
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "process_time_ns",
    "sleep",
    "thread_time",
    "thread_time_ns",
    "time",
    "time_ns",
}
_DATETIME_CLOCK_ATTRIBUTES = {"now", "utcnow", "today"}
_ASYNCIO_TIME_ATTRIBUTES = {
    "as_completed",
    "sleep",
    "timeout",
    "timeout_at",
    "wait",
    "wait_for",
}
_ESCAPE_NAMES = {
    "flaky",
    "repeat",
    "retries",
    "retry",
    "rerun",
    "skip",
    "skipif",
    "xfail",
}
_ELAPSED_NAME_PARTS = (
    "duration",
    "elapsed",
    "latency",
    "wall_clock",
    "wallclock",
)
_DYNAMIC_IMPORTS = {"__import__", "builtins.__import__", "importlib.import_module"}
_DYNAMIC_GETATTRS = {"getattr", "builtins.getattr"}
_PARTIALS = {"partial", "functools.partial"}
_LESS_THAN_FUNCTIONS = {"lt", "le", "operator.lt", "operator.le"}
_GREATER_THAN_FUNCTIONS = {"gt", "ge", "operator.gt", "operator.ge"}


class _Scope:
    """Lexical bindings and alias equations for one Python scope."""

    _next_identifier = 0

    def __init__(self, node: ast.AST, parent: _Scope | None) -> None:
        self.node = node
        self.parent = parent
        self.children: list[_Scope] = []
        self.bindings: dict[str, set[str]] = {}
        self.equations: dict[str, list[ast.AST]] = {}
        self.local_names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()
        self.identifier = _Scope._next_identifier
        _Scope._next_identifier += 1
        if parent is not None:
            parent.children.append(self)

    def add_binding(self, name: str, canonical: str) -> bool:
        values = self.bindings.setdefault(name, set())
        before = len(values)
        values.add(canonical)
        return len(values) != before

    def add_equation(self, name: str, value: ast.AST) -> None:
        self.equations.setdefault(name, []).append(value)


class _ScopeIndex(ast.NodeVisitor):
    """Map every expression to the lexical scope where it is evaluated."""

    def __init__(self, tree: ast.Module) -> None:
        self.root = _Scope(tree, None)
        self.current = self.root
        self.scope_by_node: dict[ast.AST, _Scope] = {tree: self.root}
        self.owner_scope: dict[ast.AST, _Scope] = {tree: self.root}
        self.parent_by_node: dict[ast.AST, ast.AST] = {}
        self.scopes = [self.root]
        self.visit(tree)

    def _visit_child(self, parent: ast.AST, child: ast.AST | None) -> None:
        if child is None:
            return
        self.parent_by_node[child] = parent
        self.visit(child)

    def _visit_children(
        self,
        parent: ast.AST,
        children: Iterable[ast.AST],
    ) -> None:
        for child in children:
            self._visit_child(parent, child)

    def generic_visit(self, node: ast.AST) -> None:
        self.scope_by_node.setdefault(node, self.current)
        for child in ast.iter_child_nodes(node):
            self._visit_child(node, child)

    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        self.scope_by_node[node] = self.current
        self._visit_children(node, node.body)

    def _visit_arguments_in_parent(self, parent: ast.AST, args: ast.arguments) -> None:
        self.scope_by_node[args] = self.current
        self.parent_by_node[args] = parent
        for default in [*args.defaults, *args.kw_defaults]:
            self._visit_child(args, default)
        for argument in [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *([args.vararg] if args.vararg else []),
            *([args.kwarg] if args.kwarg else []),
        ]:
            self.scope_by_node[argument] = self.current
            self.parent_by_node[argument] = args
            self._visit_child(argument, argument.annotation)

    def _function_scope(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self.scope_by_node[node] = self.current
        self.current.local_names.add(node.name)
        self._visit_children(node, node.decorator_list)
        self._visit_arguments_in_parent(node, node.args)
        self._visit_child(node, node.returns)
        for type_parameter in getattr(node, "type_params", []):
            self._visit_child(node, type_parameter)

        child = _Scope(node, self.current)
        self.scopes.append(child)
        self.owner_scope[node] = child
        previous = self.current
        self.current = child
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *([node.args.vararg] if node.args.vararg else []),
            *([node.args.kwarg] if node.args.kwarg else []),
        ]:
            child.local_names.add(argument.arg)
        self._visit_children(node, node.body)
        self.current = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._function_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._function_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope_by_node[node] = self.current
        self.current.local_names.add(node.name)
        self._visit_children(node, node.decorator_list)
        self._visit_children(node, node.bases)
        self._visit_children(node, node.keywords)
        for type_parameter in getattr(node, "type_params", []):
            self._visit_child(node, type_parameter)

        child = _Scope(node, self.current)
        self.scopes.append(child)
        self.owner_scope[node] = child
        previous = self.current
        self.current = child
        self._visit_children(node, node.body)
        self.current = previous

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self.scope_by_node[node] = self.current
        self._visit_arguments_in_parent(node, node.args)
        child = _Scope(node, self.current)
        self.scopes.append(child)
        self.owner_scope[node] = child
        previous = self.current
        self.current = child
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *([node.args.vararg] if node.args.vararg else []),
            *([node.args.kwarg] if node.args.kwarg else []),
        ]:
            child.local_names.add(argument.arg)
        self._visit_child(node, node.body)
        self.current = previous


class _Resolver:
    """Scope-aware fixed-point resolver for imports and value aliases."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.index = _ScopeIndex(tree)
        self._collect_bindings()
        self._resolve_fixed_point()

    def scope(self, node: ast.AST) -> _Scope:
        return self.index.scope_by_node.get(node, self.index.root)

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.index.parent_by_node.get(node)

    def _collect_bindings(self) -> None:
        for node, scope in self.index.scope_by_node.items():
            if isinstance(node, ast.Global):
                scope.global_names.update(node.names)
            elif isinstance(node, ast.Nonlocal):
                scope.nonlocal_names.update(node.names)
            elif isinstance(node, ast.Import):
                for imported in node.names:
                    bound = imported.asname or imported.name.split(".", 1)[0]
                    canonical = imported.name if imported.asname else bound
                    scope.local_names.add(bound)
                    scope.add_binding(bound, canonical)
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                for imported in node.names:
                    if imported.name == "*":
                        continue
                    bound = imported.asname or imported.name
                    canonical = ".".join(
                        part for part in (module, imported.name) if part
                    )
                    scope.local_names.add(bound)
                    scope.add_binding(bound, canonical)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    self._add_assignment(scope, target, node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self._add_assignment(scope, node.target, node.value)
            elif isinstance(node, ast.NamedExpr):
                self._add_assignment(scope, node.target, node.value)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope.local_names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                scope.local_names.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                scope.local_names.add(node.name)

        for scope in self.index.scopes:
            scope.local_names.difference_update(scope.global_names)
            scope.local_names.difference_update(scope.nonlocal_names)

    def _add_assignment(self, scope: _Scope, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            value, (ast.Tuple, ast.List)
        ):
            for target_item, value_item in zip(target.elts, value.elts):
                self._add_assignment(scope, target_item, value_item)
            return
        for name in _assignment_paths(target):
            if "." not in name:
                scope.local_names.add(name)
            scope.add_equation(name, value)

    def _resolve_fixed_point(self) -> None:
        equation_count = sum(
            len(values)
            for scope in self.index.scopes
            for values in scope.equations.values()
        )
        for _ in range(max(4, equation_count + len(self.index.scopes) + 1)):
            changed = False
            for scope in self.index.scopes:
                for target, values in scope.equations.items():
                    for value in values:
                        for canonical in self.reference(value, scope):
                            changed = scope.add_binding(target, canonical) or changed
            if not changed:
                return

    def _lookup(self, name: str, scope: _Scope) -> set[str]:
        current: _Scope | None = scope
        while current is not None:
            if name in current.bindings:
                return set(current.bindings[name])
            root = name.split(".", 1)[0]
            if root in current.local_names:
                return set()
            current = current.parent
        return {name}

    def _lookup_attribute_alias(self, name: str, scope: _Scope) -> set[str] | None:
        """Look up an explicitly assigned dotted path, not merely its root."""
        current: _Scope | None = scope
        while current is not None:
            if name in current.bindings:
                return set(current.bindings[name])
            root = name.split(".", 1)[0]
            if root in current.local_names:
                return None
            current = current.parent
        return None

    def reference(
        self,
        node: ast.AST,
        scope: _Scope | None = None,
        seen: set[tuple[int, int]] | None = None,
    ) -> set[str]:
        scope = scope or self.scope(node)
        seen = seen or set()
        marker = (id(node), scope.identifier)
        if marker in seen:
            return set()
        seen = {*seen, marker}

        if isinstance(node, ast.Name):
            return self._lookup(node.id, scope)
        if isinstance(node, ast.Attribute):
            lexical = _lexical_path(node)
            if lexical:
                direct = self._lookup_attribute_alias(lexical, scope)
                if direct is not None:
                    return direct
            return {
                f"{base}.{node.attr}"
                for base in self.reference(node.value, scope, seen)
            }
        if isinstance(node, ast.NamedExpr):
            return self.reference(node.value, scope, seen)
        if isinstance(node, ast.IfExp):
            return self.reference(node.body, scope, seen) | self.reference(
                node.orelse, scope, seen
            )
        if isinstance(node, ast.Call):
            functions = self.reference(node.func, scope, seen)
            if functions & _DYNAMIC_IMPORTS and node.args:
                return self.strings(node.args[0], scope)
            if functions & _DYNAMIC_GETATTRS and len(node.args) >= 2:
                bases = self.reference(node.args[0], scope, seen)
                attributes = self.strings(node.args[1], scope)
                return {
                    f"{base}.{attribute}"
                    for base in bases
                    for attribute in attributes
                }
            if functions & _PARTIALS and node.args:
                return self.reference(node.args[0], scope, seen)
            # Preserve the called symbol as the receiver identity for chains
            # such as threading.Event().wait() and get_running_loop().time().
            return functions
        if isinstance(node, ast.Subscript):
            return self.reference(node.value, scope, seen)
        return set()

    def strings(
        self,
        node: ast.AST,
        scope: _Scope | None = None,
        seen: set[tuple[int, str, int]] | None = None,
    ) -> set[str]:
        scope = scope or self.scope(node)
        seen = seen or set()
        marker = (scope.identifier, _lexical_path(node) or "", id(node))
        if marker in seen:
            return set()
        seen = {*seen, marker}
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            values: set[str] = set()
            current: _Scope | None = scope
            while current is not None:
                if node.id in current.equations:
                    for value in current.equations[node.id]:
                        values.update(self.strings(value, current, seen))
                    return values
                if node.id in current.local_names:
                    return values
                current = current.parent
            return values
        if isinstance(node, ast.NamedExpr):
            return self.strings(node.value, scope, seen)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.strings(node.left, scope, seen)
            right = self.strings(node.right, scope, seen)
            return {left_value + right_value for left_value in left for right_value in right}
        if isinstance(node, ast.IfExp):
            return self.strings(node.body, scope, seen) | self.strings(
                node.orelse, scope, seen
            )
        return set()

    def equations(self, node: ast.AST, scope: _Scope) -> list[tuple[_Scope, ast.AST]]:
        path = _lexical_path(node)
        if not path:
            return []
        current: _Scope | None = scope
        while current is not None:
            if path in current.equations:
                return [(current, value) for value in current.equations[path]]
            root = path.split(".", 1)[0]
            if root in current.local_names:
                return []
            current = current.parent
        return []


def _lexical_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _assignment_paths(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Starred):
        return _assignment_paths(target.value)
    if isinstance(target, (ast.Name, ast.Attribute)):
        path = _lexical_path(target)
        return [path] if path else []
    if isinstance(target, (ast.Tuple, ast.List)):
        paths: list[str] = []
        for element in target.elts:
            paths.extend(_assignment_paths(element))
        return paths
    return []


def _source(node: ast.AST, text: str) -> str:
    return ast.get_source_segment(text, node) or ast.dump(node, include_attributes=False)


def _finding(kind: str, node: ast.AST, text: str, detail: str) -> dict[str, object]:
    return {
        "kind": kind,
        "line": getattr(node, "lineno", None),
        "detail": detail,
        "source": _source(node, text),
    }


def _append_unique(
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
    finding: dict[str, object],
) -> None:
    key = (finding["kind"], finding["line"], finding["source"])
    if key not in seen:
        findings.append(finding)
        seen.add(key)


def _looks_elapsed_name(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in _ELAPSED_NAME_PARTS)


def _real_time_kind(symbol: str, node: ast.AST) -> tuple[str, str] | None:
    normalized = symbol.removeprefix("builtins.")
    namespace, _, attribute = normalized.rpartition(".")
    if namespace == "time" and attribute in _REAL_TIME_ATTRIBUTES:
        return "real_time_primitive", f"forbidden real-time reference: {symbol}"
    if not namespace and normalized in _REAL_TIME_ATTRIBUTES:
        return "real_time_primitive", f"forbidden real-time reference: {symbol}"
    if normalized == "timeit.default_timer" or normalized == "default_timer":
        return "real_time_primitive", f"forbidden real-time reference: {symbol}"
    if namespace == "asyncio" and attribute in _ASYNCIO_TIME_ATTRIBUTES:
        kind = "real_timer_constructor" if attribute in {"timeout", "timeout_at"} else "real_time_primitive"
        return kind, f"forbidden asyncio time reference: {symbol}"
    if normalized in {"threading.Timer", "Timer"}:
        return "real_timer_constructor", f"forbidden real timer reference: {symbol}"
    if normalized in {"threading.Event.wait", "Event.wait"}:
        return "real_time_primitive", f"forbidden real wait reference: {symbol}"
    if normalized in {"signal.alarm", "signal.setitimer", "alarm", "setitimer"}:
        return "real_timer_constructor", f"forbidden signal timer reference: {symbol}"
    if attribute in _DATETIME_CLOCK_ATTRIBUTES and (
        namespace.startswith("datetime.") or namespace in {"date", "datetime"}
    ):
        return "real_time_primitive", f"forbidden datetime reference: {symbol}"
    if attribute in {"call_at", "call_later"}:
        return "real_timer_constructor", f"forbidden event-loop timer reference: {symbol}"
    if attribute == "time" and (
        "get_event_loop" in namespace
        or "get_running_loop" in namespace
        or any("loop" in part.lower() for part in namespace.split("."))
    ):
        return "event_loop_time", f"forbidden event-loop clock reference: {symbol}"

    lexical = _lexical_path(node) or ""
    if lexical.rsplit(".", 1)[-1] in {"call_at", "call_later"}:
        return "real_timer_constructor", "event-loop timer construction is forbidden"
    if lexical.rsplit(".", 1)[-1] == "time" and any(
        "loop" in part.lower() for part in lexical.split(".")[:-1]
    ):
        return "event_loop_time", "event-loop clock .time is forbidden"
    return None


def _escape_kind(symbol: str) -> tuple[str, str] | None:
    name = symbol.rsplit(".", 1)[-1]
    if name not in _ESCAPE_NAMES:
        return None
    if name in {"skip", "skipif", "xfail"}:
        return "skip_or_xfail", f"forbidden skip/xfail reference: {symbol}"
    return "retry_or_repeat", f"forbidden retry/repeat reference: {symbol}"


def _maximal_reference(node: ast.AST, resolver: _Resolver) -> bool:
    parent = resolver.parent(node)
    if isinstance(parent, ast.Attribute) and parent.value is node:
        return False
    if isinstance(parent, ast.Call) and parent.func is node:
        return False
    return isinstance(node, (ast.Name, ast.Attribute, ast.Call))


def _iter_references(root: ast.AST, resolver: _Resolver):
    for node in ast.walk(root):
        if _maximal_reference(node, resolver):
            yield node, resolver.reference(node)


def _scan_real_time(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    for node, symbols in _iter_references(method, resolver):
        for symbol in sorted(symbols):
            classified = _real_time_kind(symbol, node)
            if classified:
                kind, detail = classified
                _append_unique(findings, seen, _finding(kind, node, text, detail))
                break


def _target_looks_clock_derived(node: ast.AST) -> bool:
    return any(_looks_elapsed_name(path) for path in _assignment_paths(node))


def _clock_derived(
    node: ast.AST,
    resolver: _Resolver,
    scope: _Scope | None = None,
    seen: set[tuple[int, int]] | None = None,
) -> bool:
    scope = scope or resolver.scope(node)
    seen = seen or set()
    marker = (id(node), scope.identifier)
    if marker in seen:
        return False
    seen = {*seen, marker}

    lexical = _lexical_path(node)
    if lexical and _looks_elapsed_name(lexical):
        return True
    if isinstance(node, ast.NamedExpr):
        return _target_looks_clock_derived(node.target) or _clock_derived(
            node.value, resolver, scope, seen
        )
    if isinstance(node, (ast.Name, ast.Attribute)):
        for symbol in resolver.reference(node, scope):
            classified = _real_time_kind(symbol, node)
            if classified and classified[0] in {"event_loop_time", "real_time_primitive"}:
                return True
        return any(
            _clock_derived(value, resolver, equation_scope, seen)
            for equation_scope, value in resolver.equations(node, scope)
        )
    if isinstance(node, ast.Call):
        for symbol in resolver.reference(node.func, scope):
            classified = _real_time_kind(symbol, node.func)
            if classified and classified[0] in {"event_loop_time", "real_time_primitive"}:
                return True
        return any(
            _clock_derived(argument, resolver, resolver.scope(argument), seen)
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]
        )
    if isinstance(node, ast.BoolOp):
        return any(_clock_derived(value, resolver, scope, seen) for value in node.values)
    if isinstance(node, ast.UnaryOp):
        return _clock_derived(node.operand, resolver, scope, seen)
    if isinstance(node, ast.BinOp):
        return _clock_derived(node.left, resolver, scope, seen) or _clock_derived(
            node.right, resolver, scope, seen
        )
    if isinstance(node, ast.IfExp):
        return any(
            _clock_derived(value, resolver, scope, seen)
            for value in (node.test, node.body, node.orelse)
        )
    if isinstance(node, ast.Compare):
        return any(
            _clock_derived(value, resolver, scope, seen)
            for value in (node.left, *node.comparators)
        )
    if isinstance(node, ast.Subscript):
        return _clock_derived(node.value, resolver, scope, seen) or _clock_derived(
            node.slice, resolver, scope, seen
        )
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_clock_derived(value, resolver, scope, seen) for value in node.elts)
    if isinstance(node, ast.Dict):
        return any(
            value is not None and _clock_derived(value, resolver, scope, seen)
            for value in [*node.keys, *node.values]
        )
    if isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom)) and node.value is not None:
        return _clock_derived(node.value, resolver, scope, seen)
    return False


def _compare_is_upper_bound(node: ast.Compare, resolver: _Resolver) -> bool:
    values = [node.left, *node.comparators]
    for operator, left, right in zip(node.ops, values, values[1:]):
        left_derived = _clock_derived(left, resolver)
        right_derived = _clock_derived(right, resolver)
        if isinstance(operator, (ast.Lt, ast.LtE)) and (left_derived or right_derived):
            return True
        if isinstance(operator, (ast.Gt, ast.GtE)) and (right_derived or left_derived):
            return True
        if isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) and any(
            isinstance(value, ast.Constant) and value.value == 0.14
            for value in (left, right)
        ):
            return True
    return False


def _call_is_upper_bound(node: ast.Call, resolver: _Resolver) -> bool:
    functions = resolver.reference(node.func)
    if len(node.args) >= 2:
        if functions & _LESS_THAN_FUNCTIONS and _clock_derived(node.args[0], resolver):
            return True
        if functions & _GREATER_THAN_FUNCTIONS and _clock_derived(node.args[1], resolver):
            return True
    if isinstance(node.func, ast.Attribute) and node.args:
        if node.func.attr in {"__lt__", "__le__"} and _clock_derived(
            node.func.value, resolver
        ):
            return True
        if node.func.attr in {"__gt__", "__ge__"} and _clock_derived(
            node.args[0], resolver
        ):
            return True
    return False


def _not_is_upper_bound(node: ast.UnaryOp, resolver: _Resolver) -> bool:
    if not isinstance(node.op, ast.Not) or not isinstance(node.operand, ast.Compare):
        return False
    values = [node.operand.left, *node.operand.comparators]
    for operator, left, right in zip(node.operand.ops, values, values[1:]):
        if isinstance(operator, (ast.Gt, ast.GtE)) and _clock_derived(left, resolver):
            return True
        if isinstance(operator, (ast.Lt, ast.LtE)) and _clock_derived(right, resolver):
            return True
    return False


def _scan_wall_clock_bounds(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    for node in sorted(ast.walk(method), key=lambda item: getattr(item, "lineno", 0)):
        forbidden = (
            isinstance(node, ast.Compare) and _compare_is_upper_bound(node, resolver)
        ) or (
            isinstance(node, ast.Call) and _call_is_upper_bound(node, resolver)
        ) or (
            isinstance(node, ast.UnaryOp) and _not_is_upper_bound(node, resolver)
        )
        if forbidden:
            _append_unique(
                findings,
                seen,
                _finding(
                    "wall_clock_bound",
                    node,
                    text,
                    "forbidden recursively clock-derived less-than wall-clock bound",
                ),
            )


def _scan_escape_expression(
    expression: ast.AST,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
    *,
    references: bool,
) -> None:
    for node, symbols in _iter_references(expression, resolver):
        if not references and not isinstance(node, ast.Call):
            continue
        for symbol in sorted(symbols):
            classified = _escape_kind(symbol)
            if classified:
                kind, detail = classified
                _append_unique(findings, seen, _finding(kind, node, text, detail))
                break


def _is_pytestmark_target(target: ast.AST) -> bool:
    return any(path.rsplit(".", 1)[-1] == "pytestmark" for path in _assignment_paths(target))


def _scan_pytestmark_assignments(
    scopes: set[_Scope],
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    for node, scope in resolver.index.scope_by_node.items():
        if scope not in scopes:
            continue
        value: ast.AST | None = None
        targets: tuple[ast.AST, ...] = ()
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
            value = node.value
        elif isinstance(node, ast.AugAssign):
            targets = (node.target,)
            value = node.value
        if value is not None and any(_is_pytestmark_target(target) for target in targets):
            _scan_escape_expression(
                value,
                text,
                resolver,
                findings,
                seen,
                references=True,
            )


def _scan_applicability(
    tree: ast.Module,
    target_class: ast.ClassDef,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    module_scope = resolver.index.root
    class_scope = resolver.index.owner_scope[target_class]
    method_scope = resolver.index.owner_scope[method]

    for decorator in [*target_class.decorator_list, *method.decorator_list]:
        _scan_escape_expression(
            decorator,
            text,
            resolver,
            findings,
            seen,
            references=True,
        )
    for default in [*method.args.defaults, *method.args.kw_defaults]:
        if default is not None:
            _scan_escape_expression(
                default,
                text,
                resolver,
                findings,
                seen,
                references=True,
            )
    for statement in method.body:
        _scan_escape_expression(
            statement,
            text,
            resolver,
            findings,
            seen,
            references=True,
        )

    for statement in tree.body:
        if not isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            _scan_escape_expression(
                statement,
                text,
                resolver,
                findings,
                seen,
                references=False,
            )
    for statement in target_class.body:
        if not isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            _scan_escape_expression(
                statement,
                text,
                resolver,
                findings,
                seen,
                references=False,
            )

    _scan_pytestmark_assignments(
        {module_scope, class_scope, method_scope},
        text,
        resolver,
        findings,
        seen,
    )


def _audit_text(text: str, target: str) -> dict[str, object]:
    tree = ast.parse(text, filename=target)
    resolver = _Resolver(tree)
    findings: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()

    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == _TARGET_CLASS
    ]
    target_class: ast.ClassDef | None = None
    method: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    if not classes:
        findings.append(
            {
                "kind": "missing_target",
                "line": None,
                "detail": f"missing {_TARGET_CLASS}.{_TARGET_TEST}",
                "source": "",
            }
        )
    elif len(classes) != 1:
        findings.append(
            _finding(
                "duplicate_target_class",
                classes[1] if len(classes) > 1 else classes[0],
                text,
                f"expected exactly one {_TARGET_CLASS}, found {len(classes)}",
            )
        )
    else:
        target_class = classes[0]
        methods = [
            child
            for child in target_class.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == _TARGET_TEST
        ]
        if not methods:
            findings.append(
                _finding(
                    "missing_target",
                    target_class,
                    text,
                    f"missing direct method {_TARGET_CLASS}.{_TARGET_TEST}",
                )
            )
        elif len(methods) != 1:
            findings.append(
                _finding(
                    "duplicate_target_method",
                    methods[1] if len(methods) > 1 else methods[0],
                    text,
                    f"expected exactly one direct {_TARGET_TEST}, found {len(methods)}",
                )
            )
        else:
            method = methods[0]

    if target_class is not None and method is not None:
        _scan_real_time(method, text, resolver, findings, seen)
        _scan_wall_clock_bounds(method, text, resolver, findings, seen)
        _scan_applicability(
            tree,
            target_class,
            method,
            text,
            resolver,
            findings,
            seen,
        )

    findings.sort(
        key=lambda item: (item["line"] is None, item["line"] or 0, str(item["kind"]))
    )
    return {
        "audit": "codex-timeout-test-determinism",
        "target": target,
        "class": _TARGET_CLASS,
        "test": _TARGET_TEST,
        "forbidden_real_time_primitives": findings,
        "forbidden_count": len(findings),
        "passed": not findings,
    }


def _fixture(
    body: str = "pass",
    *,
    imports: str = "",
    module_code: str = "",
    class_code: str = "",
    class_decorators: str = "",
    method_decorators: str = "",
    parameters: str = "self",
) -> str:
    lines: list[str] = []
    lines.extend(line for line in imports.splitlines() if line.strip())
    lines.extend(line for line in module_code.splitlines() if line.strip())
    lines.extend(line for line in class_decorators.splitlines() if line.strip())
    lines.append(f"class {_TARGET_CLASS}:")
    lines.extend(f"    {line}" for line in class_code.splitlines() if line.strip())
    lines.extend(f"    {line}" for line in method_decorators.splitlines() if line.strip())
    lines.append(f"    def {_TARGET_TEST}({parameters}):")
    lines.extend(textwrap.indent(body, "        ").splitlines())
    return "\n".join(lines) + "\n"


def _duplicate_class_fixture() -> str:
    return textwrap.dedent(
        f"""\
        class {_TARGET_CLASS}:
            def {_TARGET_TEST}(self):
                pass

        class {_TARGET_CLASS}:
            def {_TARGET_TEST}(self):
                pass
        """
    )


def _duplicate_method_fixture() -> str:
    return textwrap.dedent(
        f"""\
        class {_TARGET_CLASS}:
            def {_TARGET_TEST}(self):
                pass

            def {_TARGET_TEST}(self):
                pass
        """
    )


def _scoped_alias_fixture() -> str:
    return _fixture(
        "import time as clock\nclock.sleep(0.01)",
        module_code="def unrelated():\n    import decimal as clock",
    )


_SELF_TEST_CASES = (
    ("from_time_sleep_alias", _fixture("nap(0.03)", imports="from time import sleep as nap"), False, ("real_time_primitive",)),
    ("module_time_alias_sleep", _fixture("clock.sleep(0.03)", imports="import time as clock"), False, ("real_time_primitive",)),
    ("from_time_monotonic_alias", _fixture("tick()", imports="from time import monotonic as tick"), False, ("real_time_primitive",)),
    ("module_threading_timer_alias", _fixture("th.Timer(1, lambda: None)", imports="import threading as th"), False, ("real_timer_constructor",)),
    ("from_threading_timer_alias", _fixture("Alarm(1, lambda: None)", imports="from threading import Timer as Alarm"), False, ("real_timer_constructor",)),
    ("assignment_alias_sleep", _fixture("sleeper = time.sleep\nsleeper(0.01)", imports="import time"), False, ("real_time_primitive",)),
    ("chained_assignment_alias", _fixture("first = time.sleep\nsecond = first\nsecond(0.01)", imports="import time"), False, ("real_time_primitive",)),
    ("attribute_assignment_alias", _fixture("holder.sleep = time.sleep\nholder.sleep(0.01)", imports="import time"), False, ("real_time_primitive",)),
    ("scoped_import_alias", _scoped_alias_fixture(), False, ("real_time_primitive",)),
    ("dynamic_import_sleep", _fixture('__import__("time").sleep(0.01)'), False, ("real_time_primitive",)),
    ("dynamic_import_alias_chain", _fixture('loader = __import__\nattribute = "sleep"\nmodule = loader("time")\nsleeper = getattr(module, attribute)\nsleeper(0.01)'), False, ("real_time_primitive",)),
    ("getattr_sleep", _fixture('getattr(time, "sleep")(0.01)', imports="import time"), False, ("real_time_primitive",)),
    ("partial_sleep", _fixture("sleeper = partial(time.sleep, 0.01)\nsleeper()", imports="import time\nfrom functools import partial"), False, ("real_time_primitive",)),
    ("executor_sleep_reference", _fixture("executor.submit(time.sleep, 0.01)", imports="import time"), False, ("real_time_primitive",)),
    ("clock_gettime", _fixture("time.clock_gettime(time.CLOCK_MONOTONIC)", imports="import time"), False, ("real_time_primitive",)),
    ("timeit_default_timer", _fixture("timeit.default_timer()", imports="import timeit"), False, ("real_time_primitive",)),
    ("threading_event_wait", _fixture("threading.Event().wait(0.01)", imports="import threading"), False, ("real_time_primitive",)),
    ("asyncio_wait_for", _fixture("asyncio.wait_for(value, timeout=0.01)", imports="import asyncio"), False, ("real_time_primitive",)),
    ("signal_setitimer", _fixture("signal.setitimer(signal.ITIMER_REAL, 0.01)", imports="import signal"), False, ("real_timer_constructor",)),
    ("aliased_pytest_skip", _fixture('omit("skip")', imports="from pytest import skip as omit"), False, ("skip_or_xfail",)),
    ("module_skip_alias", _fixture("pass", imports="import pytest as pt", module_code='pt.skip("module")'), False, ("skip_or_xfail",)),
    ("module_pytestmark", _fixture("pass", imports="import pytest as pt", module_code="pytestmark = pt.mark.skip"), False, ("skip_or_xfail",)),
    ("module_annotated_pytestmark", _fixture("pass", imports="import pytest", module_code="pytestmark: object = pytest.mark.skip"), False, ("skip_or_xfail",)),
    ("module_augmented_pytestmark", _fixture("pass", imports="import pytest", module_code="pytestmark = []\npytestmark += [pytest.mark.skip]"), False, ("skip_or_xfail",)),
    ("class_pytestmark", _fixture("pass", imports="import pytest", class_code="pytestmark = pytest.mark.skip"), False, ("skip_or_xfail",)),
    ("method_pytestmark", _fixture("pytestmark = pytest.mark.skip", imports="import pytest"), False, ("skip_or_xfail",)),
    ("class_skip_decorator", _fixture("pass", imports="import pytest", class_decorators="@pytest.mark.skip"), False, ("skip_or_xfail",)),
    ("method_called_skipif", _fixture("pass", imports="import pytest", method_decorators="@pytest.mark.skipif(True)"), False, ("skip_or_xfail",)),
    ("method_called_xfail", _fixture("pass", imports="import pytest", method_decorators='@pytest.mark.xfail(reason="x")'), False, ("skip_or_xfail",)),
    ("assigned_decorator_alias", _fixture("pass", imports="import pytest", class_code="omit = pytest.mark.skip", method_decorators="@omit"), False, ("skip_or_xfail",)),
    ("parameter_skip_mark", _fixture("pass", imports="import pytest", parameters="self, case=pytest.mark.skip"), False, ("skip_or_xfail",)),
    ("called_flaky", _fixture("pass", method_decorators="@flaky(max_runs=2)"), False, ("retry_or_repeat",)),
    ("called_retry", _fixture("pass", method_decorators="@retry(2)"), False, ("retry_or_repeat",)),
    ("called_retries", _fixture("pass", method_decorators="@retries(2)"), False, ("retry_or_repeat",)),
    ("called_rerun", _fixture("pass", method_decorators="@rerun(2)"), False, ("retry_or_repeat",)),
    ("called_repeat", _fixture("pass", method_decorators="@repeat(2)"), False, ("retry_or_repeat",)),
    ("asyncio_sleep", _fixture("asyncio.sleep(0.03)", imports="import asyncio"), False, ("real_time_primitive",)),
    ("event_loop_time", _fixture("event_loop.time()"), False, ("event_loop_time",)),
    ("datetime_now_alias", _fixture("DateTime.now()", imports="from datetime import datetime as DateTime"), False, ("real_time_primitive",)),
    ("changed_elapsed_bound", _fixture("elapsed = 0.2\nassert elapsed < 0.2"), False, ("wall_clock_bound",)),
    ("walrus_elapsed_bound", _fixture("assert (elapsed := 0.1) < 0.2"), False, ("wall_clock_bound",)),
    ("boolop_elapsed_bound", _fixture("elapsed = 0.1\nassert (elapsed or 0.0) < 0.2"), False, ("wall_clock_bound",)),
    ("unary_elapsed_bound", _fixture("elapsed = 0.1\nassert -elapsed < 0.2"), False, ("wall_clock_bound",)),
    ("binop_elapsed_bound", _fixture("elapsed = 0.1\nassert elapsed + 0.01 < 0.2"), False, ("wall_clock_bound",)),
    ("wrapped_elapsed_bound", _fixture("elapsed = 0.1\nassert min(elapsed, 0.15) < 0.2"), False, ("wall_clock_bound",)),
    ("operator_elapsed_bound", _fixture("elapsed = 0.1\nassert operator.lt(elapsed, 0.2)", imports="import operator"), False, ("wall_clock_bound",)),
    ("dunder_elapsed_bound", _fixture("elapsed = 0.1\nassert elapsed.__lt__(0.2)"), False, ("wall_clock_bound",)),
    ("negated_elapsed_bound", _fixture("elapsed = 0.1\nassert not elapsed >= 0.2"), False, ("wall_clock_bound",)),
    ("duplicate_target_class", _duplicate_class_fixture(), False, ("duplicate_target_class",)),
    ("duplicate_target_method", _duplicate_method_fixture(), False, ("duplicate_target_method",)),
    ("safe_virtual_clock", _fixture(textwrap.dedent("""\
        class _VirtualClock:
            def monotonic(self):
                return 0.0
        clock = _VirtualClock()
        assert clock.monotonic() == 0.0
    """)), True, ()),
    ("safe_scoped_shadow", _fixture(textwrap.dedent("""\
        class _VirtualClock:
            def monotonic(self):
                return 0.0
        clock = _VirtualClock()
        assert clock.monotonic() == 0.0
    """), imports="import time as clock"), True, ()),
)


def _run_self_tests() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    all_passed = True
    for name, text, expected_passed, expected_kinds in _SELF_TEST_CASES:
        try:
            result = _audit_text(text, f"<self-test:{name}>")
            actual_passed = bool(result["passed"])
            finding_list = result["forbidden_real_time_primitives"]
            if not isinstance(finding_list, list):
                raise TypeError("audit findings must be a list")
            actual_kinds = sorted(
                {
                    str(finding.get("kind"))
                    for finding in finding_list
                    if isinstance(finding, dict)
                }
            )
            case_passed = actual_passed == expected_passed and set(
                expected_kinds
            ).issubset(actual_kinds)
            case = {
                "name": name,
                "expected_passed": expected_passed,
                "actual_passed": actual_passed,
                "expected_kinds": list(expected_kinds),
                "actual_kinds": actual_kinds,
                "forbidden_count": result["forbidden_count"],
                "passed": case_passed,
            }
        except (SyntaxError, TypeError, ValueError) as exc:
            case_passed = False
            case = {
                "name": name,
                "expected_passed": expected_passed,
                "actual_passed": None,
                "expected_kinds": list(expected_kinds),
                "actual_kinds": [],
                "forbidden_count": 1,
                "passed": False,
                "error": str(exc),
            }
        cases.append(case)
        all_passed = all_passed and case_passed
    return {"passed": all_passed, "count": len(cases), "cases": cases}


def _audit(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    result = _audit_text(text, str(path))
    self_tests = _run_self_tests()
    result["self_tests"] = self_tests
    result["passed"] = bool(result["passed"] and self_tests["passed"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests"
        / "agent"
        / "test_auxiliary_client.py",
    )
    parser.add_argument("--json", type=Path, required=True, metavar="ARTIFACT")
    args = parser.parse_args()

    try:
        result = _audit(args.target)
    except (OSError, SyntaxError, UnicodeError) as exc:
        result = {
            "audit": "codex-timeout-test-determinism",
            "target": str(args.target),
            "forbidden_real_time_primitives": [
                {
                    "kind": "audit_error",
                    "line": None,
                    "detail": str(exc),
                    "source": "",
                }
            ],
            "forbidden_count": 1,
            "passed": False,
            "self_tests": _run_self_tests(),
        }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
