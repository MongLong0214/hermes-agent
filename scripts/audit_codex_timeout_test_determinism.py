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
_PROVEN_COLLECTION_CALLS = {
    "_aux_mod._aux_unhealthy_logged_at.clear",
    "_aux_mod._aux_unhealthy_until.clear",
    "agent.auxiliary_client._aux_unhealthy_logged_at.clear",
    "agent.auxiliary_client._aux_unhealthy_until.clear",
    "SimpleNamespace",
    "classmethod",
    "len",
    "monkeypatch.delenv",
    "patch.object",
    "pytest.fixture",
    "pytest.mark.parametrize",
    "pytest.raises",
    "range",
    "staticmethod",
}

# Positive structural contract for the owned target.  The audit does not try to
# prove arbitrary Python dataflow.  Instead, the target may use only the names
# and attributes already needed by the deterministic virtual-clock oracle.
# Any new symbol or higher-order shape fails closed until it is reviewed and
# explicitly added here.
_APPROVED_TARGET_NAMES = {
    "FakeResponses",
    "SimpleNamespace",
    "TimeoutError",
    "_CodexCompletionsAdapter",
    "_ObservableTimer",
    "_SlowAliveCreateStream",
    "_VirtualClock",
    "adapter",
    "aux",
    "case",
    "client_close_count",
    "clock",
    "consumed_events",
    "event_number",
    "fake_client",
    "function",
    "instances",
    "interval",
    "kwargs",
    "len",
    "milliseconds",
    "patch",
    "pytest",
    "range",
    "request_kwargs",
    "self",
    "stream_close_count",
    "timer",
}
_APPROVED_TARGET_ATTRIBUTES = {
    "__class__",
    "advance",
    "append",
    "cancelled",
    "create",
    "function",
    "instances",
    "interval",
    "mark",
    "monotonic",
    "now_ms",
    "object",
    "parametrize",
    "raises",
    "started",
    "threading",
    "time",
    "timestamps_ms",
    "update",
}
_APPROVED_NESTED_DEFINITIONS = {
    "FakeResponses",
    "_ObservableTimer",
    "_SlowAliveCreateStream",
    "_VirtualClock",
    "__init__",
    "__iter__",
    "advance",
    "cancel",
    "close",
    "create",
    "monotonic",
    "start",
}
_APPROVED_ARGUMENT_NAMES = {
    "case",
    "function",
    "interval",
    "kwargs",
    "milliseconds",
    "self",
}
_APPROVED_METHOD_DECORATOR = ast.dump(
    ast.parse('@pytest.mark.parametrize("case", range(1000))\ndef _target(self, case):\n    pass\n').body[0].decorator_list[0],
    include_attributes=False,
)
_APPROVED_METHOD_ARGUMENTS = ast.dump(
    ast.parse("def _target(self, case):\n    pass\n").body[0].args,
    include_attributes=False,
)
_APPROVED_PATCH_CALLS = {
    "monotonic": ast.dump(
        ast.parse('patch.object(aux.time, "monotonic", clock.monotonic)', mode="eval").body,
        include_attributes=False,
    ),
    "timer": ast.dump(
        ast.parse('patch.object(aux.threading, "Timer", _ObservableTimer)', mode="eval").body,
        include_attributes=False,
    ),
}
_APPROVED_AUX_IMPORT = ast.dump(
    ast.parse("import agent.auxiliary_client as aux").body[0],
    include_attributes=False,
)
# There are intentionally no ordering expressions in the current target.
# Future safe order expressions require an exact AST occurrence here, rather
# than another elapsed-dataflow heuristic.
_APPROVED_ORDER_EXPRESSIONS: frozenset[str] = frozenset()
_ORDER_HELPER_ATTRIBUTES = {
    "__ge__",
    "__gt__",
    "__le__",
    "__lt__",
    "ge",
    "gt",
    "le",
    "lt",
    "max",
    "min",
    "sorted",
}

# Names that make the positive contract meaningful.  Their provenance is part
# of the contract: a lexical ``pytest``/``range``/adapter spelling is not
# enough if the module can replace the binding before pytest evaluates it.
_PROTECTED_CONTRACT_ROOTS = {
    "pytest": {"pytest"},
    "range": {"range"},
    "patch": {"unittest.mock.patch"},
    "SimpleNamespace": {"types.SimpleNamespace"},
    "_CodexCompletionsAdapter": {
        "agent.auxiliary_client._CodexCompletionsAdapter"
    },
    "aux": {"agent.auxiliary_client"},
    "len": {"len"},
    "TimeoutError": {"TimeoutError"},
    "object": {"object"},
}
_PROTECTED_IMPORTS = {
    "pytest": {"pytest"},
    "patch": {"unittest.mock.patch"},
    "SimpleNamespace": {"types.SimpleNamespace"},
    "_CodexCompletionsAdapter": {
        "agent.auxiliary_client._CodexCompletionsAdapter"
    },
}
_MODULE_XUNIT_HOOKS = {
    "pytest_generate_tests",
    "setup_module",
    "teardown_module",
    "setup_function",
    "teardown_function",
}
_CLASS_XUNIT_HOOKS = {
    "pytest_generate_tests",
    "setup_class",
    "teardown_class",
    "setup_method",
    "teardown_method",
}


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


class _CallableSummary:
    """Bounded static summary of a local callable's possible returned callables."""

    def __init__(
        self,
        returned: set[ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
        *,
        unresolved: bool = False,
        non_callable: bool = False,
        applied: set[int] | None = None,
        cyclic: bool = False,
    ) -> None:
        self.returned = returned or set()
        self.unresolved = unresolved
        self.non_callable = non_callable
        self.applied = applied or set()
        self.cyclic = cyclic


class _Resolver:
    """Scope-aware fixed-point resolver for imports and value aliases."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.index = _ScopeIndex(tree)
        self._definition_nodes: dict[
            str, set[ast.FunctionDef | ast.AsyncFunctionDef]
        ] = {}
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
                marker = f"<definition:{node.name}@{scope.identifier}>"
                scope.add_binding(node.name, marker)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._definition_nodes.setdefault(
                        marker, set()
                    ).add(node)
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

    def local_definitions(
        self,
        node: ast.AST,
        scope: _Scope | None = None,
    ) -> set[ast.FunctionDef | ast.AsyncFunctionDef]:
        """Return function definitions proven reachable from a callable path."""
        scope = scope or self.scope(node)
        definitions: set[ast.FunctionDef | ast.AsyncFunctionDef] = set()
        for symbol in self.reference(node, scope):
            definitions.update(self._definition_nodes.get(symbol, set()))
        return definitions

    def _summarize_definitions(
        self,
        definitions: set[ast.FunctionDef | ast.AsyncFunctionDef],
        stack: set[tuple[int, int]] | None = None,
        *,
        call_layer: int,
        applied: set[int] | None = None,
    ) -> _CallableSummary:
        stack = stack or set()
        applied = applied or set()
        summary = _CallableSummary()
        for definition in definitions:
            marker = (id(definition), call_layer)
            if marker in stack:
                summary.unresolved = True
                continue
            if id(definition) in applied:
                summary.cyclic = True
            nested = self.callable_summary(
                definition,
                {*stack, marker},
                {*applied, id(definition)},
            )
            summary.returned.update(nested.returned)
            summary.unresolved = summary.unresolved or nested.unresolved
            summary.non_callable = summary.non_callable or nested.non_callable
            summary.applied.update(nested.applied)
            summary.applied.add(id(definition))
            summary.cyclic = summary.cyclic or nested.cyclic
        return summary

    def returned_callable_summary(
        self,
        call: ast.Call,
        scope: _Scope | None = None,
        stack: set[tuple[int, int]] | None = None,
        applied: set[int] | None = None,
    ) -> _CallableSummary:
        """Resolve each immediately applied local call layer without executing it."""
        scope = scope or self.scope(call)
        if isinstance(call.func, ast.Call):
            returned = self.returned_callable_summary(
                call.func,
                self.scope(call.func),
                stack,
                applied,
            )
            if returned.unresolved or returned.non_callable or not returned.returned:
                return returned
            summarized = self._summarize_definitions(
                returned.returned,
                stack,
                call_layer=id(call),
                applied=returned.applied,
            )
            summarized.cyclic = summarized.cyclic or returned.cyclic
            return summarized
        definitions = self.local_definitions(call.func, self.scope(call.func))
        if not definitions:
            return _CallableSummary(unresolved=True)
        return self._summarize_definitions(
            definitions,
            stack,
            call_layer=id(call),
            applied=applied,
        )

    def callable_summary(
        self,
        definition: ast.FunctionDef | ast.AsyncFunctionDef,
        stack: set[tuple[int, int]] | None = None,
        applied: set[int] | None = None,
    ) -> _CallableSummary:
        """Summarize direct return statements while excluding nested callable bodies."""
        stack = stack or set()
        applied = applied or set()
        summary = _CallableSummary(applied=set(applied))
        saw_return = False
        for node in _iter_runtime_nodes(definition):
            if not isinstance(node, ast.Return):
                continue
            saw_return = True
            if node.value is None:
                summary.non_callable = True
                continue
            returned = self.local_definitions(node.value, self.scope(node.value))
            if returned:
                summary.returned.update(returned)
                continue
            if isinstance(node.value, ast.Call):
                nested = self.returned_callable_summary(
                    node.value,
                    self.scope(node.value),
                    stack,
                    summary.applied,
                )
                summary.returned.update(nested.returned)
                summary.unresolved = summary.unresolved or nested.unresolved
                summary.non_callable = summary.non_callable or nested.non_callable
                summary.applied.update(nested.applied)
                summary.cyclic = summary.cyclic or nested.cyclic
                continue
            if _statically_non_callable(node.value):
                summary.non_callable = True
            else:
                summary.unresolved = True
        if not saw_return:
            summary.non_callable = True
        return summary


def _lexical_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _statically_non_callable(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return isinstance(node.operand, ast.Constant)
    return False


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


def _scan_protected_contract_bindings(
    tree: ast.Module,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    """Require every approved root to retain its reviewed binding identity."""
    imported: dict[str, set[str]] = {name: set() for name in _PROTECTED_IMPORTS}
    module_scope = resolver.index.root
    target_class = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == _TARGET_CLASS
        ),
        None,
    )
    class_scope = (
        resolver.index.owner_scope.get(target_class, module_scope)
        if target_class is not None
        else module_scope
    )
    method_scope = resolver.index.owner_scope.get(method, module_scope)
    applicable_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            resolver.scope(node) is module_scope
            or resolver.scope(node) is class_scope
        )
        and _definition_body_is_applicable(
            node, resolver, class_scope=resolver.scope(node) is class_scope
        )
    ]
    relevant_scopes = {module_scope, class_scope, method_scope}
    relevant_scopes.update(
        resolver.index.owner_scope.get(node, module_scope)
        for node in applicable_functions
    )

    for node in ast.walk(tree):
        if resolver.scope(node) not in relevant_scopes:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                canonical = alias.name if alias.asname else bound
                if bound in imported:
                    imported[bound].add(canonical)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                if alias.name == "*":
                    _positive_finding(
                        findings,
                        seen,
                        "binding_contract",
                        node,
                        text,
                        "wildcard import leaves protected contract provenance ambiguous",
                    )
                    continue
                bound = alias.asname or alias.name
                canonical = ".".join(
                    part for part in (module, alias.name) if part
                )
                if bound in imported:
                    imported[bound].add(canonical)

    for name, expected in _PROTECTED_IMPORTS.items():
        if imported[name] != expected:
            _positive_finding(
                findings,
                seen,
                "binding_contract",
                method,
                text,
                f"{name} must have reviewed import provenance {sorted(expected)}; "
                f"found {sorted(imported[name])}",
            )

    checked_roots: set[int] = set()
    for node in ast.walk(method):
        candidate: ast.AST | None = node.func if isinstance(node, ast.Call) else node
        if not isinstance(candidate, (ast.Name, ast.Attribute)):
            continue
        root_node = candidate
        while isinstance(root_node, ast.Attribute):
            root_node = root_node.value
        if not isinstance(root_node, ast.Name):
            continue
        expected = _PROTECTED_CONTRACT_ROOTS.get(root_node.id)
        if expected is None or id(root_node) in checked_roots:
            continue
        checked_roots.add(id(root_node))
        actual = resolver.reference(root_node, resolver.scope(root_node))
        if actual != expected:
            _positive_finding(
                findings,
                seen,
                "binding_contract",
                root_node,
                text,
                f"protected contract root {root_node.id} resolves to "
                f"{sorted(actual)}, expected {sorted(expected)}",
            )

    for node, scope in resolver.index.scope_by_node.items():
        if scope not in relevant_scopes:
            continue
        name: str | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name = node.id
        if name in _PROTECTED_CONTRACT_ROOTS:
            _positive_finding(
                findings,
                seen,
                "binding_contract",
                node,
                text,
                f"protected contract root is rebound: {name}",
            )

    for function in applicable_functions:
        declarations = {
            name
            for statement in ast.walk(function)
            if isinstance(statement, (ast.Global, ast.Nonlocal))
            for name in statement.names
        }
        for node in ast.walk(function):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Store):
                continue
            if node.id not in declarations or node.id not in _PROTECTED_CONTRACT_ROOTS:
                continue
            _positive_finding(
                findings,
                seen,
                "binding_contract",
                node,
                text,
                f"active applicability body writes protected root {node.id} through global/nonlocal binding",
            )


def _iter_runtime_nodes(root: ast.AST):
    """Walk executable expressions without entering uncalled callable bodies."""
    pending: list[ast.AST] = [root]
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, ast.Lambda):
            children = [node.args]
        else:
            children = list(ast.iter_child_nodes(node))
        for child in reversed(children):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            pending.append(child)


def _iter_runtime_references(root: ast.AST, resolver: _Resolver):
    for node in _iter_runtime_nodes(root):
        if _maximal_reference(node, resolver):
            yield node, resolver.reference(node)


def _annotation_nodes(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    arguments = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *([node.args.vararg] if node.args.vararg else []),
        *([node.args.kwarg] if node.args.kwarg else []),
    ]
    return [
        *[argument.annotation for argument in arguments if argument.annotation is not None],
        *([node.returns] if node.returns is not None else []),
    ]


def _annotations_are_evaluated(tree: ast.Module) -> bool:
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom) or statement.module != "__future__":
            continue
        if any(alias.name == "annotations" for alias in statement.names):
            return False
    return True


def _fixture_autouse_status(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    resolver: _Resolver,
) -> bool | None:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if resolver.reference(decorator.func, resolver.scope(decorator.func)) != {
            "pytest.fixture"
        }:
            continue
        values = [keyword.value for keyword in decorator.keywords if keyword.arg == "autouse"]
        if not values:
            return False
        try:
            return bool(ast.literal_eval(values[-1]))
        except (ValueError, TypeError, SyntaxError):
            return None
    return False


def _is_autouse_fixture(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    resolver: _Resolver,
) -> bool:
    return _fixture_autouse_status(node, resolver) is True


def _scan_unresolved_collection_call(
    node: ast.Call,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    symbols = resolver.reference(node.func, resolver.scope(node.func))
    symbols.update(path for path in [_lexical_path(node.func)] if path)
    if symbols & _PROVEN_COLLECTION_CALLS:
        return
    if any(_real_time_kind(symbol, node.func) for symbol in symbols):
        return
    if any(_escape_kind(symbol) for symbol in symbols):
        return
    _positive_finding(
        findings,
        seen,
        "applicability_contract",
        node,
        text,
        "collection-time call is not resolved to a reviewed local callable or safe primitive",
    )


def _call_result_is_decorator_application(call: ast.Call, resolver: _Resolver) -> bool:
    parent = resolver.parent(call)
    return isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(
        decorator is call for decorator in parent.decorator_list
    )


def _lambda_is_decorator_application(node: ast.Lambda, resolver: _Resolver) -> bool:
    parent = resolver.parent(node)
    return isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(
        decorator is node for decorator in parent.decorator_list
    )


def _scan_unresolved_returned_callable(
    call: ast.Call,
    text: str,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    _positive_finding(
        findings,
        seen,
        "applicability_contract",
        call,
        text,
        "immediately applied local call does not resolve exclusively to returned local callables",
    )


def _scan_applicability_expression(
    expression: ast.AST,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
    *,
    annotations_evaluated: bool = True,
    callable_stack: set[int] | None = None,
) -> None:
    callable_stack = callable_stack or set()
    for node, symbols in _iter_runtime_references(expression, resolver):
        for symbol in sorted(symbols):
            classified = _escape_kind(symbol) or _real_time_kind(symbol, node)
            if classified:
                kind, detail = classified
                _append_unique(findings, seen, _finding(kind, node, text, detail))
                break

    callable_nodes: set[ast.FunctionDef | ast.AsyncFunctionDef] = set()
    for node in _iter_runtime_nodes(expression):
        if node is not expression and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            _scan_definition_import_time(
                node,
                text,
                resolver,
                findings,
                seen,
                scan_body=False,
                scan_nested_bodies=False,
                annotations_evaluated=annotations_evaluated,
                callable_stack=callable_stack,
            )
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Call):
            returned = resolver.returned_callable_summary(node.func)
            definitions = set(returned.returned)
            if returned.unresolved or returned.non_callable or not definitions:
                _scan_unresolved_returned_callable(node, text, findings, seen)
        else:
            definitions = resolver.local_definitions(
                node.func,
                resolver.scope(node.func),
            )
        if not definitions:
            _scan_unresolved_collection_call(node, text, resolver, findings, seen)
        callable_nodes.update(definitions)
        if isinstance(node.func, ast.Lambda):
            _scan_applicability_expression(
                node.func.body,
                text,
                resolver,
                findings,
                seen,
                annotations_evaluated=annotations_evaluated,
                callable_stack=callable_stack,
            )

        if _call_result_is_decorator_application(node, resolver) and definitions:
            returned = resolver.returned_callable_summary(node)
            if returned.cyclic or returned.unresolved or returned.non_callable or not returned.returned:
                _scan_unresolved_returned_callable(node, text, findings, seen)
            callable_nodes.update(returned.returned)
    if isinstance(expression, ast.Lambda) and _lambda_is_decorator_application(
        expression, resolver
    ):
        _scan_applicability_expression(
            expression.body,
            text,
            resolver,
            findings,
            seen,
            annotations_evaluated=annotations_evaluated,
            callable_stack=callable_stack,
        )
    if isinstance(expression, (ast.Name, ast.Attribute, ast.Call)):
        callable_node = expression.func if isinstance(expression, ast.Call) else expression
        callable_nodes.update(
            resolver.local_definitions(callable_node, resolver.scope(callable_node))
        )

    for node in callable_nodes:
        if id(node) in callable_stack:
            continue
        _scan_definition_import_time(
            node,
            text,
            resolver,
            findings,
            seen,
            scan_body=True,
            annotations_evaluated=annotations_evaluated,
            callable_stack={*callable_stack, id(node)},
        )


def _definition_body_is_applicable(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    resolver: _Resolver,
    *,
    class_scope: bool = False,
) -> bool:
    hooks = _CLASS_XUNIT_HOOKS if class_scope else _MODULE_XUNIT_HOOKS
    return node.name in hooks or _is_autouse_fixture(node, resolver)


def _scan_definition_import_time(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
    *,
    scan_body: bool,
    scan_nested_bodies: bool = True,
    annotations_evaluated: bool = True,
    callable_stack: set[int] | None = None,
) -> None:
    callable_stack = callable_stack or set()
    expressions: list[ast.AST] = [*node.decorator_list]
    if isinstance(node, ast.ClassDef):
        expressions.extend([*node.bases, *(keyword.value for keyword in node.keywords)])
    else:
        expressions.extend(
            default
            for default in [*node.args.defaults, *node.args.kw_defaults]
            if default is not None
        )
        if annotations_evaluated:
            expressions.extend(_annotation_nodes(node))
        autouse_status = _fixture_autouse_status(node, resolver)
        if autouse_status is None:
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and any(
                    keyword.arg == "autouse" for keyword in decorator.keywords
                ):
                    _positive_finding(
                        findings,
                        seen,
                        "applicability_contract",
                        decorator,
                        text,
                        "fixture autouse value is present but not statically resolvable",
                    )
                    break
    for expression in expressions:
        _scan_applicability_expression(
            expression,
            text,
            resolver,
            findings,
            seen,
            annotations_evaluated=annotations_evaluated,
            callable_stack=callable_stack,
        )

    if isinstance(node, ast.ClassDef):
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _scan_definition_import_time(
                    statement,
                    text,
                    resolver,
                    findings,
                    seen,
                    scan_body=scan_nested_bodies
                    and _definition_body_is_applicable(
                        statement, resolver, class_scope=True
                    ),
                    annotations_evaluated=annotations_evaluated,
                    callable_stack=callable_stack,
                )
            elif isinstance(statement, ast.ClassDef):
                _scan_definition_import_time(
                    statement,
                    text,
                    resolver,
                    findings,
                    seen,
                    scan_body=False,
                    scan_nested_bodies=False,
                    annotations_evaluated=annotations_evaluated,
                    callable_stack=callable_stack,
                )
            else:
                _scan_applicability_expression(
                    statement,
                    text,
                    resolver,
                    findings,
                    seen,
                    annotations_evaluated=annotations_evaluated,
                    callable_stack=callable_stack,
                )
    elif scan_body:
        if node.name == "pytest_generate_tests":
            _positive_finding(
                findings,
                seen,
                "applicability_contract",
                node,
                text,
                "pytest_generate_tests hook semantics are not proven; reject the definition",
            )
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _scan_definition_import_time(
                    statement,
                    text,
                    resolver,
                    findings,
                    seen,
                    scan_body=False,
                    scan_nested_bodies=False,
                    annotations_evaluated=annotations_evaluated,
                    callable_stack=callable_stack,
                )
                continue
            _scan_applicability_expression(
                statement,
                text,
                resolver,
                findings,
                seen,
                annotations_evaluated=annotations_evaluated,
                callable_stack=callable_stack,
            )


def _scan_definition_applicability(
    tree: ast.Module,
    target_class: ast.ClassDef,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    annotations_evaluated = _annotations_are_evaluated(tree)
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _scan_definition_import_time(
                statement,
                text,
                resolver,
                findings,
                seen,
                scan_body=_definition_body_is_applicable(statement, resolver),
                annotations_evaluated=annotations_evaluated,
            )
        elif isinstance(statement, ast.ClassDef):
            _scan_definition_import_time(
                statement,
                text,
                resolver,
                findings,
                seen,
                scan_body=False,
                scan_nested_bodies=statement is target_class,
                annotations_evaluated=annotations_evaluated,
            )
        else:
            _scan_applicability_expression(
                statement,
                text,
                resolver,
                findings,
                seen,
                annotations_evaluated=annotations_evaluated,
            )

    # The target class is the one class whose xunit hooks and autouse fixtures
    # can change the selected test's applicability.  Its class-body
    # expressions also execute while pytest builds the class.
    _scan_definition_import_time(
        target_class,
        text,
        resolver,
        findings,
        seen,
        scan_body=False,
        scan_nested_bodies=True,
        annotations_evaluated=annotations_evaluated,
    )


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
    _scan_definition_applicability(
        tree,
        target_class,
        text,
        resolver,
        findings,
        seen,
    )


_APPROVED_CALLABLE_PATHS = {
    "FakeResponses",
    "SimpleNamespace",
    "_CodexCompletionsAdapter",
    "_SlowAliveCreateStream",
    "_VirtualClock",
    "adapter.create",
    "client_close_count.append",
    "clock.advance",
    "clock.monotonic",
    "consumed_events.append",
    "len",
    "patch.object",
    "pytest.mark.parametrize",
    "pytest.raises",
    "range",
    "request_kwargs.update",
    "self.__class__.instances.append",
    "self.timestamps_ms.append",
    "stream_close_count.append",
}
_APPROVED_LAMBDA = ast.dump(
    ast.parse("lambda: client_close_count.append(True)", mode="eval").body,
    include_attributes=False,
)


def _parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(root)
        for child in ast.iter_child_nodes(parent)
    }


def _inside_approved_patch(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.Call) and ast.dump(
            current, include_attributes=False
        ) in _APPROVED_PATCH_CALLS.values():
            return True
        current = parents.get(current)
    return False


def _positive_finding(
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
    kind: str,
    node: ast.AST,
    text: str,
    detail: str,
) -> None:
    _append_unique(findings, seen, _finding(kind, node, text, detail))


def _immutable_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (bool, bytes, complex, float, int, str, type(None)))
    if isinstance(node, ast.Tuple):
        return all(_immutable_literal(element) for element in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, (int, float, complex)
        )
    return False


def _safe_module_literal_assignment(tree: ast.Module, statement: ast.AST) -> bool:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    if not isinstance(target, ast.Name) or target.id in _PROTECTED_CONTRACT_ROOTS:
        return False
    if not _immutable_literal(statement.value):
        return False
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == target.id
        and isinstance(node.ctx, ast.Store)
    ]
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == target.id
        and isinstance(node.ctx, ast.Load)
    ]
    return len(stores) == 1 and not loads


def _safe_fixture_alias_assignment(statement: ast.AST, resolver: _Resolver) -> bool:
    """Allow one-shot module/class aliases to the canonical fixture decorator."""
    value: ast.AST | None = None
    targets: tuple[ast.AST, ...] = ()
    if isinstance(statement, ast.Assign):
        value = statement.value
        targets = tuple(statement.targets)
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        value = statement.value
        targets = (statement.target,)
    if value is None or not targets or not isinstance(value, (ast.Name, ast.Attribute)):
        return False
    if not all(isinstance(target, ast.Name) for target in targets):
        return False
    names = [target.id for target in targets if isinstance(target, ast.Name)]
    if any(name in _PROTECTED_CONTRACT_ROOTS for name in names):
        return False
    scope = resolver.scope(statement)
    if any(len(scope.equations.get(name, ())) != 1 for name in names):
        return False
    return resolver.reference(value, resolver.scope(value)) == {"pytest.fixture"}


def _target_owned_yields(method: ast.FunctionDef | ast.AsyncFunctionDef):
    pending = list(ast.iter_child_nodes(method))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            yield node
        pending.extend(ast.iter_child_nodes(node))


def _scan_positive_structural_contract(
    tree: ast.Module,
    target_class: ast.ClassDef,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    text: str,
    resolver: _Resolver,
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    """Fail closed on any target structure outside the reviewed safe vocabulary."""
    if isinstance(method, ast.AsyncFunctionDef):
        _positive_finding(
            findings,
            seen,
            "signature_contract",
            method,
            text,
            "the target method must be a synchronous FunctionDef",
        )

    for statement in tree.body:
        if isinstance(
            statement,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ) and isinstance(statement.value.value, str):
            continue
        if _safe_module_literal_assignment(tree, statement) or _safe_fixture_alias_assignment(
            statement, resolver
        ):
            continue
        _positive_finding(
            findings,
            seen,
            "applicability_contract",
            statement,
            text,
            "module applicability permits only imports, definitions, and the module docstring",
        )

    if target_class.decorator_list:
        for decorator in target_class.decorator_list:
            _positive_finding(
                findings,
                seen,
                "decorator_contract",
                decorator,
                text,
                "the target class must remain undecorated",
            )
    if target_class.bases or target_class.keywords:
        _positive_finding(
            findings,
            seen,
            "applicability_contract",
            target_class,
            text,
            "the target class must not gain bases or metaclass keywords",
        )
    for index, statement in enumerate(target_class.body):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if _safe_fixture_alias_assignment(statement, resolver):
            continue
        _positive_finding(
            findings,
            seen,
            "applicability_contract",
            statement,
            text,
            "the target class permits only a leading docstring and direct test methods",
        )

    for yield_node in _target_owned_yields(method):
        _positive_finding(
            findings,
            seen,
            "signature_contract",
            yield_node,
            text,
            "the target method must not be a generator",
        )

    decorator_dumps = [
        ast.dump(decorator, include_attributes=False)
        for decorator in method.decorator_list
    ]
    if decorator_dumps != [_APPROVED_METHOD_DECORATOR]:
        node = method.decorator_list[0] if method.decorator_list else method
        _positive_finding(
            findings,
            seen,
            "decorator_contract",
            node,
            text,
            "expected only @pytest.mark.parametrize('case', range(1000))",
        )
    if ast.dump(method.args, include_attributes=False) != _APPROVED_METHOD_ARGUMENTS:
        _positive_finding(
            findings,
            seen,
            "signature_contract",
            method,
            text,
            "expected exact target signature (self, case) with no defaults",
        )

    method_nodes = list(ast.walk(method))
    imports = [
        node for node in method_nodes if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    exact_aux_imports = sum(
        ast.dump(node, include_attributes=False) == _APPROVED_AUX_IMPORT
        for node in imports
    )
    if exact_aux_imports != 1 or len(imports) != 1:
        node = imports[0] if imports else method
        _positive_finding(
            findings,
            seen,
            "target_import_contract",
            node,
            text,
            "expected exactly one target-local import: import agent.auxiliary_client as aux",
        )

    for node in method_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not method:
            if node.name not in _APPROVED_NESTED_DEFINITIONS:
                _positive_finding(
                    findings,
                    seen,
                    "unapproved_definition",
                    node,
                    text,
                    f"unapproved nested function: {node.name}",
                )
            if node.decorator_list or node.args.defaults or any(
                default is not None for default in node.args.kw_defaults
            ):
                _positive_finding(
                    findings,
                    seen,
                    "higher_order_contract",
                    node,
                    text,
                    "nested decorators and default captures are not approved",
                )
        elif isinstance(node, ast.ClassDef) and node is not target_class:
            if node.name not in _APPROVED_NESTED_DEFINITIONS:
                _positive_finding(
                    findings,
                    seen,
                    "unapproved_definition",
                    node,
                    text,
                    f"unapproved nested class: {node.name}",
                )
            if node.decorator_list or node.bases or node.keywords:
                _positive_finding(
                    findings,
                    seen,
                    "higher_order_contract",
                    node,
                    text,
                    "nested class decorators, bases, and keywords are not approved",
                )
        elif isinstance(node, ast.arg) and node.arg not in _APPROVED_ARGUMENT_NAMES:
            _positive_finding(
                findings,
                seen,
                "unapproved_symbol",
                node,
                text,
                f"unapproved argument symbol: {node.arg}",
            )
        elif isinstance(node, ast.Name) and node.id not in _APPROVED_TARGET_NAMES:
            _positive_finding(
                findings,
                seen,
                "unapproved_symbol",
                node,
                text,
                f"unapproved target symbol: {node.id}",
            )
        elif isinstance(node, ast.Attribute) and node.attr not in _APPROVED_TARGET_ATTRIBUTES:
            _positive_finding(
                findings,
                seen,
                "unapproved_symbol",
                node,
                text,
                f"unapproved target attribute: {node.attr}",
            )

    for node in method_nodes:
        if isinstance(node, ast.Call):
            path = _lexical_path(node.func)
            if path not in _APPROVED_CALLABLE_PATHS:
                _positive_finding(
                    findings,
                    seen,
                    "higher_order_contract",
                    node,
                    text,
                    "call target is outside the reviewed deterministic callable allowlist",
                )
        elif isinstance(node, ast.Lambda):
            if ast.dump(node, include_attributes=False) != _APPROVED_LAMBDA:
                _positive_finding(
                    findings,
                    seen,
                    "higher_order_contract",
                    node,
                    text,
                    "lambda capture is outside the single reviewed fake-client close shape",
                )
        elif isinstance(
            node,
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.NamedExpr),
        ):
            _positive_finding(
                findings,
                seen,
                "higher_order_contract",
                node,
                text,
                "comprehension, generator, or walrus indirection is not approved",
            )

    parents = _parent_map(method)
    patch_counts = {name: 0 for name in _APPROVED_PATCH_CALLS}
    for node in method_nodes:
        if isinstance(node, ast.Call):
            dump = ast.dump(node, include_attributes=False)
            for name, approved in _APPROVED_PATCH_CALLS.items():
                if dump == approved:
                    patch_counts[name] += 1
    for name, count in patch_counts.items():
        if count != 1:
            _positive_finding(
                findings,
                seen,
                "patch_contract",
                method,
                text,
                f"expected exactly one approved {name} patch.object call, found {count}",
            )

    for node in method_nodes:
        lexical = _lexical_path(node) if isinstance(node, (ast.Name, ast.Attribute)) else None
        production_reference = bool(
            lexical
            and (
                lexical == "aux.time"
                or lexical.startswith("aux.time.")
                or lexical == "aux.threading"
                or lexical.startswith("aux.threading.")
            )
        )
        patch_string = isinstance(node, ast.Constant) and node.value in {
            "Timer",
            "monotonic",
        }
        if (production_reference or patch_string) and not _inside_approved_patch(
            node, parents
        ):
            _positive_finding(
                findings,
                seen,
                "real_time_primitive",
                node,
                text,
                "production time/threading symbols are allowed only in the two exact virtual replacement patches",
            )

    for node in method_nodes:
        dump = ast.dump(node, include_attributes=False)
        forbidden_order = isinstance(node, ast.Compare) and any(
            isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
            for operator in node.ops
        )
        if isinstance(node, ast.Call):
            path = _lexical_path(node.func)
            terminal = (
                path.rsplit(".", 1)[-1]
                if path
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            forbidden_order = forbidden_order or terminal in _ORDER_HELPER_ATTRIBUTES
        if forbidden_order and dump not in _APPROVED_ORDER_EXPRESSIONS:
            _positive_finding(
                findings,
                seen,
                "wall_clock_bound",
                node,
                text,
                "ordering and upper-bound expressions require an exact positive AST allowlist occurrence",
            )

    _scan_protected_contract_bindings(
        tree,
        method,
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
    direct_classes = [
        node
        for node in tree.body
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
        if direct_classes != [target_class]:
            _positive_finding(
                findings,
                seen,
                "target_location_contract",
                target_class,
                text,
                f"exactly one {_TARGET_CLASS} must be directly in module tree.body",
            )
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
        _scan_positive_structural_contract(
            tree,
            target_class,
            method,
            text,
            resolver,
            findings,
            seen,
        )
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
    parameters: str = "self, case",
) -> str:
    lines: list[str] = [
        "import pytest",
        "from types import SimpleNamespace",
        "from unittest.mock import patch",
        "from agent.auxiliary_client import _CodexCompletionsAdapter",
    ]
    lines.extend(line for line in imports.splitlines() if line.strip())
    lines.extend(line for line in module_code.splitlines() if line.strip())
    lines.extend(line for line in class_decorators.splitlines() if line.strip())
    lines.append(f"class {_TARGET_CLASS}:")
    lines.extend(f"    {line}" for line in class_code.splitlines() if line.strip())
    lines.append('    @pytest.mark.parametrize("case", range(1000))')
    lines.extend(f"    {line}" for line in method_decorators.splitlines() if line.strip())
    lines.append(f"    def {_TARGET_TEST}({parameters}):")
    lines.extend(textwrap.indent(body, "        ").splitlines())
    safety_scaffold = textwrap.dedent(
        """\
        class _VirtualClock:
            def monotonic(self):
                return 0.0
        class _ObservableTimer:
            pass
        clock = _VirtualClock()
        import agent.auxiliary_client as aux
        with (
            patch.object(aux.time, "monotonic", clock.monotonic),
            patch.object(aux.threading, "Timer", _ObservableTimer),
        ):
            pass
        """
    )
    lines.extend(textwrap.indent(safety_scaffold, "        ").splitlines())
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


def _nested_target_class_fixture() -> str:
    text = _fixture("pass")
    start = text.index(f"class {_TARGET_CLASS}:\n")
    return text[:start] + "def qa_build_hidden_target():\n" + textwrap.indent(
        text[start:], "    "
    )


_APPLICABILITY_SELF_TEST_CASES = (
    (
        "applicability_module_pytest_generate_tests_skip",
        _fixture(
            "pass",
            module_code=(
                "def pytest_generate_tests(metafunc):\n"
                "    pytest.skip(\"module bypass\", allow_module_level=True)"
            ),
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "applicability_class_pytest_generate_tests_skip",
        _fixture(
            "pass",
            class_code=(
                "def pytest_generate_tests(self, metafunc):\n"
                "    pytest.skip(\"class bypass\")"
            ),
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "applicability_module_pytest_generate_tests_xfail",
        _fixture(
            "pass",
            module_code=(
                "def pytest_generate_tests(metafunc):\n"
                "    pytest.xfail(\"module bypass\")"
            ),
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "applicability_class_pytest_generate_tests_xfail",
        _fixture(
            "pass",
            class_code=(
                "def pytest_generate_tests(self, metafunc):\n"
                "    pytest.xfail(\"class bypass\")"
            ),
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "applicability_module_pytest_generate_tests_collection_reduction",
        _fixture(
            "pass",
            module_code=(
                "def pytest_generate_tests(metafunc):\n"
                "    metafunc.parametrize(\"case\", range(1))"
            ),
        ),
        False,
        ("applicability_contract",),
    ),
    (
        "applicability_class_pytest_generate_tests_parameter_mutation",
        _fixture(
            "pass",
            class_code=(
                "def pytest_generate_tests(self, metafunc):\n"
                "    metafunc.parametrize(\"case\", range(1))"
            ),
        ),
        False,
        ("applicability_contract",),
    ),
    (
        "applicability_module_pytest_generate_tests_retry_marker",
        _fixture(
            "pass",
            module_code=(
                "def pytest_generate_tests(metafunc):\n"
                "    metafunc.definition.add_marker(pytest.mark.retry)"
            ),
        ),
        False,
        ("retry_or_repeat", "applicability_contract"),
    ),
    (
        "applicability_class_pytest_generate_tests_retry_marker",
        _fixture(
            "pass",
            class_code=(
                "def pytest_generate_tests(self, metafunc):\n"
                "    metafunc.definition.add_marker(pytest.mark.retry)"
            ),
        ),
        False,
        ("retry_or_repeat", "applicability_contract"),
    ),
    (
        "applicability_module_pytest_generate_tests_real_time",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def pytest_generate_tests(metafunc):\n"
                "    time.monotonic()"
            ),
        ),
        False,
        ("real_time_primitive", "applicability_contract"),
    ),
    (
        "applicability_class_pytest_generate_tests_real_time",
        _fixture(
            "pass",
            imports="import time",
            class_code=(
                "def pytest_generate_tests(self, metafunc):\n"
                "    time.monotonic()"
            ),
        ),
        False,
        ("real_time_primitive", "applicability_contract"),
    ),
    (
        "applicability_unrelated_class_pytest_generate_tests",
        _fixture(
            "pass",
            module_code=(
                "class _ReviewOther:\n"
                "    def pytest_generate_tests(self, metafunc):\n"
                "        pytest.skip(\"unrelated\")"
            ),
        ),
        True,
        (),
    ),
    (
        "applicability_nested_pytest_generate_tests_helper",
        _fixture(
            "pass",
            module_code=(
                "def _review_helper():\n"
                "    def pytest_generate_tests(metafunc):\n"
                "        pytest.skip(\"unrelated\")\n"
                "    return pytest_generate_tests"
            ),
        ),
        True,
        (),
    ),
    (
        "applicability_module_range_shadow",
        _fixture(
            "pass",
            imports="import pytest",
            module_code=(
                "import builtins as _review_builtins\n"
                "def range(*args):\n"
                "    if args == (1000,):\n"
                "        return _review_builtins.range(1)\n"
                "    return _review_builtins.range(*args)"
            ),
        ),
        False,
        ("binding_contract",),
    ),
    (
        "applicability_setup_method_skip",
        _fixture(
            "pass",
            imports="import pytest",
            class_code='def setup_method(self):\n    pytest.skip("setup bypass")',
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "applicability_setup_method_xfail",
        _fixture(
            "pass",
            imports="import pytest",
            class_code='def setup_method(self):\n    pytest.xfail("setup bypass")',
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "applicability_setup_method_sleep",
        _fixture(
            "pass",
            imports="import pytest\nimport time",
            class_code="def setup_method(self):\n    time.sleep(0.001)",
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "applicability_teardown_method_sleep",
        _fixture(
            "pass",
            imports="import pytest\nimport time",
            class_code="def teardown_method(self):\n    time.sleep(0.001)",
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "applicability_class_autouse_fixture",
        _fixture(
            "pass",
            imports="import pytest",
            class_code=(
                "@pytest.fixture(autouse=True)\n"
                "def _review_autouse(self):\n"
                "    pytest.skip(\"fixture bypass\")"
            ),
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "applicability_module_setup_module",
        _fixture(
            "pass",
            imports="import pytest",
            module_code='def setup_module(module):\n    pytest.skip("module bypass")',
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "applicability_module_default_skip",
        _fixture(
            "pass",
            imports="import pytest",
            module_code='def _review_default(value=pytest.skip("default bypass")):\n    pass',
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "applicability_module_default_sleep",
        _fixture(
            "pass",
            imports="import pytest\nimport time",
            module_code="def _review_default(value=time.sleep(0.001)):\n    pass",
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "applicability_module_pytest_shadow",
        _fixture(
            "pass",
            imports="import pytest",
            module_code="class pytest:\n    mark = pytest.mark\n    raises = pytest.raises",
        ),
        False,
        ("binding_contract",),
    ),
    (
        "applicability_module_production_adapter_shadow",
        _fixture(
            "pass",
            imports="from agent.auxiliary_client import _CodexCompletionsAdapter",
            module_code=(
                "class _CodexCompletionsAdapter:\n"
                "    def create(self, **kwargs):\n"
                "        raise TimeoutError"
            ),
        ),
        False,
        ("binding_contract",),
    ),
    (
        "applicability_async_target_conversion",
        _fixture("pass").replace(
            f"    def {_TARGET_TEST}(",
            f"    async def {_TARGET_TEST}(",
            1,
        ),
        False,
        ("signature_contract",),
    ),
)

_D8_SELF_TEST_CASES = (
    (
        "d8_r1_module_autouse_import_alias_skip",
        _fixture(
            "pass",
            imports="from pytest import fixture as automatic",
            module_code=(
                "@automatic(autouse=True)\n"
                "def qa_alias_guard():\n"
                "    pytest.skip('alias autouse')"
            ),
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "d8_r2_class_autouse_truthy_integer_skip",
        _fixture(
            "pass",
            class_code=(
                "@pytest.fixture(autouse=1)\n"
                "def qa_truthy_guard(self):\n"
                "    pytest.skip('truthy autouse')"
            ),
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "d8_r3_indirect_default_factory_sleep",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_make_default():\n"
                "    time.sleep(0)\n"
                "    return None\n"
                "def qa_indirect_default(value=qa_make_default()):\n"
                "    return value"
            ),
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d8_r4_return_annotation_sleep",
        _fixture(
            "pass",
            imports="import time",
            module_code="def qa_return() -> time.sleep(0):\n    pass",
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d8_r4_positional_annotation_skip",
        _fixture(
            "pass",
            module_code="def qa_positional(value: pytest.skip('annotation')):\n    pass",
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "d8_r4_keyword_annotation_sleep",
        _fixture(
            "pass",
            imports="import time",
            module_code="def qa_keyword(*, value: time.sleep(0)):\n    pass",
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d8_r4_vararg_annotation_skip",
        _fixture(
            "pass",
            module_code="def qa_vararg(*values: pytest.skip('annotation')):\n    pass",
        ),
        False,
        ("skip_or_xfail",),
    ),
    (
        "d8_r4_kwarg_annotation_sleep",
        _fixture(
            "pass",
            imports="import time",
            module_code="def qa_kwarg(**values: time.sleep(0)):\n    pass",
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d8_r5_decorator_function_body_sleep",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_decorate(function):\n"
                "    time.sleep(0)\n"
                "    return function\n"
                "@qa_decorate\n"
                "def qa_decorated():\n"
                "    pass"
            ),
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d8_r6_wildcard_import_binding_ambiguity",
        _fixture("pass", module_code="from qa_shadow_star import *"),
        False,
        ("binding_contract",),
    ),
    (
        "d8_r7_autouse_global_range_rebind",
        _fixture(
            "pass",
            module_code=(
                "@pytest.fixture(autouse=True)\n"
                "def qa_range_rebinder():\n"
                "    global range\n"
                "    range = lambda *values: [1, 2]"
            ),
        ),
        False,
        ("binding_contract",),
    ),
    (
        "d8_r8_generator_target",
        _fixture("if False:\n    yield None"),
        False,
        ("signature_contract",),
    ),
    (
        "d8_r9_target_class_nested_in_factory",
        _nested_target_class_fixture(),
        False,
        ("target_location_contract",),
    ),
    (
        "d8_s1_safe_module_literal_constant",
        _fixture("pass", module_code="QA_UNRELATED_SENTINEL = 7"),
        True,
        (),
    ),
    (
        "d8_s2_safe_target_class_docstring",
        _fixture("pass", class_code="'QA harmless class documentation'"),
        True,
        (),
    ),
    (
        "d8_annotation_postponed_semantics",
        _fixture(
            "pass",
            imports="import time",
            module_code="def qa_postponed(value: time.sleep(0)) -> pytest.skip('annotation'):\n    pass",
        ).replace("import pytest\n", "from __future__ import annotations\nimport pytest\n", 1),
        True,
        (),
    ),
)

_D9_SELF_TEST_CASES = (
    (
        "d9_r1_returned_decorator_callable_sleep",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_returned_decorator(function):\n"
                "    time.sleep(0)\n"
                "    return function\n"
                "def qa_decorator_factory():\n"
                "    return qa_returned_decorator\n"
                "@qa_decorator_factory()\n"
                "def qa_decorated():\n"
                "    pass"
            ),
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d9_r2_function_return_default_callable_sleep",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_returned_default():\n"
                "    time.sleep(0)\n"
                "    return 7\n"
                "def qa_default_factory():\n"
                "    return qa_returned_default\n"
                "def qa_default(value=qa_default_factory()()):\n"
                "    return value"
            ),
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d9_s1_false_assigned_autouse_module",
        _fixture(
            "pass",
            module_code=(
                "qa_fixture_alias = pytest.fixture\n"
                "qa_fixture_alias_chain = qa_fixture_alias\n"
                "@qa_fixture_alias_chain(autouse=())\n"
                "def qa_inactive_fixture():\n"
                "    pytest.skip('inactive fixture')"
            ),
        ),
        True,
        (),
    ),
    (
        "d9_s2_false_assigned_autouse_class",
        _fixture(
            "pass",
            class_code=(
                "qa_fixture_alias = pytest.fixture\n"
                "qa_fixture_alias_chain = qa_fixture_alias\n"
                "@qa_fixture_alias_chain(autouse=0)\n"
                "def qa_inactive_fixture(self):\n"
                "    pytest.xfail('inactive fixture')"
            ),
        ),
        True,
        (),
    ),
    (
        "d9_s3_called_factory_dormant_nested_helper",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_dormant_factory():\n"
                "    def qa_dormant_helper():\n"
                "        time.sleep(0)\n"
                "    return 7\n"
                "def qa_default(value=qa_dormant_factory()):\n"
                "    return value"
            ),
        ),
        True,
        (),
    ),
    (
        "d9_returned_callable_cycle_is_bounded",
        _fixture(
            "pass",
            module_code=(
                "def qa_cycle_first():\n"
                "    return qa_cycle_second\n"
                "def qa_cycle_second():\n"
                "    return qa_cycle_first\n"
                "def qa_default(value=qa_cycle_first()()()):\n"
                "    return value"
            ),
        ),
        True,
        (),
    ),
    (
        "d9_unresolved_returned_callable_fails_closed",
        _fixture(
            "pass",
            module_code=(
                "def qa_dynamic_factory(source):\n"
                "    return source\n"
                "def qa_default(value=qa_dynamic_factory(qa_dynamic_factory)()):\n"
                "    return value"
            ),
        ),
        False,
        ("applicability_contract",),
    ),
    (
        "d9_benign_returned_callable_control",
        _fixture(
            "pass",
            module_code=(
                "def qa_benign_callable():\n"
                "    return 7\n"
                "def qa_benign_factory():\n"
                "    return qa_benign_callable\n"
                "def qa_default(value=qa_benign_factory()()):\n"
                "    return value"
            ),
        ),
        True,
        (),
    ),
)

_D10_SELF_TEST_CASES = (
    (
        "d10_r1_triple_returned_callable_chain",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_d10_sleep_decorator(function):\n"
                "    time.sleep(0)\n"
                "    return function\n"
                "def qa_d10_leaf():\n"
                "    return qa_d10_sleep_decorator\n"
                "def qa_d10_middle():\n"
                "    return qa_d10_leaf\n"
                "def qa_d10_factory():\n"
                "    return qa_d10_middle\n"
                "@qa_d10_factory()()()\n"
                "def qa_d10_decorated():\n"
                "    pass"
            ),
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d10_r1_alias_at_each_returned_callable_layer",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_d10_alias_sleep(function):\n"
                "    time.sleep(0)\n"
                "    return function\n"
                "qa_d10_leaf_alias = qa_d10_alias_sleep\n"
                "def qa_d10_alias_next():\n"
                "    return qa_d10_leaf_alias\n"
                "qa_d10_middle_alias = qa_d10_alias_next\n"
                "def qa_d10_alias_factory():\n"
                "    return qa_d10_middle_alias\n"
                "@qa_d10_alias_factory()()\n"
                "def qa_d10_alias_decorated():\n"
                "    pass"
            ),
        ),
        False,
        ("applicability_contract",),
    ),
    (
        "d10_r2_mixed_branch_returned_callables",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_d10_safe_identity(function):\n"
                "    return function\n"
                "def qa_d10_branch_sleep(function):\n"
                "    time.sleep(0)\n"
                "    return function\n"
                "def qa_d10_branch_factory():\n"
                "    if True:\n"
                "        return qa_d10_branch_sleep\n"
                "    return qa_d10_safe_identity\n"
                "@qa_d10_branch_factory()\n"
                "def qa_d10_branch_decorated():\n"
                "    pass"
            ),
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d10_r2_cyclic_returned_decorator_fails_closed",
        _fixture(
            "pass",
            module_code=(
                "def qa_d10_cycle_first():\n"
                "    return qa_d10_cycle_second\n"
                "def qa_d10_cycle_second():\n"
                "    return qa_d10_cycle_first\n"
                "@qa_d10_cycle_first()()()\n"
                "def qa_d10_cycle_decorated():\n"
                "    pass"
            ),
        ),
        False,
        ("applicability_contract",),
    ),
    (
        "d10_r2_unresolved_returned_chain_fails_closed",
        _fixture(
            "pass",
            module_code=(
                "def qa_d10_unresolved_factory(source):\n"
                "    return source\n"
                "@qa_d10_unresolved_factory(qa_d10_unresolved_factory)()\n"
                "def qa_d10_unresolved_decorated():\n"
                "    pass"
            ),
        ),
        False,
        ("applicability_contract",),
    ),
    (
        "d10_s1_dormant_nested_class_body_is_inert",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_d10_identity(function):\n"
                "    return function\n"
                "def qa_d10_dormant_class_factory():\n"
                "    def qa_d10_dormant_helper():\n"
                "        class QaD10DormantClass:\n"
                "            def qa_d10_method(self):\n"
                "                time.sleep(0)\n"
                "        return QaD10DormantClass\n"
                "    return qa_d10_identity\n"
                "@qa_d10_dormant_class_factory()\n"
                "def qa_d10_dormant_class_decorated():\n"
                "    pass"
            ),
        ),
        True,
        (),
    ),
    (
        "d10_s2_dormant_nested_async_body_is_inert",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_d10_identity(function):\n"
                "    return function\n"
                "def qa_d10_dormant_async_factory():\n"
                "    async def qa_d10_dormant_async():\n"
                "        time.sleep(0)\n"
                "    return qa_d10_identity\n"
                "@qa_d10_dormant_async_factory()\n"
                "def qa_d10_dormant_async_decorated():\n"
                "    pass"
            ),
        ),
        True,
        (),
    ),
    (
        "d10_r3_nested_definition_decorator_executes",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_d10_identity(function):\n"
                "    return function\n"
                "def qa_d10_nested_decorator(function):\n"
                "    time.sleep(0)\n"
                "    return function\n"
                "def qa_d10_nested_decorator_factory():\n"
                "    @qa_d10_nested_decorator\n"
                "    def qa_d10_nested_body():\n"
                "        return 1\n"
                "    return qa_d10_identity\n"
                "@qa_d10_nested_decorator_factory()\n"
                "def qa_d10_nested_decorator_target():\n"
                "    pass"
            ),
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d10_r3_nested_definition_default_executes",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_d10_identity(function):\n"
                "    return function\n"
                "def qa_d10_nested_default_trigger():\n"
                "    time.sleep(0)\n"
                "    return 1\n"
                "def qa_d10_nested_default_factory():\n"
                "    def qa_d10_nested_body(value=qa_d10_nested_default_trigger()):\n"
                "        return value\n"
                "    return qa_d10_identity\n"
                "@qa_d10_nested_default_factory()\n"
                "def qa_d10_nested_default_target():\n"
                "    pass"
            ),
        ),
        False,
        ("real_time_primitive",),
    ),
    (
        "d10_r3_nested_class_base_executes",
        _fixture(
            "pass",
            imports="import time",
            module_code=(
                "def qa_d10_identity(function):\n"
                "    return function\n"
                "def qa_d10_nested_base_trigger():\n"
                "    time.sleep(0)\n"
                "    return object\n"
                "def qa_d10_nested_base_factory():\n"
                "    class QaD10NestedClass(qa_d10_nested_base_trigger()):\n"
                "        pass\n"
                "    return qa_d10_identity\n"
                "@qa_d10_nested_base_factory()\n"
                "def qa_d10_nested_base_target():\n"
                "    pass"
            ),
        ),
        False,
        ("real_time_primitive",),
    ),
)

_SELF_TEST_CASES = _APPLICABILITY_SELF_TEST_CASES + _D8_SELF_TEST_CASES + _D9_SELF_TEST_CASES + _D10_SELF_TEST_CASES + (
    ("from_time_sleep_alias", _fixture("nap(0.03)", imports="from time import sleep as nap"), False, ("real_time_primitive",)),
    ("module_time_alias_sleep", _fixture("clock.sleep(0.03)", imports="import time as clock"), False, ("unapproved_symbol",)),
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
    ("positive_module_list_callback", _fixture("_review_callbacks[0]", module_code="_review_callbacks = [time.sleep]"), False, ("applicability_contract", "unapproved_symbol")),
    ("positive_module_tuple_callback", _fixture("_review_callbacks[0]", module_code="_review_callbacks = (time.sleep,)"), False, ("applicability_contract", "unapproved_symbol")),
    ("positive_module_dict_callback", _fixture("_review_callbacks['callback']", module_code="_review_callbacks = {'callback': time.sleep}"), False, ("applicability_contract", "unapproved_symbol")),
    ("positive_multihop_attribute", _fixture("_review_holder.layer.callback", module_code="class _ReviewLayer:\n    callback = time.sleep\nclass _ReviewHolder:\n    layer = _ReviewLayer()\n_review_holder = _ReviewHolder()"), False, ("applicability_contract", "unapproved_symbol")),
    ("positive_function_return", _fixture("_review_factory()", module_code="def _review_factory():\n    return time.sleep"), False, ("unapproved_symbol", "higher_order_contract")),
    ("positive_lambda_default", _fixture("_review_probe = lambda source=time.sleep: source", imports="import time"), False, ("higher_order_contract", "unapproved_symbol")),
    ("positive_function_default", _fixture("def create(self, function=time.sleep):\n    return function", imports="import time"), False, ("higher_order_contract", "unapproved_symbol")),
    ("positive_nested_closure", _fixture("def create(self):\n    def _review_read():\n        return time.sleep\n    return _review_read", imports="import time"), False, ("unapproved_definition", "unapproved_symbol")),
    ("positive_list_comprehension", _fixture("_review_values = [value for value in [time.sleep]]", imports="import time"), False, ("higher_order_contract", "unapproved_symbol")),
    ("positive_generator_expression", _fixture("_review_values = (value for value in [time.sleep])", imports="import time"), False, ("higher_order_contract", "unapproved_symbol")),
    ("positive_callback_argument", _fixture("SimpleNamespace(close=time.sleep)", imports="import time"), False, ("unapproved_symbol",)),
    ("positive_dynamic_vars", _fixture("vars(time)['sleep']", imports="import time"), False, ("unapproved_symbol", "higher_order_contract")),
    ("positive_condition_wait", _fixture("condition.wait"), False, ("unapproved_symbol",)),
    ("positive_queue_timeout", _fixture("queue.get(timeout=0.01)"), False, ("unapproved_symbol", "higher_order_contract")),
    ("positive_select_wait", _fixture("select.select([], [], [], 0.01)"), False, ("unapproved_symbol", "higher_order_contract")),
    ("positive_thread_join", _fixture("worker.join(0.01)"), False, ("unapproved_symbol", "higher_order_contract")),
    ("positive_process_wait", _fixture("process.wait(timeout=0.01)"), False, ("unapproved_symbol", "higher_order_contract")),
    ("positive_socket_timeout", _fixture("sock.settimeout(0.01)"), False, ("unapproved_symbol", "higher_order_contract")),
    ("positive_reruns_escape", _fixture("pytest.mark.reruns"), False, ("unapproved_symbol",)),
    ("positive_decorator_factory", _fixture("pass", module_code="def _review_mark_factory(mark=pytest.mark.skip):\n    return mark", method_decorators="@_review_mark_factory()"), False, ("decorator_contract", "unapproved_symbol")),
    ("positive_indexed_decorator", _fixture("pass", module_code="_review_marks = [pytest.mark.skip]", method_decorators="@_review_marks[0]"), False, ("decorator_contract", "applicability_contract")),
    ("positive_factory_pytestmark", _fixture("pass", module_code="def _review_mark_factory():\n    return pytest.mark.skip\npytestmark = _review_mark_factory()"), False, ("applicability_contract",)),
    ("positive_class_pytestmark_factory", _fixture("pass", module_code="def _review_mark_factory():\n    return pytest.mark.skip", class_code="pytestmark = _review_mark_factory()"), False, ("applicability_contract",)),
    ("positive_virtual_lt", _fixture("assert clock.monotonic() < 1"), False, ("wall_clock_bound",)),
    ("positive_virtual_lte", _fixture("assert clock.monotonic() <= 1"), False, ("wall_clock_bound",)),
    ("positive_virtual_operator_lt", _fixture("assert operator.lt(clock.monotonic(), 1)", imports="import operator"), False, ("wall_clock_bound",)),
    ("positive_virtual_dunder_lt", _fixture("assert clock.monotonic().__lt__(1)"), False, ("wall_clock_bound",)),
    ("positive_virtual_min_bound", _fixture("assert min(clock.monotonic(), 0.5) < 1"), False, ("wall_clock_bound",)),
    ("positive_virtual_max_bound", _fixture("assert max(clock.monotonic(), 0.5) <= 1"), False, ("wall_clock_bound",)),
    ("positive_virtual_not_ge", _fixture("assert not clock.monotonic() >= 1"), False, ("wall_clock_bound",)),
    ("positive_bad_monotonic_replacement", _fixture().replace("clock.monotonic),", "_ObservableTimer),", 1), False, ("patch_contract",)),
    ("positive_bad_timer_replacement", _fixture().replace("_ObservableTimer),", "clock.monotonic),", 1), False, ("patch_contract",)),
    ("positive_missing_monotonic_patch", _fixture().replace('            patch.object(aux.time, "monotonic", clock.monotonic),\n', "", 1), False, ("patch_contract",)),
    ("positive_missing_timer_patch", _fixture().replace('            patch.object(aux.threading, "Timer", _ObservableTimer),\n', "", 1), False, ("patch_contract",)),
    ("positive_duplicate_monotonic_patch", _fixture().replace('            patch.object(aux.time, "monotonic", clock.monotonic),\n', '            patch.object(aux.time, "monotonic", clock.monotonic),\n            patch.object(aux.time, "monotonic", clock.monotonic),\n', 1), False, ("patch_contract",)),
    ("positive_direct_aux_clock", _fixture("aux.time.monotonic", imports="import agent.auxiliary_client as aux"), False, ("real_time_primitive",)),
    ("positive_direct_aux_timer", _fixture("aux.threading.Timer", imports="import agent.auxiliary_client as aux"), False, ("real_time_primitive",)),
    ("positive_extra_target_import", _fixture("import time"), False, ("target_import_contract",)),
    ("positive_signature_default", _fixture("pass", parameters="self, case=0"), False, ("signature_contract",)),
    ("positive_safe_equality", _fixture("assert case == case"), True, ()),
    ("positive_safe_count_equality", _fixture("consumed_events = []\nassert len(consumed_events) == 0"), True, ()),
    ("positive_safe_virtual_reference", _fixture("assert clock.monotonic() == 0.0"), True, ()),
    ("positive_safe_patch_quotes", _fixture().replace('"monotonic"', "'monotonic'", 1).replace('"Timer"', "'Timer'", 1), True, ()),
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
