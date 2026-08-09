#!/usr/bin/env python3
"""AST audit for the deterministic Codex timeout regression test.

This audit deliberately inspects only the owned timeout test.  It rejects
real-time primitives and wall-clock assertions so the behavioral test cannot
silently become timing-sensitive again.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
from pathlib import Path


_TARGET_CLASS = "TestCodexAuxiliaryAdapterTimeout"
_TARGET_TEST = "test_enforces_total_timeout_while_stream_keeps_emitting_events"
_REAL_TIME_ATTRIBUTES = {
    "sleep",
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "process_time_ns",
    "thread_time",
    "thread_time_ns",
    "time",
    "time_ns",
}
_DATETIME_CLOCK_ATTRIBUTES = {"now", "utcnow", "today"}
_ESCAPE_NAMES = {
    "skip",
    "skipif",
    "xfail",
    "flaky",
    "retry",
    "retries",
    "rerun",
    "repeat",
}
_ELAPSED_NAME_PARTS = ("elapsed", "duration", "latency", "wall_clock", "wallclock")


def _qualified_name(node: ast.AST, aliases: dict[str, str] | None = None) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    qualified = ".".join(reversed(parts))
    if aliases:
        root, *rest = qualified.split(".")
        if root in aliases:
            qualified = ".".join([aliases[root], *rest])
    return qualified


def _source(node: ast.AST, text: str) -> str:
    return ast.get_source_segment(text, node) or ast.dump(node, include_attributes=False)


def _finding(kind: str, node: ast.AST, text: str, detail: str) -> dict[str, object]:
    return {
        "kind": kind,
        "line": getattr(node, "lineno", None),
        "detail": detail,
        "source": _source(node, text),
    }


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    """Return the canonical dotted name for every imported local binding."""
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imports.sort(key=lambda node: getattr(node, "lineno", 0))

    aliases: dict[str, str] = {}
    for node in imports:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.asname:
                    aliases[imported.asname] = imported.name
                else:
                    # ``import package.module`` binds ``package``.
                    bound = imported.name.split(".", 1)[0]
                    aliases[bound] = bound
            continue

        module = "." * node.level + (node.module or "")
        for imported in node.names:
            if imported.name == "*":
                continue
            bound = imported.asname or imported.name
            aliases[bound] = ".".join(part for part in (module, imported.name) if part)
    return aliases


def _target_method(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the unique direct target method, or ``None`` when not unique."""
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == _TARGET_CLASS
    ]
    if len(classes) != 1:
        return None
    methods = [
        child
        for child in classes[0].body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and child.name == _TARGET_TEST
    ]
    return methods[0] if len(methods) == 1 else None


def _looks_elapsed_name(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in _ELAPSED_NAME_PARTS)


def _event_loop_time_call(node: ast.Call, aliases: dict[str, str]) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "time":
        return False
    qualified = _qualified_name(node.func, aliases)
    if qualified == "time.time":
        return False
    # A call-chain such as asyncio.get_running_loop().time() has no simple
    # qualified name because its receiver is itself a Call, so the generic
    # attribute check intentionally fails closed as event-loop time too.
    return True


def _real_time_call(node: ast.Call, aliases: dict[str, str]) -> tuple[str, str] | None:
    qualified = _qualified_name(node.func, aliases)
    if qualified:
        namespace, _, attribute = qualified.rpartition(".")
        if namespace == "time" and attribute in _REAL_TIME_ATTRIBUTES:
            return "real_time_primitive", f"forbidden call: {qualified}"
        if namespace == "asyncio" and attribute == "sleep":
            return "real_time_primitive", f"forbidden call: {qualified}"
        if namespace == "asyncio" and attribute in {"timeout", "timeout_at"}:
            return "real_timer_constructor", f"forbidden asyncio timer: {qualified}"
        if qualified == "threading.Timer":
            return "real_timer_constructor", "real threading.Timer construction is forbidden"
        if namespace.startswith("datetime.") and attribute in _DATETIME_CLOCK_ATTRIBUTES:
            return "real_time_primitive", f"forbidden datetime clock: {qualified}"

    if isinstance(node.func, ast.Attribute) and node.func.attr in {"call_at", "call_later"}:
        return "real_timer_constructor", "event-loop timer construction is forbidden"
    if _event_loop_time_call(node, aliases):
        return "event_loop_time", "event-loop or real clock .time() is forbidden"
    return None


def _escape_kind(qualified: str | None) -> tuple[str, str] | None:
    if not qualified:
        return None
    name = qualified.rsplit(".", 1)[-1]
    if name not in _ESCAPE_NAMES:
        return None
    if name in {"skip", "skipif", "xfail"}:
        return "skip_or_xfail", f"forbidden skip/xfail form: {qualified}"
    return "retry_or_repeat", f"forbidden retry/repeat form: {qualified}"


def _append_unique(
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
    finding: dict[str, object],
) -> None:
    key = (finding["kind"], finding["line"], finding["source"])
    if key not in seen:
        findings.append(finding)
        seen.add(key)


def _assignment_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_assignment_names(element))
        return names
    return []


def _clock_derived(
    node: ast.AST,
    aliases: dict[str, str],
    derived_names: set[str],
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in derived_names or _looks_elapsed_name(node.id)
    if isinstance(node, ast.Call):
        return _real_time_call(node, aliases) is not None
    if isinstance(node, ast.BinOp):
        return _clock_derived(node.left, aliases, derived_names) or _clock_derived(
            node.right, aliases, derived_names
        )
    if isinstance(node, ast.UnaryOp):
        return _clock_derived(node.operand, aliases, derived_names)
    if isinstance(node, ast.IfExp):
        return _clock_derived(node.body, aliases, derived_names) or _clock_derived(
            node.orelse, aliases, derived_names
        )
    return False


def _elapsed_less_than_bound(
    node: ast.Compare,
    aliases: dict[str, str],
    derived_names: set[str],
) -> bool:
    values = [node.left, *node.comparators]
    for operator, left, right in zip(node.ops, values, values[1:]):
        left_derived = _clock_derived(left, aliases, derived_names)
        right_derived = _clock_derived(right, aliases, derived_names)
        if isinstance(operator, (ast.Lt, ast.LtE)) and (left_derived or right_derived):
            return True
        if isinstance(operator, (ast.Gt, ast.GtE)) and right_derived:
            return True
        # Keep the original literal guard as a compatibility backstop while
        # the derived-expression guard catches any changed elapsed threshold.
        if isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) and any(
            isinstance(value, ast.Constant) and value.value == 0.14 for value in (left, right)
        ):
            return True
    return False


def _scan_applicability(
    tree: ast.AST,
    text: str,
    aliases: dict[str, str],
    findings: list[dict[str, object]],
    seen: set[tuple[object, object, object]],
) -> None:
    """Reject collection/runtime escapes at module, class, and method scope."""
    for node in sorted(ast.walk(tree), key=lambda item: getattr(item, "lineno", 0)):
        if isinstance(node, ast.Call):
            escape = _escape_kind(_qualified_name(node.func, aliases))
            if escape:
                kind, detail = escape
                _append_unique(findings, seen, _finding(kind, node, text, detail))

        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                base = decorator.func if isinstance(decorator, ast.Call) else decorator
                escape = _escape_kind(_qualified_name(base, aliases))
                if escape:
                    kind, detail = escape
                    _append_unique(findings, seen, _finding(kind, decorator, text, detail))

        # pytest's module/class applicability marker is an assignment rather
        # than a decorator, including the bare ``pytest.mark.skip`` form.
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            for value in ast.walk(node.value):
                if isinstance(value, (ast.Attribute, ast.Name)):
                    escape = _escape_kind(_qualified_name(value, aliases))
                    if escape:
                        kind, detail = escape
                        _append_unique(findings, seen, _finding(kind, value, text, detail))


def _audit_text(text: str, target: str) -> dict[str, object]:
    tree = ast.parse(text, filename=target)
    aliases = _import_aliases(tree)
    findings: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()

    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == _TARGET_CLASS
    ]
    method: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    if not classes:
        findings.append({
            "kind": "missing_target",
            "line": None,
            "detail": f"missing {_TARGET_CLASS}.{_TARGET_TEST}",
            "source": "",
        })
    elif len(classes) != 1:
        findings.append(_finding(
            "duplicate_target_class",
            classes[1] if len(classes) > 1 else classes[0],
            text,
            f"expected exactly one {_TARGET_CLASS}, found {len(classes)}",
        ))
    else:
        methods = [
            child
            for child in classes[0].body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == _TARGET_TEST
        ]
        if not methods:
            findings.append(_finding(
                "missing_target",
                classes[0],
                text,
                f"missing direct method {_TARGET_CLASS}.{_TARGET_TEST}",
            ))
        elif len(methods) != 1:
            findings.append(_finding(
                "duplicate_target_method",
                methods[1] if len(methods) > 1 else methods[0],
                text,
                f"expected exactly one direct {_TARGET_TEST}, found {len(methods)}",
            ))
        else:
            method = methods[0]

    if method is not None:
        derived_names: set[str] = set()
        assignments = [
            node
            for node in ast.walk(method)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        ]
        assignments.sort(key=lambda node: getattr(node, "lineno", 0))
        for assignment in assignments:
            value = assignment.value
            if not isinstance(value, ast.AST) or not _clock_derived(value, aliases, derived_names):
                continue
            if isinstance(assignment, ast.Assign):
                targets = assignment.targets
            elif isinstance(assignment, ast.AnnAssign):
                targets = [assignment.target]
            else:
                targets = [assignment.target]
            for target_node in targets:
                derived_names.update(_assignment_names(target_node))

        for node in sorted(ast.walk(method), key=lambda item: getattr(item, "lineno", 0)):
            if isinstance(node, ast.Call):
                real_time = _real_time_call(node, aliases)
                if real_time:
                    kind, detail = real_time
                    _append_unique(findings, seen, _finding(kind, node, text, detail))
            elif isinstance(node, ast.Compare) and _elapsed_less_than_bound(node, aliases, derived_names):
                _append_unique(findings, seen, _finding(
                    "wall_clock_bound",
                    node,
                    text,
                    "forbidden elapsed less-than wall-clock bound",
                ))

    _scan_applicability(tree, text, aliases, findings, seen)
    findings.sort(key=lambda item: (item["line"] is None, item["line"] or 0, str(item["kind"])))
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
    class_decorators: str = "",
    method_decorators: str = "",
) -> str:
    lines: list[str] = []
    lines.extend(line for line in imports.splitlines() if line.strip())
    lines.extend(line for line in module_code.splitlines() if line.strip())
    lines.extend(line for line in class_decorators.splitlines() if line.strip())
    lines.append(f"class {_TARGET_CLASS}:")
    lines.extend(f"    {line}" for line in method_decorators.splitlines() if line.strip())
    lines.append(f"    def {_TARGET_TEST}(self):")
    lines.extend(textwrap.indent(body, "        ").splitlines())
    return "\n".join(lines) + "\n"


def _duplicate_class_fixture() -> str:
    return textwrap.dedent(f"""\
        class {_TARGET_CLASS}:
            def {_TARGET_TEST}(self):
                pass

        class {_TARGET_CLASS}:
            def {_TARGET_TEST}(self):
                pass
    """)


def _duplicate_method_fixture() -> str:
    return textwrap.dedent(f"""\
        class {_TARGET_CLASS}:
            def {_TARGET_TEST}(self):
                pass

            def {_TARGET_TEST}(self):
                pass
    """)


_SELF_TEST_CASES = (
    ("from_time_sleep_alias", _fixture("nap(0.03)", imports="from time import sleep as nap"), False, ("real_time_primitive",)),
    ("module_time_alias_sleep", _fixture("clock.sleep(0.03)", imports="import time as clock"), False, ("real_time_primitive",)),
    ("from_time_monotonic_alias", _fixture("tick()", imports="from time import monotonic as tick"), False, ("real_time_primitive",)),
    ("module_threading_timer_alias", _fixture("th.Timer(1, lambda: None)", imports="import threading as th"), False, ("real_timer_constructor",)),
    ("from_threading_timer_alias", _fixture("Alarm(1, lambda: None)", imports="from threading import Timer as Alarm"), False, ("real_timer_constructor",)),
    ("aliased_pytest_skip", _fixture("omit(\"skip\")", imports="import pytest as pt\nfrom pytest import skip as omit"), False, ("skip_or_xfail",)),
    ("module_skip_alias", _fixture("pass", imports="import pytest as pt", module_code="pt.skip(\"module\")"), False, ("skip_or_xfail",)),
    ("module_pytestmark", _fixture("pass", imports="import pytest as pt", module_code="pytestmark = pt.mark.skip"), False, ("skip_or_xfail",)),
    ("class_skip_decorator", _fixture("pass", imports="import pytest", class_decorators="@pytest.mark.skip"), False, ("skip_or_xfail",)),
    ("method_called_skipif", _fixture("pass", imports="import pytest", method_decorators="@pytest.mark.skipif(True)"), False, ("skip_or_xfail",)),
    ("method_called_xfail", _fixture("pass", imports="import pytest", method_decorators="@pytest.mark.xfail(reason=\"x\")"), False, ("skip_or_xfail",)),
    ("aliased_called_skipif", _fixture("pass", imports="from pytest import mark as marks", method_decorators="@marks.skipif(True)"), False, ("skip_or_xfail",)),
    ("called_flaky", _fixture("pass", method_decorators="@flaky(max_runs=2)"), False, ("retry_or_repeat",)),
    ("called_retry", _fixture("pass", method_decorators="@retry(2)"), False, ("retry_or_repeat",)),
    ("called_retries", _fixture("pass", method_decorators="@retries(2)"), False, ("retry_or_repeat",)),
    ("called_rerun", _fixture("pass", method_decorators="@rerun(2)"), False, ("retry_or_repeat",)),
    ("called_repeat", _fixture("pass", method_decorators="@repeat(2)"), False, ("retry_or_repeat",)),
    ("asyncio_sleep", _fixture("asyncio.sleep(0.03)", imports="import asyncio"), False, ("real_time_primitive",)),
    ("event_loop_time", _fixture("event_loop.time()"), False, ("event_loop_time",)),
    ("datetime_now_alias", _fixture("DateTime.now()", imports="from datetime import datetime as DateTime"), False, ("real_time_primitive",)),
    ("changed_elapsed_bound", _fixture("elapsed = 0.2\nassert elapsed < 0.2"), False, ("wall_clock_bound",)),
    ("duplicate_target_class", _duplicate_class_fixture(), False, ("duplicate_target_class",)),
    ("duplicate_target_method", _duplicate_method_fixture(), False, ("duplicate_target_method",)),
    ("safe_virtual_clock", _fixture(textwrap.dedent("""\
        class _VirtualClock:
            def monotonic(self):
                return 0.0
        clock = _VirtualClock()
        assert clock.monotonic() == 0.0
    """)), True, ()),
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
            actual_kinds = sorted({
                str(finding.get("kind"))
                for finding in finding_list
                if isinstance(finding, dict)
            })
            case_passed = actual_passed == expected_passed and set(expected_kinds).issubset(actual_kinds)
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
        default=Path(__file__).resolve().parents[1] / "tests" / "agent" / "test_auxiliary_client.py",
    )
    parser.add_argument("--json", type=Path, required=True, metavar="ARTIFACT")
    args = parser.parse_args()

    try:
        result = _audit(args.target)
    except (OSError, SyntaxError, UnicodeError) as exc:
        result = {
            "audit": "codex-timeout-test-determinism",
            "target": str(args.target),
            "forbidden_real_time_primitives": [{
                "kind": "audit_error",
                "line": None,
                "detail": str(exc),
                "source": "",
            }],
            "forbidden_count": 1,
            "passed": False,
            "self_tests": _run_self_tests(),
        }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
