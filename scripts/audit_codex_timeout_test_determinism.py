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
from pathlib import Path


_TARGET_CLASS = "TestCodexAuxiliaryAdapterTimeout"
_TARGET_TEST = "test_enforces_total_timeout_while_stream_keeps_emitting_events"
_REAL_TIME_ATTRIBUTES = {
    "sleep",
    "monotonic",
    "perf_counter",
    "process_time",
    "thread_time",
    "time",
}


def _qualified_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _source(node: ast.AST, text: str) -> str:
    return ast.get_source_segment(text, node) or ast.dump(node, include_attributes=False)


def _finding(kind: str, node: ast.AST, text: str, detail: str) -> dict[str, object]:
    return {
        "kind": kind,
        "line": getattr(node, "lineno", None),
        "detail": detail,
        "source": _source(node, text),
    }


def _target_method(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != _TARGET_CLASS:
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == _TARGET_TEST:
                return child
    return None


def _audit(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    method = _target_method(tree)
    findings: list[dict[str, object]] = []

    if method is None:
        findings.append({
            "kind": "missing_target",
            "line": None,
            "detail": f"missing {_TARGET_CLASS}.{_TARGET_TEST}",
            "source": "",
        })
    else:
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                qualified = _qualified_name(node.func)
                if qualified.startswith("time.") and qualified.rsplit(".", 1)[-1] in _REAL_TIME_ATTRIBUTES:
                    findings.append(_finding("real_time_primitive", node, text, f"forbidden call: {qualified}"))
                elif qualified == "threading.Timer":
                    findings.append(_finding("real_timer_constructor", node, text, "direct threading.Timer construction is forbidden"))
                elif qualified in {"pytest.skip", "pytest.xfail"}:
                    findings.append(_finding("skip_or_xfail", node, text, f"forbidden call: {qualified}"))
                elif qualified.endswith(".retry") or qualified in {"retry", "retries", "repeat"}:
                    findings.append(_finding("retry_or_repeat", node, text, f"forbidden retry/repeat call: {qualified}"))

            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value == 0.14:
                        findings.append(_finding(
                            "wall_clock_bound",
                            node,
                            text,
                            "forbidden hard wall-clock bound: < 0.14",
                        ))

        for decorator in method.decorator_list:
            qualified = _qualified_name(decorator)
            if qualified in {"pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail"}:
                findings.append(_finding("skip_or_xfail", decorator, text, f"forbidden decorator: {qualified}"))

    findings.sort(key=lambda item: (item["line"] is None, item["line"] or 0, str(item["kind"])))
    return {
        "audit": "codex-timeout-test-determinism",
        "target": str(path),
        "class": _TARGET_CLASS,
        "test": _TARGET_TEST,
        "forbidden_real_time_primitives": findings,
        "forbidden_count": len(findings),
        "passed": not findings,
    }


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
        }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
