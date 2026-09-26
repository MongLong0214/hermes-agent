"""Regression for #29335: compressed session repoints survive a gateway restart.

Runner repoints are guarded by SessionStore's CAS; a successful repoint must
persist the new key→session mapping. Manual /compress uses the store's own
advance operation, which must persist before reporting success.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from gateway import run as gateway_run
from gateway import session as gateway_session
from gateway import slash_commands


def _call_on_store(node: ast.AST, method: str, store: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and ast.dump(node.func.value) == ast.dump(store)
    )


def _saves(body: list[ast.stmt], store: ast.AST) -> bool:
    """Only a direct save in this success block, not one in another branch."""
    return any(
        isinstance(stmt, (ast.Expr, ast.Assign))
        and any(_call_on_store(node, "_save", store) for node in ast.walk(stmt))
        for stmt in body
    )


def _repoint_persistence(source: str) -> list[tuple[int, bool]]:
    """Find each runner CAS repoint and check its guarded success path."""
    tree = ast.parse(textwrap.dedent(source))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    sites: list[tuple[int, bool]] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not (isinstance(call.func, ast.Attribute)
                and call.func.attr == "repoint_session_entry"):
            continue
        store = call.func.value
        parent = parents[call]
        while isinstance(parent, ast.Await):
            parent = parents[parent]
        if isinstance(parent, ast.If) and parent.test is call:
            sites.append((call.lineno, _saves(parent.body, store)))
            continue
        if not (isinstance(parent, ast.Assign) and len(parent.targets) == 1
                and isinstance(parent.targets[0], ast.Name)):
            sites.append((call.lineno, False))
            continue
        result_name = parent.targets[0].id
        current: ast.AST = parent
        persisted = False
        # The hygiene CAS is in an inner else block; its success guard lives
        # immediately after that enclosing if. The agent-result CAS checks
        # failure with a raising guard, then saves in the same success block.
        while current in parents and not persisted:
            owner = parents[current]
            for field in ("body", "orelse", "finalbody"):
                body = getattr(owner, field, None)
                if not isinstance(body, list) or current not in body:
                    continue
                following = body[body.index(current) + 1:]
                for index, stmt in enumerate(following):
                    if not isinstance(stmt, ast.If):
                        continue
                    if isinstance(stmt.test, ast.Name) and stmt.test.id == result_name:
                        persisted = _saves(stmt.body, store)
                    elif (isinstance(stmt.test, ast.UnaryOp)
                          and isinstance(stmt.test.op, ast.Not)
                          and isinstance(stmt.test.operand, ast.Name)
                          and stmt.test.operand.id == result_name
                          and stmt.body
                          and isinstance(stmt.body[-1], (ast.Raise, ast.Return))):
                        persisted = _saves(following[index + 1:], store)
                break
            current = owner
        sites.append((call.lineno, persisted))
    return sites


def test_every_guarded_compression_repoint_persists():
    sites = _repoint_persistence(inspect.getsource(gateway_run))
    assert len(sites) == 3, f"Expected all three runner compression repoints, found {sites}"
    assert all(saved for _, saved in sites), f"Compression repoints without a success-path _save: {sites}"


def test_manual_compression_advance_persists():
    """The /compress caller uses the store advance, not a direct assignment."""
    command = ast.parse(textwrap.dedent(inspect.getsource(slash_commands.GatewaySlashCommandsMixin._execute_compress_command)))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "advance_compression_session"
        for node in ast.walk(command)
    )
    advance = ast.parse(textwrap.dedent(inspect.getsource(gateway_session.SessionStore.advance_compression_session))).body[0]
    assert isinstance(advance, ast.FunctionDef)
    assert len(advance.body) >= 2
    # The successful route falls through the guarded heal and saves before
    # returning the entry; rejected CAS/lineage attempts return inside guards.
    assert isinstance(advance.body[-1], ast.Return)
    assert isinstance(advance.body[-1].value, ast.Name)
    assert advance.body[-1].value.id == "entry"
    assert isinstance(advance.body[-2], ast.Expr)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_save"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        for node in ast.walk(advance.body[-2])
    )
    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Call)
        and isinstance(node.test.operand.func, ast.Attribute)
        and node.test.operand.func.attr == "_heal_compression_tip_locked"
        and any(isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is None for stmt in node.body)
        for node in ast.walk(advance)
    )


def test_repoint_persistence_check_rejects_unsaved_success_path():
    """A save in the failure branch or a nearby unrelated branch is insufficient."""
    sites = _repoint_persistence('''
        def handler(self, entry):
            if self.session_store.repoint_session_entry(entry, "old", "new"):
                pass
            else:
                self.session_store._save()
    ''')
    assert len(sites) == 1
    assert not sites[0][1]
    sites = _repoint_persistence('''
        def handler(self, entry):
            repointed = self.session_store.repoint_session_entry(entry, "old", "new")
            if not repointed:
                pass
            self.session_store._save()
    ''')
    assert len(sites) == 1
    assert not sites[0][1]
    sites = _repoint_persistence('''
        def handler(self, entry):
            if ready:
                repointed = self.session_store.repoint_session_entry(entry, "old", "new")
            if repointed:
                pass
            else:
                self.session_store._save()
    ''')
    assert len(sites) == 1
    assert not sites[0][1]
