"""MCP call provenance (ACP #1037 prerequisite): every outbound tools/call carries a host-derived
_meta block identifying the real caller, and nothing a model or tool-call argument supplies can
forge or override it."""
import pytest

from tools.mcp_call_provenance import PROVENANCE_META_KEY, build_call_provenance, strip_caller_provenance


class TestStripCallerProvenance:
    def test_a_bare_meta_key_is_removed(self):
        cleaned = strip_caller_provenance({"name": "world", "_meta": {"anything": "goes"}})
        assert cleaned == {"name": "world"}

    def test_a_forged_provenance_key_is_removed(self):
        cleaned = strip_caller_provenance({"x": 1, PROVENANCE_META_KEY: {"principal": "owner"}})
        assert cleaned == {"x": 1}

    def test_ordinary_arguments_are_untouched(self):
        assert strip_caller_provenance({"a": 1, "b": "two"}) == {"a": 1, "b": "two"}

    def test_non_dict_input_is_defanged(self):
        assert strip_caller_provenance(None) == {}
        assert strip_caller_provenance("not a dict") == {}

    def test_the_original_dict_is_not_mutated(self):
        original = {"name": "world", "_meta": {"x": 1}}
        strip_caller_provenance(original)
        assert original == {"name": "world", "_meta": {"x": 1}}


class TestBuildCallProvenance:
    def test_an_ordinary_owner_turn_reports_owner_principal_and_zero_depth(self, monkeypatch):
        from gateway import session_context

        monkeypatch.setattr(session_context, "get_session_env",
                            lambda name, default="": {
                                "HERMES_SESSION_ID": "sess-1", "HERMES_SESSION_KEY": "key-1",
                                "HERMES_SESSION_PLATFORM": "telegram", "HERMES_SESSION_CHAT_ID": "chat-1",
                            }.get(name, default))
        from tools import mcp_call_provenance as prov
        monkeypatch.setattr(prov, "_lineage_root_digest", lambda sid: "sha256:stub")
        provenance = build_call_provenance()
        assert provenance["principal"] == "owner"
        assert provenance["delegation_depth"] == 0
        assert provenance["cron"] is False
        assert provenance["session_id"] == "sess-1"
        assert provenance["lineage_root_digest"] == "sha256:stub"

    def test_a_cron_session_reports_cron_true(self, monkeypatch):
        from gateway import session_context
        monkeypatch.setattr(session_context, "get_session_env",
                            lambda name, default="": {"HERMES_CRON_SESSION": "1"}.get(name, default))
        from tools import mcp_call_provenance as prov
        monkeypatch.setattr(prov, "_lineage_root_digest", lambda sid: None)
        assert build_call_provenance()["cron"] is True

    def test_a_peer_turn_never_reports_owner(self, monkeypatch):
        from tools.approval_context import bind_delegation_depth, reset_turn_principal, set_turn_principal
        from gateway import session_context
        monkeypatch.setattr(session_context, "get_session_env", lambda name, default="": default)
        from tools import mcp_call_provenance as prov
        monkeypatch.setattr(prov, "_lineage_root_digest", lambda sid: None)
        token = set_turn_principal("peer")
        try:
            assert build_call_provenance()["principal"] == "peer"
        finally:
            reset_turn_principal(token)

    def test_a_subagent_never_reports_depth_zero(self, monkeypatch):
        from tools.approval_context import bind_delegation_depth, get_delegation_depth
        from gateway import session_context
        monkeypatch.setattr(session_context, "get_session_env", lambda name, default="": default)
        from tools import mcp_call_provenance as prov
        monkeypatch.setattr(prov, "_lineage_root_digest", lambda sid: None)
        before = get_delegation_depth()
        token = bind_delegation_depth(before + 1)
        try:
            assert build_call_provenance()["delegation_depth"] >= 1
        finally:
            bind_delegation_depth(before)

    def test_nested_subagents_increment_depth_through_run_single_child(self, monkeypatch):
        """The real bump site (tools.delegate_tool._run_single_child), not a direct unit call: a
        grandchild dispatched from inside a child's own delegation must report depth 2, not 1."""
        import tools.delegate_tool as dt
        from tools.approval_context import get_delegation_depth

        seen = []

        def fake_lease(child):
            return None, None

        monkeypatch.setattr(dt, "_lease_child_credential", fake_lease)

        class _Boom(Exception):
            pass

        def record_and_raise(*a, **kw):
            seen.append(get_delegation_depth())
            raise _Boom()

        monkeypatch.setattr(dt, "_run_single_child", dt._run_single_child)  # sanity: not monkeypatched away
        # Patch deep enough in the body to observe depth without running a real child turn.
        monkeypatch.setattr(dt, "_lease_child_credential", lambda child: (_ for _ in ()).throw(_Boom()))
        before = get_delegation_depth()
        with pytest.raises(_Boom):
            dt._run_single_child(task_index=0, goal="x", child=object(), parent_agent=None)
        assert get_delegation_depth() == before + 1, "the depth bump must land before the child body runs"
