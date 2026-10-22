"""Roster state remains bound to its config home under live A→B→A profile alternation."""
from gateway.config import GatewayConfig
from gateway.run import _profile_runtime_scope
from agent.secret_scope import is_multiplex_active, set_multiplex_active
from tests.bridge.test_canonical_roster import A, B, CTO, OTHER, OWNER, room, document, replace


def test_roster_readers_do_not_borrow_another_profiles_home_or_generation(tmp_path):
    homes = [tmp_path / 'a', tmp_path / 'b']
    bindings = []
    active = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for home, channel, author, generation in [(homes[0], A, CTO, 9), (homes[1], B, OTHER, 2)]:
            bridge = home / 'bridge'
            (bridge / 'keys').mkdir(parents=True)
            (bridge / 'keys/owner.pub').write_text(OWNER)
            replace(bridge / 'required-mentions.json', document(generation=generation, channels={channel: room(author)}))
            with _profile_runtime_scope(home, {}):
                bindings.append(GatewayConfig.from_dict({'canonical_surface_bindings': {'acp-canonical-ceo': {
                    'session_key': 'agent:main:telegram:dm:owner', 'session_id': 'existing',
                    'telegram': {'chat_id': 'owner', 'chat_type': 'dm', 'user_id': 'owner', 'thread_id': None},
                    'buzz': {'author_ids': ['legacy'], 'channel_ids': ['legacy-room']},
                }}}).canonical_surface_bindings['acp-canonical-ceo'])
        for home in [homes[0], homes[1], homes[0]]:
            with _profile_runtime_scope(home, {}):
                assert bindings[0].admits(CTO, A) and not bindings[0].admits(OTHER, B)
                assert bindings[1].admits(OTHER, B) and not bindings[1].admits(CTO, A)
                assert [b.roster.last_good.generation for b in bindings] == [9, 2]
                assert all(b.admits('legacy', 'legacy-room') for b in bindings)
    finally:
        set_multiplex_active(active)
