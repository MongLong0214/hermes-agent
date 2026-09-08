"""Deferred and exhausted compression results retain actor and routing.

A lock-contended defer is transient. Exhaustion explicitly blocks this request;
neither result authorizes reset, cache eviction, or topic rebinding.
"""
import ast
from pathlib import Path


def test_deferred_and_exhausted_branches_never_mutate_actor():
    path = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    tree = ast.parse(path.read_text())
    chains = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Call)
              and isinstance(n.test.func, ast.Attribute)
              and isinstance(n.test.func.value, ast.Name)
              and n.test.func.value.id == "agent_result"
              and n.test.args and isinstance(n.test.args[0], ast.Constant)
              and n.test.args[0].value == "compression_deferred" and n.orelse]
    assert len(chains) == 1
    chain = chains[0]
    assert any(isinstance(n, ast.Constant) and n.value == "compression_exhausted"
               for n in ast.walk(chain.orelse[0].test))
    calls = {n.func.attr for n in ast.walk(chain)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"reset_session", "_evict_cached_agent",
                        "_clear_conversation_scope", "_sync_telegram_topic_binding"}


import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


def result_consumer():
    # Compile the actual deferred/exhausted result-consumer, not a copy or a
    # replacement helper. This seam is otherwise inside a large async handler.
    path = Path(__file__).resolve().parents[2] / 'gateway' / 'run.py'
    tree = ast.parse(path.read_text())
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Call)
                  and isinstance(n.test.func, ast.Attribute)
                  and isinstance(n.test.func.value, ast.Name)
                  and n.test.func.value.id == 'agent_result'
                  and n.test.args and isinstance(n.test.args[0], ast.Constant)
                  and n.test.args[0].value == 'compression_deferred'
                  and n.orelse]
    assert len(candidates) == 1
    wrapper = ast.parse('async def consume(self, agent_result, session_entry, session_key, source, response):\n    return response, session_entry\n')
    wrapper.body[0].body.insert(0, candidates[0])
    ast.fix_missing_locations(wrapper)
    namespace = {'logger': logging.getLogger(__name__)}
    exec(compile(wrapper, str(path), 'exec'), namespace)
    return namespace['consume']


def forbidden(*args, **kwargs):
    pytest.fail('compression failure attempted a session mutation')


@pytest.mark.asyncio
@pytest.mark.parametrize('deferred', [False, True])
async def test_exhaustion_preserves_actor_and_routing(deferred):
    consume = result_consumer()
    entry = SimpleNamespace(session_id='same-actor')
    source = SimpleNamespace(chat_id='chat', thread_id='topic')
    runner = SimpleNamespace(
        async_session_store=SimpleNamespace(reset_session=forbidden),
        _evict_cached_agent=forbidden, _clear_conversation_scope=forbidden,
        _sync_telegram_topic_binding=forbidden,
    )
    result = {'compression_exhausted': True, 'compression_deferred': deferred}
    for _ in range(3):
        response, returned_entry = await consume(runner, result, entry, 'same-key', source, 'original failure')
        assert returned_entry is entry
        assert entry.session_id == 'same-actor'
        assert source.thread_id == 'topic'
        assert 'original failure' in response
        if not deferred:
            assert 'blocked' in response.lower()
            assert 'same session' in response.lower()
            assert '/compress' in response
            assert 'cooldown' in response.lower()
        assert 'auto-reset' not in response.lower()
        assert '/new' not in response
