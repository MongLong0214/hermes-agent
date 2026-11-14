"""Offline follow-up of the repaired ownership and suffix paths."""
import errno
import json
import os
import select
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import test_shim_d2_refused_answer_kept as o4
import test_h92_r1_review as h92
from test_h92_r1_review import _reaper

gw = o4.gw
shim = o4.shim
_pin_the_module_home = o4._pin_the_module_home
gated = h92.gated


@pytest.mark.parametrize("stage", ["write", "fsync", "link"])
def test_claim_failure_closes_fd_and_removes_staging_file(shim, monkeypatch, stage):
    opened = []
    real_open = os.open

    def capture(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "controlled failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", capture)
        patch.setattr(os, stage, fail)
        fd, reason = shim._claim_inflight(o4.EVENT, o4.CHANNEL)
    assert fd is None and reason
    assert list((shim.root / "published").iterdir()) == []
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
    fd, reason = shim._claim_inflight(o4.EVENT, o4.CHANNEL)
    assert fd is not None and reason is None
    shim._drop_inflight(o4.EVENT, fd)


def test_eexist_is_not_ownership_and_does_not_release_the_winners_lock(shim):
    fd, reason = shim._claim_inflight(o4.EVENT, o4.CHANNEL)
    assert fd is not None and reason is None
    try:
        path = shim.root / "published" / (o4.EVENT + ".inflight")
        before = path.read_bytes()
        other, reason = shim._claim_inflight(o4.EVENT, o4.CHANNEL)
        assert other is None and reason
        assert list(path.parent.iterdir()) == [path]
        assert path.read_bytes() == before
        _reaper(shim)
        assert path.exists() and shim.posts() == []
    finally:
        shim._drop_inflight(o4.EVENT, fd)


def test_threads_and_other_fds_cannot_release_a_live_claim(shim):
    fd, reason = shim._claim_inflight(o4.EVENT, o4.CHANNEL)
    assert fd is not None and reason is None
    path = shim.root / "published" / (o4.EVENT + ".inflight")
    try:
        worker = threading.Thread(target=shim.reap_unfinished)
        worker.start()
        worker.join(10)
        assert not worker.is_alive()
        another_fd = os.open(path, os.O_RDONLY)
        os.close(another_fd)
        assert not os.get_inheritable(fd)
        _reaper(shim)
        assert path.exists() and shim.posts() == []
    finally:
        shim._drop_inflight(o4.EVENT, fd)
    assert not path.exists()


def test_a_kill_after_link_leaves_inert_claim_and_reaper_still_prevents_resubmission(gw, shim):
    code = f'''
import importlib.util, os, threading
spec = importlib.util.spec_from_file_location("claim_child", {str(o4._SHIM)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
real_link = os.link
def pause_after_link(*args, **kwargs):
    real_link(*args, **kwargs)
    print("linked", flush=True)
    threading.Event().wait(30)
os.link = pause_after_link
module._claim_inflight({o4.EVENT!r}, {o4.CHANNEL!r})
'''
    process = subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ),
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert select.select([process.stdout], [], [], 10)[0]
        assert process.stdout.readline() == b"linked\n"
        process.kill()
        process.wait(10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(10)
        process.stdout.close()
        process.stderr.close()
    directory = shim.root / "published"
    [claim] = directory.glob("*.claim")
    assert json.loads(claim.read_text())["event_id"] == o4.EVENT
    _reaper(shim)
    assert claim.exists()  # The filename filter never reaps staging files.
    assert not (directory / (o4.EVENT + ".inflight")).exists()
    assert json.loads((directory / o4.EVENT).read_text())["kind"] == "unfinished_notice"
    o4._drive(shim)
    assert gw.runs == [] and gw.responses == [] and len(shim.posts()) == 1


@pytest.mark.parametrize("role", ["assistant", "tool"])
@pytest.mark.parametrize("marker", ["kind", "metadata"])
def test_peer_markers_on_other_suffix_roles_refuse_preservation(gw, shim, monkeypatch, role, marker):
    original = gw.actor.run_conversation

    def inject(*args, **kwargs):
        result = original(*args, **kwargs)
        row = {"role": role, "content": "foreign row"}
        if marker == "kind":
            row["display_kind"] = "canonical_peer"
        else:
            row["display_metadata"] = {"canonical_peer": {"event_id": "f" * 64}}
        result["messages"].insert(-1, row)
        return result

    monkeypatch.setattr(gw.actor, "run_conversation", inject)
    o4._drive(shim)
    o4._reached_the_refusal(gw, shim)
    assert gw.bound == [None] and o4._held(gw) == []
    assert o4._codes(gw) == [(409, "canonical_event_failed")]
    assert len(shim.posts()) == 1


def test_peer_marker_on_the_final_row_refuses_preservation(gw, shim, monkeypatch):
    original = gw.actor.run_conversation

    def inject(*args, **kwargs):
        result = original(*args, **kwargs)
        result["messages"][-1]["display_metadata"] = {"canonical_peer": {"event_id": "f" * 64}}
        return result

    monkeypatch.setattr(gw.actor, "run_conversation", inject)
    o4._drive(shim)
    o4._reached_the_refusal(gw, shim)
    assert gw.bound == [None] and o4._held(gw) == []
    assert o4._codes(gw) == [(409, "canonical_event_failed")]


def test_reaper_journal_failure_cannot_erase_submission_and_allow_resubmission(gated):
    first = gated.spawn("before-journal-failure")
    session = first.start_session()
    first.prompt(2, session, h92.o2.channel_prompt(h92.o2.E1, h92.o2.ROOT_TEXT))
    assert gated.gw.claimed.wait(10)
    first.kill()
    code = f'''
import errno, importlib.util, os
spec = importlib.util.spec_from_file_location("reaper_failed_marker", {str(o4._SHIM)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
real_open = os.open
def fail_marker_open(path, flags, *args, **kwargs):
    if path == os.path.join(module.PUBLISHED, {h92.o2.E1!r}) and flags & os.O_CREAT:
        raise OSError(errno.ENOSPC, "controlled journal creation failure")
    return real_open(path, flags, *args, **kwargs)
os.open = fail_marker_open
module.reap_unfinished()
'''
    reaper = subprocess.run([sys.executable, "-c", code], env=gated.env,
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    assert reaper.returncode == 0, reaper.stderr
    assert not (gated.root / "published" / h92.o2.E1).exists()
    remaining = sorted(p.name for p in (gated.root / "published").iterdir())
    second = gated.spawn("after-journal-failure")
    session = second.start_session()
    second.prompt(3, session, h92.o2.channel_prompt(h92.o2.E1, h92.o2.ROOT_TEXT))
    assert second.wait_result(3, 30) is not None
    assert gated.gw.pending(h92.o2.E1), "the original gateway turn must still be running"
    received = list(gated.gw.received)
    print("REAPER_MARKER_FAILURE", json.dumps({"remaining_after_reap": remaining,
        "received_while_running": received, "runs": gated.gw.runs,
        "responses": gated.gw.responses, "posts": gated.posts()}))
    assert received == [h92.o2.E1], "reaper discarded the only guard when its durable notice could not be created"
