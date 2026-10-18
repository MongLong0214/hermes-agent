"""#92 review round 2 (H92-01, the reaper path): an unfinished event's ``.inflight`` entry is retired only
once the record that refuses its resubmission is persisted.

That record is the event's existing ``PUBLISHED/<event>`` journal entry of kind ``unfinished_notice``,
read back complete, naming this very event, and fsync'd with its directory. A file that merely exists is
not it, and a posted notice is a separate fact from a persisted record. Until it is persisted the entry
stays — an explicit unresolved state, never success or delivery — and a redelivery is not submitted; a
later reap, once storage works again, retires it normally. No new record kind.

The reviewer's gated ENOSPC counterexample is ``test_h92_r2_review.py`` (byte-identical). These drive the
real shim, handler, coordinator and SQLite in-process (``test_shim_d2_refused_answer_kept``), with failures
injected at the exact call; no sleeps.
"""
import errno
import json
import os

import pytest
import test_shim_d2_refused_answer_kept as o4

gw = o4.gw
shim = o4.shim
_pin_the_module_home = o4._pin_the_module_home


def _orphan(shim, channel=o4.CHANNEL):
    """The entry a shim leaves when it is replaced after submitting: written, linked, no live owner."""
    published = shim.root / "published"
    published.mkdir(exist_ok=True)
    entry = published / (o4.EVENT + ".inflight")
    entry.write_text(json.dumps({"event_id": o4.EVENT, "channel": channel, "pid": 1, "at": "-"}) + "\n")
    entry.chmod(0o600)
    return entry


def _journal(shim):
    return shim.root / "published" / o4.EVENT


def _redelivery_is_not_submitted(gw, shim):
    o4._drive(shim, rpc_id=7)  # buzz-acp redelivers the same, valid single event
    assert f"base={o4.EVENT[:12]}" in (shim.root / "shim.log").read_text(), "reach: attributed"
    assert gw.responses == [] and gw.runs == [], "an unresolved event was submitted again"


def _failing(target, error=errno.ENOSPC):
    def fail(*args, **kwargs):
        raise OSError(error, "controlled storage failure")
    return fail


@pytest.mark.parametrize("channel", [o4.CHANNEL, None], ids=["valid_channel", "channel_less"])
def test_journal_creation_failure_keeps_the_entry_then_a_later_reap_retires_it(gw, shim, monkeypatch, channel):
    gw.actor.shape = "plain"
    entry = _orphan(shim, channel)
    real_open, journal = os.open, str(_journal(shim))

    def no_space_for_the_journal(path, flags, *args, **kwargs):
        if path == journal and flags & os.O_CREAT:
            raise OSError(errno.ENOSPC, "controlled journal creation failure")
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", no_space_for_the_journal)
        shim.reap_unfinished()
    assert not _journal(shim).exists(), "reach: the journal entry was not created"
    assert entry.exists(), "the only guard was retired with no record persisted"
    assert "UNRESOLVED" in (shim.root / "shim.log").read_text()
    _redelivery_is_not_submitted(gw, shim)

    shim.reap_unfinished()  # storage works again
    record = json.loads(_journal(shim).read_text())
    assert (record["kind"], record["event_id"]) == ("unfinished_notice", o4.EVENT)
    assert not entry.exists()
    assert len(shim.posts()) == (1 if channel else 0)
    _redelivery_is_not_submitted(gw, shim)


@pytest.mark.parametrize("channel", [o4.CHANNEL, None], ids=["valid_channel", "channel_less"])
def test_a_journal_entry_claimed_but_not_written_keeps_the_entry(gw, shim, monkeypatch, channel):
    """The post may even have gone out: a posted notice is not a persisted record."""
    gw.actor.shape = "plain"
    entry = _orphan(shim, channel)
    real_fdopen = os.fdopen

    class NoSpace:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()
            return False

        def write(self, _text):
            raise OSError(errno.ENOSPC, "controlled journal write failure")

    def fdopen(fd, mode="r", *args, **kwargs):
        handle = real_fdopen(fd, mode, *args, **kwargs)
        return NoSpace(handle) if "w" in mode else handle

    with monkeypatch.context() as patch:
        patch.setattr(os, "fdopen", fdopen)
        shim.reap_unfinished()
    assert _journal(shim).exists() and _journal(shim).read_text() == "", "reach: claimed, nothing written"
    assert entry.exists(), "an empty journal entry was taken for the record"
    assert len(shim.posts()) == (1 if channel else 0)
    _redelivery_is_not_submitted(gw, shim)


def test_a_journal_record_that_cannot_be_flushed_keeps_the_entry(gw, shim, monkeypatch):
    gw.actor.shape = "plain"
    entry = _orphan(shim)
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", _failing(None, errno.EIO))
        shim.reap_unfinished()
    assert json.loads(_journal(shim).read_text())["kind"] == "unfinished_notice", "reach: written, not flushed"
    assert entry.exists(), "a record not flushed was taken as persisted"
    _redelivery_is_not_submitted(gw, shim)
    shim.reap_unfinished()  # flushing works again: the same record is now persisted
    assert not entry.exists() and len(shim.posts()) == 1


@pytest.mark.parametrize("existing", ["empty", "garbage", "another_event", "answer_kind"])
def test_a_journal_file_that_merely_exists_is_not_the_record(gw, shim, existing):
    gw.actor.shape = "plain"
    entry = _orphan(shim)
    _journal(shim).write_text({
        "empty": "",
        "garbage": "{not json\n",
        "another_event": json.dumps({"kind": "unfinished_notice", "event_id": "f" * 64}) + "\n",
        "answer_kind": json.dumps({"kind": "answer", "event_id": o4.EVENT, "sent": True}) + "\n",
    }[existing])
    shim.reap_unfinished()
    assert entry.exists(), f"a journal file ({existing}) was taken for the resubmission-refusing record"
    assert shim.posts() == []  # the event's one post is already taken
    _redelivery_is_not_submitted(gw, shim)


def test_control_with_working_storage_a_reap_posts_once_and_retires_the_entry(gw, shim):
    gw.actor.shape = "plain"
    entry = _orphan(shim)
    shim.reap_unfinished()
    record = json.loads(_journal(shim).read_text())
    assert (record["kind"], record["event_id"], record["sent"]) == ("unfinished_notice", o4.EVENT, True)
    assert not entry.exists() and len(shim.posts()) == 1
    _redelivery_is_not_submitted(gw, shim)
