"""``hermes gateway deliveries``: the owner-message replies the delivery ledger has parked.

A part of a stored owner message's reply (its text or an attachment) that ``MAX_REACTIVATIONS`` later gateway
starts could not deliver is parked (gateway/delivery_ledger.py, ``parked_parts``): no restart tries it again,
and the message's inbound record stays. ``list`` shows the parked parts with the stored message each belongs
to; ``resume`` gives them another bounded set of tries, made by the next gateway start.
"""

from __future__ import annotations

import sqlite3
import sys


def _replies_by_record() -> dict:
    """reply obligation id -> the id of the stored owner message that reply settles."""
    from gateway.delivery_ledger import inbound_record_replies
    from gateway.shutdown_flush import load_inbound_records
    owners = {}
    for record in load_inbound_records():
        try:
            owners.update(dict.fromkeys(inbound_record_replies(record), record["recovery_id"]))
        except Exception:
            continue  # an unreadable record is never settled, so it owns no reply
    return owners


def cmd_gateway_deliveries(args) -> None:
    from gateway import delivery_ledger
    try:
        owners = _replies_by_record()
        inbound = getattr(args, "inbound", None)
        scope = None
        if inbound is not None:
            scope = [oid for oid, owner in owners.items() if owner == inbound]
            if not scope:
                print(f"No stored owner message {inbound}", file=sys.stderr)
                sys.exit(1)
        if getattr(args, "deliveries_command", None) == "resume":
            count = delivery_ledger.resume_parked(scope)
            print(f"Resumed {count} parked reply part(s); the next gateway start tries them again, "
                  f"for up to {delivery_ledger.MAX_REACTIVATIONS} starts.")
            return
        parked = delivery_ledger.parked_parts(scope)
    except (OSError, sqlite3.Error) as exc:
        print(f"Could not read the delivery ledger: {exc}", file=sys.stderr)
        sys.exit(1)
    if not parked:
        print("No parked replies.")
        return
    for part in parked:
        print(f"{owners.get(part['obligation_id'], '-')}  {part['obligation_id']}  {part['id']}  {part['state']}  "
              f"tries={part['tries']}  {part.get('path') or ''}".rstrip())
    print("Resume once the cause is fixed: hermes gateway deliveries resume [--inbound <id>]")
