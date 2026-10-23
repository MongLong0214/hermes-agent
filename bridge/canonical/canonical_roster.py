"""Validated, request-driven bridge roster reads (also installed beside the standalone shim).

An invalid replacement never becomes authority. Keep the last good snapshot for recovery, but
refuse the current read: using stale grants during a broken retirement would fail open.
"""
import hashlib
import json
import re
import threading
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HANDLE = re.compile(r"[A-Za-z0-9_-]+\Z")


class RosterError(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RosterError("duplicate field")
        result[key] = value
    return result


class Roster:
    def __init__(self, raw, digest):
        if not isinstance(raw, dict):
            raise RosterError("expected object")
        self.version = raw.get("version", 1)  # pre-version v1 bridge fixtures/installations
        if type(self.version) is not int or self.version not in (1, 2):
            raise RosterError("unsupported version")
        self.generation = raw.get("generation", 0) if self.version == 2 else 0
        if type(self.generation) is not int or self.generation < 0:
            raise RosterError("generation must be a nonnegative integer")
        if self.version == 2 and "generation" not in raw:
            raise RosterError("v2 requires generation")
        if self.version == 2 and not isinstance(raw.get("relay"), str):
            raise RosterError("v2 requires relay")
        self.relay = raw.get("relay", "")
        channels = raw.get("channels")
        if not isinstance(channels, dict):
            raise RosterError("channels must be an object")
        rooms = {}
        for channel, rule in channels.items():
            try:
                if str(UUID(channel)) != channel:
                    raise ValueError()
            except (ValueError, TypeError, AttributeError):
                raise RosterError("invalid room UUID") from None
            if not isinstance(rule, dict):
                raise RosterError("room must be an object")
            status = rule.get("status", "active")
            if status not in ("active", "retired") or (self.version == 2 and "status" not in rule):
                raise RosterError("invalid room status")
            for role in ("ceo", "peer"):
                identity = rule.get(role)
                if (not isinstance(identity, dict)
                        or not isinstance(identity.get("pubkey"), str)
                        or not HEX64.fullmatch(identity["pubkey"])
                        or not isinstance(identity.get("handle"), str)
                        or not HANDLE.fullmatch(identity["handle"])):
                    raise RosterError("invalid " + role + " identity")
            if rule["ceo"]["pubkey"] == rule["peer"]["pubkey"]:
                raise RosterError("CEO and peer must differ")
            if self.version == 2:
                for key in ("relay_name", "acp_project_id", "classification"):
                    if not isinstance(rule.get(key), str):
                        raise RosterError("v2 requires room " + key)
                if rule["ceo"]["handle"] != "ceo":
                    raise RosterError("v2 CEO handle must be ceo")
            rooms[channel] = MappingProxyType({
                **rule, "status": status,
                "ceo": MappingProxyType(dict(rule["ceo"])),
                "peer": MappingProxyType(dict(rule["peer"])),
            })
        self.channels = MappingProxyType(rooms)
        known = raw.get("known_profiles", {}) if self.version == 1 else {}
        if not isinstance(known, dict) or any(
                not isinstance(handle, str) or not HANDLE.fullmatch(handle)
                or not isinstance(key, str) or not HEX64.fullmatch(key)
                for handle, key in known.items()):
            raise RosterError("invalid legacy known profiles")
        self.known_profiles = MappingProxyType(dict(known))
        self.digest = digest

    def active(self, channel):
        room = self.channels.get(channel)
        return room if room and room["status"] == "active" else None

    def admits(self, author, channel, owner):
        room = self.active(channel)
        return bool(room and author in (room["peer"]["pubkey"], owner))


class RosterReader:
    def __init__(self, path):
        self.path = Path(path)
        self.last_good = None
        self._lock = threading.Lock()

    def read(self):
        # One open observes one atomic writer generation. No timer, polling task or TTL window.
        with self._lock:
            try:
                data = self.path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                previous = self.last_good
                if previous is not None and digest == previous.digest:
                    return previous
                roster = Roster(json.loads(data, object_pairs_hook=_object), digest)
                if previous is not None and previous.version == 2:
                    if roster.version != 2 or roster.generation < previous.generation:
                        raise RosterError("generation rollback")
                    if roster.generation == previous.generation:
                        raise RosterError("generation reused with different contents")
                self.last_good = roster
                return roster
            except (OSError, ValueError, TypeError, RecursionError) as exc:
                # Category and validation reason only; never echo roster contents or filesystem paths.
                reason = str(exc) if isinstance(exc, RosterError) else type(exc).__name__
                raise RosterError("roster unavailable: " + reason) from exc


def read_owner(path):
    try:
        owner = Path(path).read_text().strip()
    except OSError as exc:
        raise RosterError("owner public identity unavailable: " + type(exc).__name__) from exc
    if not HEX64.fullmatch(owner):
        raise RosterError("invalid owner public identity")
    return owner
