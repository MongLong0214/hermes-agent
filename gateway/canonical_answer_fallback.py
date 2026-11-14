"""The local fallback that keeps a canonical answer when its state.db refuses it (#1110).

A canonical turn whose terminal record did not land keeps its answer beside the receipt, in the
same state.db. When that write fails too (the database that just failed the terminal), the answer is
written here instead: a file beside the binding's state.db, never through it.

- One file per event and claim: ``canonical-answer-kept/<receipt digest>.<claim owner>.json``. The
  name binds the event and the claim that ran it, so no other claim's file is ever written over.
- The directory is 0700 and the file 0600, both owned by this user. Every open refuses a symlink, the
  file is created exclusively, fsync'd, published by ``link`` (which never replaces a name), the
  directory is fsync'd, and the bytes are read back before the request is answered.
- Nothing here delivers or resends. A replay reads the file only to answer the event's explicit
  failure: kept is not delivered.
"""

import os
import re
import secrets
import stat
from pathlib import Path

DIR_NAME = "canonical-answer-kept"
_NAME = re.compile(r"(?P<digest>[0-9a-f]{64})\.(?P<owner>[0-9a-f]{32})\.json\Z")
# A kept record is bounded by the coordinator (_MAX_KEPT_CHARS of JSON, at most 4 UTF-8 bytes each).
# It holds the whole answer the actor returned, not the receipt's capped summary of it.
_MAX_BYTES = 4 * (6 * (1 << 20) + 4_096 + 16_384)
_NOFOLLOW = os.O_NOFOLLOW | os.O_CLOEXEC


class FallbackUnsafe(OSError):
    """The fallback directory or file is not this user's private, regular, unlinked-elsewhere entry."""


def _open_dir(home: Path, *, create: bool) -> int | None:
    """The fallback directory's fd, refusing a symlink or a directory others can reach; None when it
    does not exist and ``create`` is false. ``home`` (the state.db's own directory) may itself be a
    symlinked home, as the state.db path may; only the entries this module owns are never followed."""
    parent = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        created = False
        if create:
            try:
                os.mkdir(DIR_NAME, 0o700, dir_fd=parent)
                created = True
            except FileExistsError:
                pass
        try:
            fd = os.open(DIR_NAME, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW, dir_fd=parent)
        except FileNotFoundError:
            if create:
                raise
            return None
        if created:
            os.fsync(parent)
    finally:
        os.close(parent)
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        os.close(fd)
        raise FallbackUnsafe(f"{DIR_NAME} is not a private directory of this user")
    return fd


def _read(dir_fd: int, name: str) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | _NOFOLLOW, dir_fd=dir_fd)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or info.st_size > _MAX_BYTES):
            raise FallbackUnsafe(f"{name} is not a private regular file of this user")
        chunks, size = [], 0
        while chunk := os.read(fd, 65_536):
            size += len(chunk)
            if size > _MAX_BYTES:
                raise FallbackUnsafe(f"{name} is over its bound")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def keep(home: Path, digest: str, owner: str, data: bytes) -> str:
    """Write ``data`` as the event's kept answer for this claim and read it back; the file's name
    relative to ``home``. Raises when the answer is not on disk exactly as given: an existing file of
    the same name is never replaced, and a symlink is never followed."""
    name = f"{digest}.{owner}.json"
    if _NAME.fullmatch(name) is None or len(data) > _MAX_BYTES:
        raise ValueError("canonical_fallback_key_invalid")
    dir_fd = _open_dir(home, create=True)
    try:
        temporary = f".{name}.{secrets.token_hex(8)}.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600, dir_fd=dir_fd)
        try:
            try:
                os.fchmod(fd, 0o600)
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            # link() publishes the complete file under its name in one step and fails when the name
            # exists (a file or a symlink), so nothing is ever written over or through.
            os.link(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
        finally:
            os.unlink(temporary, dir_fd=dir_fd)
        os.fsync(dir_fd)
        if _read(dir_fd, name) != data:
            raise FallbackUnsafe(f"{name} does not read back as written")
    finally:
        os.close(dir_fd)
    return f"{DIR_NAME}/{name}"


def kept_for(home: Path, digest: str) -> list[tuple[str, bytes]]:
    """Every (claim owner, bytes) kept for the event ``digest``, in name order. Raises when the
    directory or a file for the event is unsafe or unreadable: unknown is never read as nothing."""
    dir_fd = _open_dir(home, create=False)
    if dir_fd is None:
        return []
    try:
        names = sorted(name for name in os.listdir(dir_fd)
                       if (match := _NAME.fullmatch(name)) is not None and match["digest"] == digest)
        return [(_NAME.fullmatch(name)["owner"], _read(dir_fd, name)) for name in names]
    finally:
        os.close(dir_fd)
