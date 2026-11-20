# Implementer response to reviews r1, r2, r3, r4, the slice 1 reviews, the s1d review (slice A), the slice A review and the slice A2 review

Commit history on `fix/owner-input-durable-claim`, none of it pushed:
- `479a19db05`: the first change.
- `1600122bce`: fixes for review r1.
- `1524ad041a`: fixes for review r2.
- `36ed681e89`: fixes for review r3.
- `f89a57ae87`: fixes for review r4, findings 1, 2 and 4 (slice 1).
- `5bc45c6250`: slice 1 follow-up: the in-turn first response, the run_turn.py read, and one obligation per run.
- `ace7c2f5a5`: slice 1 follow-up 2: a turn cancelled or raising after its answer was stored, and the run of a rewritten event.
- `b9a0879622`: slice 1 structural correction (the slice 1 review of ace7c2f5): one settlement boundary.
- `2aa38989c0`: slice 1 component obligations and ledger-first recovery (the slice 1 review of b9a08796).
- `7289dc6b8c`: slice 1 one completion decision (the slice 1 review of 2aa38989).
- `c9760626b4`: follow-up to 7289dc6b: a notice after a stored answer never closes the message; attachment tries bounded.
- `1f70e1e476`: slice A (the s1d review of c9760626, findings 1 and 3): attachment obligations and send authority.
- `8c746b3ba3`: slice A correction (the slice A review of 1f70e1e4, P1-1 and P1-2): every declaration owed, and each send bound to the bytes its check approved.
- `95fdcb86d6`: slice A2 correction (the slice A review of 8c746b3b, P1 and P2): declarations bounded by grammar alone, and snapshot cleanup that survives cancellation.

## Slice A2 correction (the slice A review of 8c746b3b, P1 and P2)

Slice B is untouched: `claim_attachment_redelivery`, `MAX_REACTIVATIONS`, `reactivate_expired` and
`_prune_unlocked` are unchanged, and `gateway/delivery_ledger.py` did not change (a component's payload already
keeps every field it is given). Only `gateway/platforms/base.py` and the owner/peer test file changed.

### P1: where a declared path ends is decided by the tag grammar alone

`declared_media` no longer calls `_match_extensionless_path` (which validates candidates on disk) and no longer
reads any file state. It returns `(path, is_voice, ambiguous)`. The grammar:

- A quoted path (`` MEDIA:`…` ``, `MEDIA:"…"`, `MEDIA:'…'`) ends at its closing quote (unchanged).
- A whole tag in inline code (`` `MEDIA:/…/report with spaces.log` ``) ends at the code span's closing
  backtick, spaces included. These spans are read first and then blanked, so the patterns below never run
  over them (the bare pattern used to run past the closing backtick into later text). A relative path there
  is not a declaration, as for every other form.
- A bare path with a deliverable extension ends at that extension (`MEDIA_TAG_CLEANUP_RE`, unchanged).
- Any other bare path ends where `MEDIA_EXTENSIONLESS_TAG_RE` ends it. If its line goes on after it (up to
  the next `MEDIA:`) with anything other than whitespace and closing punctuation, the tag could also name a
  longer path with spaces (#24032), and no file may decide which. That tag is **ambiguous**. It is declared
  under the rest of its line, stored with `"ambiguous": true` on its component, and refused at every send
  (`_snapshot_owed_attachment` returns `blocked` without looking at any file). It is logged, the chat is told
  once, and the record stays.

The three places that store an owner reply's components (adapter final, `owe_explicit_reply`,
`extract_reply_parts`) go through `_owed_media`. It checks each clear declaration with `_reply_media`, as
before, and keeps the ambiguous ones under their declared text without any check. The ambiguity travels as
`_ExtractedResponse.ambiguous`, goes into `components()`, and is read back by `of_attachments`, so a later
start refuses the declaration too.

### P2: snapshot cleanup survives cancellation

- `_snapshot_owed_attachment` awaits `asyncio.to_thread(prepare)` directly. There is no separate worker task
  and no done-callback any more. `_SnapshotHandoff` sits between the worker and the send, guarded by one lock:
  - the worker hands over what `snapshot_attachment` made;
  - a send whose wait ends without the snapshot (cancelled) abandons it and re-raises;
  - whichever of the two comes second removes the copy.

  So a copy the worker writes after the send, or its asyncio future, was cancelled is removed by the worker
  itself. That holds even when it writes the copy while `asyncio.run` is joining the executor at shutdown.
  The cancellation is never turned into a send. A copy the send already holds is removed by `upload`'s
  `finally`, as before.
- `_AttachmentSnapshot.discard` logs the directory and the error when the removal fails, where it was
  silent before.
- `_cleanup_cache_dir` (the hourly document cache cleanup, 24 h) now also removes stale `hermes-attachment-*`
  directories. That covers ones a send could not remove and ones a killed process left. It removes no other
  directory, follows no symlink (rmtree refuses one), and leaves a recent snapshot alone.

### Remaining limits (stated, not fixed)

- **Ambiguous bare tags are refused, even when the intended file is there.** One example is
  `MEDIA:/tmp/run.log (12 KB)`: an unknown extension with prose after it on the same line. At 8c746b3b such
  a tag was sent when `/tmp/run.log` validated, which meant existence decided where the path ended. Now it is
  recorded blocked, the chat is told, and the record stays. The retry bound of slice B applies. A model has to
  put such a path on its own line, in inline code or in quotes. Deliverable-extension tags, quoted tags,
  inline-code tags and bare tags that end their line (or meet the next `MEDIA:`) are not affected.
- **A whole tag in inline code is the whole code span.** `` `MEDIA:/a/x.png (chart)` `` declares
  `/a/x.png (chart)`, which is then missing. At 8c746b3b the deliverable-extension pattern stopped at `.png`.
  Two tags in one code span declare one path.
- **What the owner sees in the text is still `extract_media`'s** (unchanged). Suppose a file exists at the
  prefix (`…/report`) of a declared `` `MEDIA:…/report with spaces.log` ``. Then `extract_media`'s
  validated extension strips only `MEDIA:…/report` from the shown text, and ` with spaces.log`` stays
  visible. The upload is the declared report.
- **The grammar's own bounds are the old patterns'.** A bare deliverable-extension path may still run across
  spaces to the first deliverable extension on its line (`MEDIA_TAG_CLEANUP_RE`, unchanged). That is
  file-independent, but it is not the only reading of such a line.
- **Snapshot cleanup.** A snapshot the send could not remove, or one a process killed mid-send (SIGKILL,
  power loss) left, waits up to the cleanup's 24 hours. A send still holding a snapshot after 24 hours would
  lose it to the cleanup; no upload runs that long. The handoff lock rules out a snapshot that both sides
  keep, or that both remove. No test can order that race; the two orders it allows are each witnessed (M2a,
  M2b).
- The previous section's other limits hold: in-place writes before the copy, hard links, same-user
  processes, disk.

### Tests (owner/peer file)

New: 7 functions, 28 cases. One existing test changed: `test_a_send_cancelled_while_its_snapshot_is_made_leaves_no_copy`
now orders its cancellation and its late snapshot with events (`asyncio.to_thread(event.wait)` and a hook on
`discard`) instead of polling waits. Its assertions are unchanged, plus one: nothing reaches the transport.

| Test | Covers | At 8c746b3b |
|---|---|---|
| `test_a_declaration_is_parsed_by_the_tag_grammar_alone[8 replies × nothing / every_prefix on disk]` | the grammar, unit-level. The same answer comes back whether nothing is on disk or every prefix of every path is: inline code spaced, inline code followed by a later `.png`, quoted, bare spaced (ambiguous), bare followed by prose (ambiguous), two bare tags on one line, deliverable extension with spaces, relative inline code | 8 of 16 fail: inline code spaced, inline code + later `.png` (the bare pattern ran past the backtick), bare spaced, bare + prose, each with nothing and with every prefix on disk. The quoted, two-tag, deliverable-extension and relative cases pass (regression guards) |
| `test_a_spaced_inline_declaration_is_owed_whole_and_a_file_at_its_prefix_is_never_sent[normal / queued / recovery × missing / blocked]` | P1, the fixed counterexample. The original and its prefix are absent (or the original is refused), so the obligation is stored under the whole path. The unrelated prefix file `report` then appears and is allowed: the next start does not send it, and the original stays missing. Once the original is there and allowed, it alone is uploaded, the record goes, and the message is never run again | 6 fail: the component is stored as `…/report` |
| `test_a_spaced_inline_declaration_whose_file_is_there_uploads_that_file[absent / present]` | the normal case: the real spaced file exists (and, in `present`, an unrelated prefix file too). It is uploaded and the record goes | `present` fails (the prefix file is uploaded); `absent` passes (regression guard) |
| `test_a_bare_declaration_whose_line_goes_on_is_refused_as_ambiguous` | the bare spaced tag with both files there and allowed: recorded blocked with `ambiguous`, the chat is told, nothing is uploaded at this start or the next, and the record stays | fails: the prefix file is uploaded and the record deleted |
| `test_a_snapshot_whose_send_is_cancelled_while_it_is_made_is_removed_by_its_late_worker` | P2, the reviewer's case, deterministic. The copy is held until the loop's executor shutdown (`asyncio.run`) releases it, so it is written after the send and its future were cancelled. The send ends cancelled, the late copy and its directory are gone, and the source and another run's snapshot stay | fails: the late copy stays |
| `test_a_snapshot_made_for_a_send_cancelled_before_it_was_handed_over_is_removed` | after creation. The executor's future cancels the send as the worker ends, ahead of the loop's own callback. The send ends cancelled and the snapshot it never took is removed | passes (regression guard: the old done-callback handled this order) |
| `test_a_snapshot_that_cannot_be_removed_is_logged_and_reclaimed_by_the_document_cache_cleanup` | a failed removal is logged with its directory and error. `cleanup_document_cache(24)` then removes that stale snapshot directory only: never the source, another run's recent snapshot, or another directory | fails: the removal is silent and the cleanup ignores directories |

### Checks

Tests ran through a copy of `hermes-pytest.sh` that uses `/private/tmp/hermes-owner-review-venv/bin/python`
(Python 3.12) with the same isolation: `env -i`, a scratch HOME, `-B`, no cacheprovider. The prescribed
wrapper's venv has no `pyyaml`, which this clone imports at collection (`hermes_cli/config.py`), so it fails
before any test runs.

| Check | Result |
|---|---|
| RED at 8c746b3b (the final test file on its sources) | 18 of the 28 new cases fail, as in the table. 169 pass, including the changed existing test and the 10 regression guards |
| Owner/peer file at head | 187 passed (159 of 8c746b3b + 28 new) |
| Neighbours (57 files that touch the ledger, inbound records, restart/flush, media extraction or the media caches) | 32 failed, 1162 passed, 3 skipped, at 8c746b3b and at head. The failure sets are identical: 28 in `test_acp_managed_ingress.py::TestReceiptWireAgainstAcp`, plus wecom ×1 and weixin ×3 (environment, not this change) |
| Mutations (on a copy of the tree) | 16 of 16 killed, below |
| ruff | not_run: no ruff in either interpreter |
| commitlore context | run on every edited file before it was edited |
| commitlore guard (with a proposal) | incomplete on every file: the notes mirror is not fetched, and there is no network |
| commitlore capture | not attempted. The clone has no hooks installed to apply a record (`commitlore commit` exits 2 without one), and installing them writes outside the owned files. Committed through `commitlore commit --none`; the decision context is in the commit body and above |

| Rule | Mutant | Killed by |
|---|---|---|
| P1 | M1a: inline code does not bound a whole tag | `tag_grammar_alone[inline ×4]`, `spaced_inline*` |
| P1 | M1b: no anchor check for an inline-code path | `tag_grammar_alone[relative ×2]` |
| P1 | M1c: whole-tag code spans not blanked before the patterns | `tag_grammar_alone[inline + later .png ×2]` |
| P1 | M1d: no ambiguity (the bare token is the path) | `tag_grammar_alone[bare ×4]`, `refused_as_ambiguous` |
| P1 | M1e: the rest of the line is not cut at the next `MEDIA:` | `tag_grammar_alone[two tags ×2]` |
| P1 | M1f: only whitespace closes a bare path | `tag_grammar_alone[two tags ×2]` |
| P1 | M1g: an ambiguous declaration is not refused at its send | `refused_as_ambiguous` |
| P1 | M1h: `components()` drops the ambiguous flag | `refused_as_ambiguous` |
| P1 | M1i: `of_attachments` drops the ambiguous flag | `refused_as_ambiguous` (the next start) |
| P2 | M2a: a snapshot written after its send was abandoned is kept | `late_worker`, `while_its_snapshot_is_made` |
| P2 | M2b: a snapshot made before its send was abandoned is kept | `handed_over` |
| P2 | M2c: the cancellation is swallowed (returns missing) | `late_worker`, `handed_over` |
| P2 | M2d: a failed removal is silent | `cannot_be_removed` |
| P2 | M2e: the cache cleanup skips snapshot directories | `cannot_be_removed` |
| P2 | M2f: the cache cleanup removes any stale directory | `cannot_be_removed` |
| P2 | M2g: the cache cleanup removes a snapshot directory of any age | `cannot_be_removed` |

The reviewer's probe file (`/private/tmp/rv-sa2-review-probes.py`) was used as a guide and not run as it is.
Its P2 probe expects exactly one separate worker task (`assert len(workers) == 1`), and there is none now.
`test_a_snapshot_whose_send_is_cancelled_while_it_is_made_is_removed_by_its_late_worker` is the same
counterexample: the copy is released by the executor shutdown that `asyncio.run` performs, instead of by
that task's done-callback. The P1 probe is covered by the `blocked` cases of the counterexample test.

## Slice A correction (the slice A review of 1f70e1e4, P1-1 and P1-2)

Slice B is untouched: `claim_attachment_redelivery`, `MAX_REACTIVATIONS`, `reactivate_expired` and
`_prune_unlocked` are unchanged. Only `gateway/platforms/base.py` and the owner/peer test file changed.
This replaces two things in the slice A section below: its check "right before each send" (it checked the
whole batch before the first send) and its open item on the check-to-upload window.

### Rule 1: every `MEDIA:` declaration is owed before extraction or filtering (P1-1)

- `BasePlatformAdapter.declared_media(content)` (new) parses what the raw reply declares, apart from
  whether it can be sent: every `MEDIA:` tag outside fenced code, blockquotes and JSON string values, whatever
  its extension, whether or not its file exists, and whether or not the policy allows it. A whole tag in inline
  code counts as a declaration (#35695 already calls it a directive) without the validation `extract_media`'s
  mask applies (`_mask_protected_spans(declarations=True)`). When an unknown-extension path validates,
  `declared_media` uses the path `extract_media` would take (paths with spaces, #24032). Otherwise it uses the
  path up to the next whitespace.
- An owner message's reply passes `declared_media` to `_reply_media` in all three places that store
  components: the adapter's final (`_extract_response_content`), the queued first response and streamed final
  (`owe_explicit_reply`), and a later start reading a stored answer (`extract_reply_parts`). An unsendable file
  is stored under its declared path. Its send records it `missing` or `blocked`, so the record stays after the
  text is delivered (`inbound_completion`, unchanged).
- Unchanged: `extract_media`, its allowed extensions, the text a reply shows, and replies that owe nothing. An
  unknown-extension tag that does not validate still stays visible in the text, as before, and it is now owed
  as well.

### Rule 2: the check is bound to the bytes that are sent (P1-2)

- **Each attachment is checked right before its own send.** `_deliver_owed_attachments` no longer checks the
  batch up front. It gives `_deliver_attachments` an `upload(component id)` async context manager.
  `_deliver_media_attachments` enters it right before each individual send and leaves it when that send ends.
  That covers each image of an owed reply on its own, then each document, voice note or video. On entry it
  makes a snapshot off the loop, inside the routed profile's media scope (`_snapshot_owed_attachment`). If the
  file is refused, the attachment is recorded `missing` or `blocked` and nothing is sent.
- **`snapshot_attachment(path, session_key)`** (new) does five things in order:
  1. Runs `attachment_verdict`.
  2. Opens the resolved path with `O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC`.
  3. Checks again, after the open, that the path still resolves to itself (no symlink anywhere on it) and
     that it names the open file (a regular file with the same `st_dev`/`st_ino`).
  4. Copies the open file into a private snapshot directory inside the document cache
     (`tempfile.mkdtemp(dir=DOCUMENT_CACHE_DIR)`, 0700; `DOCUMENT_CACHE_DIR` is one of `MEDIA_DELIVERY_SAFE_ROOTS`). The copy keeps the file's own name, so the
     upload name and the type routing stay the same.
  5. Closes the fd.

  If anything changed between the check and the open, it returns `blocked` (a file is there) or `missing`.
- **Why a snapshot and not the fd.** The adapters open the file again by its path: `send_document(file_path=…)`,
  `send_voice`, `send_video`, and `send_image_file` through `file://`. A validated fd alone would not bind what
  they read. So each one is given the snapshot's path. One adapter checks that path again: photon's
  `_sidecar_send_attachment` runs `validate_media_delivery_path` on it. That is why the snapshot lives in the
  document cache, which the policy allows in every mode; a snapshot in the system temp directory would be
  refused under a strict policy without recent-file trust (witnessed by M2f). No other adapter in
  `gateway/platforms/` or `plugins/platforms/` calls the media policy on the path it is given (grep for
  `validate_media_delivery_path`, `_validated_delivery_path` and the two filters).
- **Cleanup.** The snapshot and its directory are removed when the send ends, whether it was delivered,
  refused, raised or cancelled (the `finally` of `upload`). If a send is cancelled while its snapshot is being
  made, the snapshot finishes in its thread and a done-callback removes it (`_discard_late_snapshot`). The fd
  is closed in `snapshot_attachment`'s `finally`. Cleanup never touches the source file or the ledger rows.

### Remaining limits (stated, not fixed)

- **In-place writes.** The snapshot holds the approved file's content as it is during the copy. A write into
  that same file (same inode) before or during the copy is what gets sent, and holding the fd does not stop
  it. Once the copy is made, changes to the source never reach the transport.
- **Hard links.** The policy is path-based, so a hard link to a protected file under an allowed path passes
  (unchanged).
- **Same-user processes.** The snapshot directory is private to the user (0700), not to the process. A
  same-user process that targets it between the copy and the upload is not covered.
- **Leftover snapshots.** A process killed during a send (SIGKILL, power loss) leaves its snapshot directory in
  the document cache. `cleanup_document_cache` removes only top-level files, so nothing sweeps it. The same
  happens if the event loop shuts down while a snapshot is being copied.
- **Disk.** Each send attempt of an owed attachment copies it once, so space is bounded by the platforms'
  upload sizes.
- **Replies that owe nothing** (no inbound record) are unchanged: they are checked once at extraction and
  uploaded by path. Only an owner message's attachments go through the snapshot.
- **Relative `MEDIA:` paths** (no `/`, `~/` or drive-letter anchor) are not declarations, because the tag
  grammar never accepted them.

### Tests (owner/peer file)

New: 5 functions, 16 cases.

| Test | Covers | At 1f70e1e4 |
|---|---|---|
| `test_a_declared_attachment_of_any_extension_is_owed_until_it_is_sent[normal / queued / recovery / inline_code × missing / blocked]` | P1-1: `report.log` missing or refused in the adapter's final, the queued first response, a later start, and the inline-code form. It is recorded missing or blocked and never uploaded, and the record outlives the delivered text. When the file is later there and allowed, the next start uploads it alone and cleans up | 8 fail: the record is deleted after the text |
| `test_each_owed_attachment_is_checked_right_before_its_own_send[normal / recovery]` | P1-2, deterministic: the first upload is held at the transport by an `asyncio.Event`, the second path is swapped for a symlink outside the allowlist, then the upload resumes. No sleep, no mock of the check. The second is blocked, its bytes are never read by the transport, and the record stays | 2 fail: the forbidden bytes are uploaded and counted delivered (`normal`: the record is already deleted; `recovery`: `FORBIDDEN` is among the uploads) |
| `test_an_owed_attachment_uploads_the_bytes_its_check_approved_and_leaves_no_copy[delivered / refused / raised / cancelled]` | the bytes are bound: the path is swapped for an outside symlink while the upload is held at the transport, where an adapter opens its path. The transport reads the approved bytes. The strict policy accepts the path it is handed (as photon checks it). The snapshot is gone and the source untouched however the send ends. The record is kept unless the reply was delivered | 4 fail: 3 upload `FORBIDDEN`; `cancelled` is handed the source path |
| `test_an_owed_attachment_whose_directory_is_swapped_as_it_is_opened_is_never_uploaded` | the window between the check and the open: the parent directory becomes a symlink outside the allowlist as the checked path is opened (an `os.open` hook). The check after the open blocks it | fails: no check at the open, the PDF is delivered and the record deleted |
| `test_a_send_cancelled_while_its_snapshot_is_made_leaves_no_copy` | the owner turn is cancelled while its snapshot is being made: the late snapshot is removed, and the source, record and obligation (`pending`) stay | fails: `snapshot_attachment` does not exist |

**Changed tests.** The transport is now given a snapshot path, not the source path. So `_attachment_sends`
and the recorder in `test_store_partial_delivery_interruption_restart_recovery_and_cleanup` now record a
document as its name and its bytes, which is what the transport reads (`_upload`). Five assertions changed
from the path to (name, bytes). For `symlink_to_allowed`, the name still shows that the link's target was
uploaded, not the link. No other test changed.

### Checks at 8c746b3ba3

| Check | Result |
|---|---|
| RED at 1f70e1e4 (the final test file on its sources) | 16 of 16 new cases fail, as in the table |
| The reviewer's six probes (`/private/tmp/rv-sa-review-probes.py`) | at 1f70e1e4: 6 failed. At head: 6 passed, but the 2 for P1-2 pass vacuously. Their swap runs only when `Path(file_path) == first`, and at head the transport is given a snapshot path, so the swap never runs (their output shows both PDFs uploaded with their own bytes). `test_each_owed_attachment_is_checked_right_before_its_own_send` is the same counterexample, keyed on the file's name |
| Owner/peer file | 159 passed (the 143 of 1f70e1e4 + 16 new) |
| Startup/flush/resume/ledger set (15 files) | 190 passed |
| Finalizer set | 107 passed, 2 skipped |
| Extra: media extraction seam tests (`test_tool_response_drop_recovery`, `test_media_resend_dedup`, `test_platform_base`, `test_remote_media_fetch`) | 130 passed, 1 skipped (the count 1f70e1e4's response reports) |
| Extra: the media files of earlier rounds (9 files) | 62 passed (62 at 1f70e1e4, run this round) |
| ruff | passes on `gateway/` and the test file |
| Mutations (on a copy of the tree) | 11 of 11 killed, below |
| commitlore context / guard | context was run on every edited file before it was edited. guard (with the proposal) is incomplete on every file because the notes mirror is not fetched |
| commitlore capture | refused: verification did not run (notes-unfetched). The prepared transaction was removed and the commit went through `commitlore commit --none` |

| Rule | Mutant | Killed by |
|---|---|---|
| 1 | M1a: the adapter's final keeps `extract_media`'s list only | `any_extension[normal-*, inline_code-*]` |
| 1 | M1b: the queued reply keeps `extract_media`'s list only | `any_extension[queued-*]` |
| 1 | M1c: recovery keeps `extract_media`'s list only | `any_extension[recovery-*]` |
| 1 | M1d: an unknown-extension tag is declared only once it validates | `any_extension[*]` (8) |
| 1 | M1e: an inline-code tag is declared only once it validates | `any_extension[inline_code-*]` |
| 2 | M2a: every attachment is checked (and copied) before the sends, as at 1f70e1e4 | `right_before_its_own_send[normal, recovery]` |
| 2 | M2b: the checked path is uploaded, not a snapshot (the adapter re-opens it) | `bytes_its_check_approved[*]` (4) |
| 2 | M2c: no check after the open | `swapped_as_it_is_opened` |
| 2 | M2d: the snapshot is kept after its send | `bytes_its_check_approved[*]` (4) |
| 2 | M2e: a snapshot made after its send was cancelled is kept | `while_its_snapshot_is_made` |
| 2 | M2f: the snapshot lies in the system temp directory | `bytes_its_check_approved[*]` (4): the strict policy refuses the path the transport is handed |

Not witnessed: `O_NOFOLLOW` alone. Without it, a final-component symlink swapped in at the open is still
caught by the check after the open, so no test can tell the two apart.

## Slice A (the s1d review of c9760626, findings 1 and 3): attachment obligations and send authority

Findings 2 (retry bounds) and 4 (retention) are slice B and are not touched: `claim_attachment_redelivery`,
`MAX_REACTIVATIONS`, `reactivate_expired` and `_prune_unlocked` are unchanged.

### The one check

`gateway/platforms/base.py`, `attachment_verdict(path, session_key)`. It checks a local attachment (a `MEDIA:`
file or a bare local file) against the media delivery policy of now and the file the path names now, with
symlinks resolved (`_validated_delivery_path`, the validator the live final already used). It returns one of:
- `sendable`, with the host path to upload;
- `missing`: no file is there;
- `blocked`: a file is there and the policy refuses it.

It runs at two points, on every path:
- **When the components are stored.** `_reply_media` (and `_reply_local_files` for bare paths) calls it from:
  - the adapter's final (`_extract_response_content`);
  - the queued first response and the streamed final (`owe_explicit_reply`);
  - a later start reading a stored answer (`extract_reply_parts`).
- **Right before each send.** `_deliver_owed_attachments` is the one sender of an owner message's attachments
  (the adapter's final, `_OwedReply.finish`, `redeliver_owed_attachments`). It checks every attachment it is
  about to send again, inside the routed profile's media scope. It uploads the path that check resolved, not
  the stored path. `redeliver_owed_attachments` no longer does its own `os.path.isfile` check.

### By rule

1. **Every declared attachment is an obligation before any filtering** (s1d P1-1, base.py:4893 and :4435).
   An owner message's reply keeps every `MEDIA:` file it declares as a component, in the live final, the
   queued lane and recovery alike. A sendable file is stored under its resolved path; any other file under
   its declared path. The send records one that is gone as `missing`: never sent, never completion or
   deletion evidence. Other replies, which have no obligation, still drop an unsendable file, as before.
2. **One check** for the normal, queued and recovery paths: `attachment_verdict` (above).
3. **Checked again right before each send.** A policy change, a changed file or a swapped symlink between
   storage and send is caught at the send. The upload is the file that check resolved.
4. **A forbidden attachment is never uploaded.** Its component is recorded `blocked`
   (`delivery_ledger.COMPONENT_BLOCKED`). `inbound_completion` reads `BLOCKED`, which keeps the record and
   the marker the way `MISSING` does. The chat is told once (on the change of state), an ERROR is logged at
   each check, and the component is never dropped from the obligation.

**Decision taken (the packet does not say).** `blocked` blocks completion, the same as `missing`. A message
whose attachment the policy refuses keeps its record. A later start that finds the policy now allows the
file sends it. It is tried within the same attachment-try bound as `missing`, which belongs to slice B. The
other reading would let a blocked component count as finished, so the record goes once the text is
delivered. That means a reply could complete with an undelivered component, and the rules here never allow
that.

**Bare local paths.** A bare path in the text is detected, not declared: it is an attachment only when its
file exists and the policy accepts it, as before (now through the same check). Once stored, it is checked
again before each send like any other attachment.

### Tests (owner/peer file)

New: 3 functions, 12 cases.

| Test | Covers | At c9760626 |
|---|---|---|
| `test_a_live_reply_whose_attachment_is_gone_keeps_it_owed_as_missing[normal / queued]` | P1-1 (live, queued), rule 1 | fails: the record is deleted after the text |
| `test_an_attachment_the_current_media_policy_refuses_is_kept_blocked_and_never_uploaded[normal_refused / queued_refused]` | rule 4: refused when the reply goes out | fails: the PDF is dropped and the record is deleted |
| `...[normal_restored / queued_restored]` | P1-3 (live, queued): restored outside the allowlist after it was recorded missing | fails: the record is deleted after the text |
| `...[stored_restored]` | P1-3, the reviewer's reproduction | fails: the restart uploads the PDF and deletes the record |
| `test_an_owed_attachment_is_checked_again_right_before_each_send[allowed]` | a normal allowed attachment (first upload refused by the transport, then delivered at the next start) | passes (guard) |
| `...[file_changed]` | the file under its path is replaced by an older one that the strict policy's recent-file trust no longer covers | fails: uploaded, record deleted |
| `...[policy_changed]` | the policy turns strict between the two starts | fails: uploaded, record deleted |
| `...[symlink_swapped]` | the path becomes a symlink to a file outside the allowlist | fails: the link is uploaded, record deleted |
| `...[symlink_to_allowed]` | the path becomes a symlink to a file inside the allowlist: the check's target is uploaded, never the link | fails: the link path is uploaded |

No existing test changed.

### Checks at 1f70e1e476

| Check | Result |
|---|---|
| RED at c9760626 (the new test file on its sources) | 11 of 12 fail, as in the table; `[allowed]` passes there |
| The reviewer's six probes (`/private/tmp/rv-s1d-review-probes.py`, on a copy of this tree) | the 3 for findings 1 (normal, queued) and 3 pass. The 3 for findings 2 and 4 (slice B) still fail |
| Owner/peer file | 143 passed (131 at c9760626 + 12 new) |
| Startup/flush/resume/ledger set (15 files) | 190 passed |
| Finalizer set | 107 passed, 2 skipped |
| Extra: the media extraction seam tests (`test_tool_response_drop_recovery`, `test_media_resend_dedup`, `test_platform_base`, `test_remote_media_fetch`) | 130 passed, 1 skipped, the same as c9760626 |
| Extra: the media files of earlier rounds (9 files: media-only outcome, mixed routing, post-stream media, TTS routing, stream final contract, session recovery, ...) | 62 passed, the same as c9760626 |
| ruff | passes on `gateway/` and the test file |
| Mutations (on a copy of the tree) | 9 of 9 killed, below |
| commitlore context / guard | context was run on every edited file before it was edited. guard (with the proposal) is incomplete on every file: the notes mirror is not fetched |
| commitlore capture | refused (notes-unfetched; its Ruled-out also failed the vocabulary check), so the commit went through `commitlore commit --none` |

| Rule | Mutant | Killed by |
|---|---|---|
| 1 | MA1a: the live final drops an unsendable attachment | `gone[normal]`, `refuses[normal_*]` |
| 1 | MA1b: the queued reply drops an unsendable attachment | `gone[queued]`, `refuses[queued_*]` |
| 1 | MA1c: a missing attachment is recorded delivered | `gone[*]`, `refuses[*_restored]` |
| 2 | MA2: the one check ignores the policy (any existing file is sendable) | `refuses[*]` (normal, queued, stored), `checked_again[file_changed / policy_changed / symlink_swapped]` |
| 3 | MA3a: the send checks existence only (the c9760626 recovery check) | `refuses[*]`, `checked_again[file_changed / policy_changed / symlink_*]` |
| 3 | MA3b: the stored path is uploaded, not the checked one | `checked_again[symlink_to_allowed]` |
| 4 | MA4a: a refused file is recorded missing (no explicit blocked state) | `refuses[*]`, `checked_again[file_changed / policy_changed / symlink_swapped]` |
| 4 | MA4b: a blocked attachment completes the reply | the same 8 cases |
| 4 | MA4c: a refused attachment is dropped when the components are stored | `refuses[normal_refused / queued_refused]` |

### Open items

- **Slice B.** Findings 2 (an attachment or text whose tries are spent is never tried again) and 4 (retention
  prunes delivered rows while the inbound record remains) are untouched. A blocked attachment shares the
  attachment-try bound with a missing one, so after `MAX_REACTIVATIONS` starts it is not checked again
  either. Its record then stays, with an ERROR at each start.
- **Check-to-upload window.** The upload uses the path the check resolved a moment earlier. A swap of that
  resolved file itself in the instant between the check and the adapter opening it is not covered. Closing
  it would need an upload from an open file descriptor, which the adapters' `send_document`/`send_voice`
  signatures do not take.
- **Hard links.** The policy is path-based. A hard link to a protected file, placed under an allowed
  path, passes the check (unchanged from before).
- **Untracked replies.** A reply that no owner message is owed (no inbound record) still checks its
  attachments once, when it is extracted, and drops an unsendable one, as before. Nothing of it is stored,
  so nothing waits for a restart.
- **Docker.** The check before a send gets the stored host path. With a Docker terminal backend active,
  `_translate_docker_container_media_path` finds no container prefix for it and logs its "did not resolve"
  WARNING before the host path is validated as is.
- Still open from earlier rounds: not exactly-once, no in-process retry for attachments, album grouping,
  unledgered finals, and session recovery not copying `delivery_components`.

## Follow-up at c9760626b4 (CTO, on 7289dc6b's open items)

- **A notice after a stored answer.** When a run produced its answer and then raised (a post-turn hook,
  with no restart), its error notice used to terminate the message. Now the turn records that the run
  answered (`run_turn`: `_inbound_answer_stored`, set right after `_run_agent` returns a completed answer;
  `run_inbound` carries it to the adapter). The notice's components are stored with
  `delivery_ledger.ANSWER_STORED`, and `inbound_completion` reads such a notice as superseded (`OPEN`),
  the same as a cut-short one. It is never completion evidence. The record and the marker stay, and the
  next start owes the stored answer under the answer's own obligation. The live process does not send
  that answer: it waits for the next start, as a cut-short run's does.
- **Attachment tries bounded.** An undelivered attachment (pending, refused or missing) is tried by at
  most `MAX_REACTIVATIONS` later starts. `delivery_ledger.claim_attachment_redelivery` counts each try
  in a new `delivery_components.redeliveries` column before making it. After the bound the attachment
  keeps its state (a missing one stays `missing` and still blocks completion), the record stays, and
  each start logs an ERROR ("not tried again") instead of trying it.
- **Tests.** Two existing tests are now parametrized:
  - `test_a_notice_a_restart_cut_short_never_stands_in_for_the_answer_the_run_stored[cut_short / answer_stored]`
  - `test_a_stored_answer_whose_attachment_is_gone_keeps_it_owed_as_missing[next_start / after_the_bound]`

  At 7289dc6b, `[answer_stored]` fails because the record is deleted, and `[after_the_bound]` fails
  because the PDF is sent and the record deleted. Both pass at c9760626.
- **Mutations.** M1c (an `ANSWER_STORED` notice completes the message) and M3b (attachment tries
  unbounded) are both killed. All 11 mutants are killed at c9760626.
- **Suites at c9760626.** Owner/peer file 131 passed. Startup/flush/resume/ledger set 190 passed.
  Finalizer set 107 passed, 2 skipped. ruff passes.

## Slice 1 review of 2aa38989: one completion decision

The five reproductions came from three places deciding completion three ways: the boundary read
`obligation_complete` (ignoring the cut-short flag), startup read `obligation_progress` (honouring it),
and the transcript step decided "answered" from stripped display text. Those are gone. Slice 2 (steer)
is not touched. `run_inbound._settle_inbound_records` / `_inbound_release_proven` is still the only code
that deletes a record.

### The decision

`gateway/delivery_ledger.py`, `inbound_completion(obligation)`. It takes the `InboundObligation`
(input, run, destination) and returns `(state, reply_id)`, from three inputs read in one transaction:
- **What the reply is** (`REPLY` on each component): `answer`, or `notice` (a reply the run sent instead
  of an answer). Each has its own obligation id (`inbound_obligation_id(..., kind=)`; the answer id is
  unchanged). The answer's components decide when any are stored; otherwise the notice's.
- **Each component's state**: `delivered`, `pending`, `failed`, `missing`.
- **The text's retry state**: the row's state, attempts and age (`_retry_spent`, the same predicate the
  sweeps now use to abandon).

States: `OPEN` (the reply of a run a restart cut short: a superseded notice, or a provisional answer;
never completion), `COMPLETE`, `EXPIRED` (undelivered text whose budget is spent), `MISSING`, `OWED`,
`UNTRACKED` (a legacy row with no components), or nothing. The boundary releases only on `COMPLETE`.
Startup recovery reads the same call first, and only reads the transcript on nothing or `OPEN`.

The relation between a notice and the input is explicit and kept:
- A notice of a run that **ended** terminates the input. Delivered, the input is `COMPLETE` (unchanged
  from s1c: an ended run's error notice is not run again).
- A notice of a run a restart **cut short** (`CUT_SHORT`) is superseded. The next start owes the
  transcript's completed answer under the answer obligation, or runs the message again under the next
  run. The notice's row and components stay as history.
- An answer a cut-short run sent is provisional. When the next start finds that run's completed answer
  in the transcript, `confirm_answer` clears the flag, and the same one decision then completes it.

### What changed, by rule

1. **Notice vs answer** (P1-1). The turn tags its reply (`run_turn._inbound_reply_kind`): a notice from
   the error reply, or when the agent result is failed, interrupted or partial; otherwise the answer.
   `run_inbound` carries the tag to the adapter's event. The adapter stores the components and writes
   the text row under the reply's own kind (`_record_reply_components(kind=)`,
   `_record_delivery_obligation`). The in-band first response and the streamed final are always the
   answer.
2. **Store failure** (P1-2). `_record_reply_components` raises `delivery_ledger.ReplyNotStored` instead
   of returning `(None, [])`. `(None, [])` now means only "nothing tracks this reply" (not an owner
   message's, or the ledger is off). On `ReplyNotStored` the adapter final sends nothing: no voice, no
   text, no attachment. The reply counts as owed, so the record and the marker stay, and the next start
   owes it from its transcript. `owe_explicit_reply` returns a withheld `_OwedReply`. Its `finish` sends
   nothing, and the queued lane does not send the first response.
3. **Missing attachment** (P1-3). `extract_reply_parts` keeps a `MEDIA:` file that is gone from the host
   as one of the reply's attachments. Only a path the delivery policy refuses is dropped, as the live
   final drops it. The send then records it `missing`. The decision says `MISSING`, which blocks
   completion and deletion. Once the file is back, a later start sends it.
4. **MEDIA-only answer** (P1-4). The transcript step reads the answer as stored
   (`_crash_left_reply(keep_media=True)`) and takes its components from that
   (`_stored_answer_parts(content, ...)`). A `MEDIA:`-only answer is owed as an attachment, never
   called unanswered, and never run again. The generic crash-left marker path keeps its stripped text.
5. **Retry exhausted or expired** (P1-5). `EXPIRED` leads `_settle_stored_inbound` to
   `_reactivate_expired_reply`, then `delivery_ledger.reactivate_expired`. That claims the row for exactly
   one send by this process (`attempting`, attempts + 1, the duplicate marker) and bumps a new
   `reactivations` column. The send goes through `_redeliver_claimed_obligations`. It refuses, with the
   record and the text kept and an ERROR logged, in these cases: a managed /acp row; a row a live process
   holds; a dead target (`classify_dead_error`: send-forbidden); a row not expired; a row reactivated
   `MAX_REACTIVATIONS` (3) times already. Retention no longer prunes a row whose components are still
   owed (`_NOT_OWED`). Without that, a prune followed by re-creation would hand the text a fresh budget
   every 7 days, with no bound.

Removed: `obligation_complete`, `obligation_progress`.

### Tests (owner/peer file)

New: 6 functions, 9 cases: the review's five reproductions, then one end-to-end acceptance.

| Test | Rule | At 2aa38989 |
|---|---|---|
| `test_a_notice_a_restart_cut_short_never_stands_in_for_the_answer_the_run_stored` | 1 | fails: the notice's components sit under the answer's obligation |
| `test_a_reply_whose_components_cannot_be_stored_sends_nothing_until_they_are` | 2 | fails: the image goes out untracked |
| `test_a_stored_answer_whose_attachment_is_gone_keeps_it_owed_as_missing` | 3 | fails: the record is deleted |
| `test_a_stored_answer_that_is_only_an_attachment_is_delivered_and_never_run_again` | 4 | fails: the message runs again |
| `test_a_reply_whose_retry_budget_is_spent_is_reactivated_a_bounded_number_of_times[stale / attempts]` | 5 | fails: nothing is sent and the record stays |
| `...[send_forbidden]` | 5 (never revived) | fails: base sends it at the first start |
| `...[bound_spent]` | 5 (bound) | fails on the new `reactivations` column only |
| `test_store_partial_delivery_interruption_restart_recovery_and_cleanup` | acceptance | passes (s1c already recovered this cycle) |

The acceptance test runs in order: components stored before the first send, then the text delivered,
then the process killed with the PDF in flight (record and marker kept, PDF `pending`). The restart
then sends the PDF alone (no text, no rerun), every component reads back delivered, and the record and
the marker go.

Changed: `test_a_rerun_is_fixed_before_its_hooks_and_a_failed_lookup_or_read_back_never_runs_it`. The
`ledger_lookup` and `read_back` failures are now injected into `inbound_completion`, since lookup and
read-back are the same call. `read_back` arms only once the rerun has run. In `agent_start`, the
rerun's error notice is read under its notice obligation. The helpers `_inbound_obligation` and
`_components` take a `kind`.

### Checks at 7289dc6b8c

| Check | Result |
|---|---|
| RED at 2aa38989 (new tests on its sources) | 8 of 9 fail: 7 on behaviour, `bound_spent` on the new column. The acceptance test passes there |
| The reviewer's six probes (`/private/tmp/rv-s1c-review-probes.py`, on a copy of this tree) | 5 passed, 1 failed. P1-1 fails on `held[0][CUT_SHORT]` (`IndexError`) because it reads the notice under the answer's obligation, the s1c layout rule 1 removes. With that one line reading the notice obligation (`_inbound_obligation(..., 'notice')`), all 6 pass |
| Owner/peer file | 129 passed (120 at 2aa38989 + 9 new) |
| Startup/flush/resume/ledger files (15) | 190 passed |
| Finalizer and queued-ledger files | 107 passed, 2 skipped |
| Extra: `test_acp_managed_ingress.py`, `test_state_fence_refusal.py` (both read the ledger internals) | 28 failed, 370 passed. The same 28 fail at 2aa38989 |
| 68-file set | not run (the packet excludes it) |
| Mutations (on a copy of the tree) | 9 of 9 killed, listed below |
| ruff | passes on `gateway/` and the test file |
| commitlore context / guard | context was run on every edited file before editing it. guard (with a proposal per file) is incomplete on every file: the notes mirror is not fetched |
| commitlore capture | refused (notes-unfetched), so the commit went through `commitlore commit --none`. The guard advisory named a start-time compare in the settle fence; this change adds none |

| Rule | Mutant | Killed by |
|---|---|---|
| 1 | M1a: a notice takes the answer's obligation | the notice test |
| 1 | M1b: a cut-short notice completes the message | the notice test |
| 2 | M2: a failed store returns `(None, [])` and sends | the store test |
| 3 | M3: recovery drops a gone `MEDIA:` file | the missing test |
| 4 | M4: answered-ness from the stripped text | the media-only test |
| 5 | M5a: no reactivation | `stale`, `attempts` |
| 5 | M5b: no bound | `bound_spent` |
| 5 | M5c: a send-forbidden row is revived | `send_forbidden` |
| 5 | M5d: retention prunes a row whose components are owed | `bound_spent` |

### Documented uncertainty and open items

- (Closed at c9760626.) An ended run's error notice no longer terminates its message when the run had
  already stored its answer.
- **The reply kind comes from the turn** (`_inbound_reply_kind`): the error reply, or a failed,
  interrupted or partial agent result. The in-band first response and the streamed final are always the
  answer. A streamed reply is stored before its run's end state is known, so it is never marked cut short.
- **Bound reached.** After `MAX_REACTIVATIONS`, an expired text stays undelivered with its record and its
  row, and an ERROR is logged at each start until someone handles it by hand. The same happens to a
  send-forbidden row.
- (Closed at c9760626.) Attachment tries are bounded by `MAX_REACTIVATIONS` starts.
- **Withheld replies.** When the components cannot be stored, the reply waits for the next start. The
  queued lane's first response is not sent even when the stream had already shown its text, so the next
  start sends that text again with the duplicate marker.
- Still open from earlier rounds: not exactly-once, no in-process retry for attachments, album grouping,
  unledgered finals, and session recovery not copying `delivery_components`.

## Slice 1 review of b9a08796: what a reply requires, and the ledger before the transcript

The four reproductions came from two causes, both in scope here. Slice 2 (steer) is not touched.
The settlement boundary is unchanged in place: `run_inbound._settle_inbound_records` /
`_inbound_release_proven` is still the only code that deletes a record, and only its proof check changed.

### A. Component obligations, stored before the first send

**Ledger change** (`gateway/delivery_ledger.py`): a new table `delivery_components` in `state.db`, keyed
`(obligation_id, component_id)`, with `kind`, `payload` (JSON) and `state`. It is created by
`_initialize_schema`, next to `delivery_obligations`. `delivery_obligations` is not altered, and no
existing row is rewritten.
- `record_components(obligation_id, components)` stores a reply's components before any of them is
  sent. Component ids are stable within the reply: `text`, `image:<n>`, `media:<n>`, `file:<n>`. The text
  component carries its text, and each attachment carries what its send needs (URL and alt, or path,
  voice and as-document). It is insert-only: the first stored set is what the obligation requires.
- The text component follows its `delivery_obligations` row. `_update_state` (that is, `mark_delivered`)
  and `record_delivered` mark it delivered in the same transaction. This also covers a row that is later
  pruned.
- `mark_components` records each attachment's outcome. A delivered component never goes back.
- `obligation_complete` is the check the boundary uses. It is true only when components are stored and
  every one reads back `delivered`. No stored components means not complete.
- Complete component sets are pruned together after the 7-day retention.
- `InboundObligation.attachments_ok` is removed. `sent` (default False) is still the proof with the
  ledger off, and now covers every component.

**Paths.** All of them store first and settle through `obligation_complete`:
- Adapter final (`platforms/base.py`): `_record_reply_components` runs after extraction and before TTS,
  text or any attachment. An attachment-only reply gets components too. Attachments go through
  `_deliver_owed_attachments`, which records each outcome (in a `finally`, shielded) and never resends a
  component the stored set shows delivered.
- Images: a batch reports success when any image went out. So a tracked image goes as its own batch, and
  only its own result proves it delivered.
- Queued first response and streamed final (`run_turn.py`): `adapter.owe_explicit_reply` stores the text
  and the explicit `MEDIA:` files. `_OwedReply.finish` records the text delivered (stream, edit or ledgered
  send), then uploads what is still owed. A refused text sends nothing more. These paths no longer go
  through `_deliver_media_from_response` for an owner message, so its uploads have outcomes.
  `_record_streamed_inbound_answer` is gone.
- Restart (`run_startup.py`): for an incomplete set, a text component whose row is missing is owed
  again from its stored text (as an `attempting` crash-left row, so it carries the marker). The sweep
  sends it. `adapter.redeliver_owed_attachments` sends only components not yet delivered. A local source
  that is gone is never sent or counted: the component is recorded `missing` and logged at ERROR, the chat
  is told once (on the change to `missing`), and the record stays.
- A row without components (written before this change) is never complete. Its components are read from
  that run's transcript answer and stored next to the row (the text state copied from the row). With no
  such answer, the record is kept, an ERROR is logged, and the message is not run again. The row is not
  touched.

### B. The obligation is read before the transcript

`_owe_stored_answer` first reads the obligation of (record, the run its record admitted, destination)
through `obligation_progress`:
- `complete` means cleanup only (`_settle_stored_inbound` sends nothing).
- `incomplete` or `untracked` goes to `_resume_stored_obligation`.
- A failed read raises, so the result is `unknown`: nothing is run, the run counter is not bumped, and
  the marker is held.

Only when the ledger holds nothing for that run is the transcript read. The transcript may name another
obligation (a message folded into another), which is checked the same way. Otherwise the answer is owed:
its components are extracted from the stored content with `BasePlatformAdapter.extract_reply_parts`, so
`MEDIA:` files become attachments and are no longer stripped as text.

Run end state: a turn whose records stay because a restart cut it short (`_settle_inbound_record` returns
kept) marks its adapter event `_inbound_cut_short`. The reply it still sends, an interruption notice, is
stored with `run_cut_short` on its components. `obligation_progress` reports `cut_short`, and recovery
lets the transcript decide. No completed answer means the message runs again. A completed answer means
the same obligation and cleanup only. An error notice of a run that ended (P1-4) is not cut short: it
completes its obligation.

### Tests (owner/peer file)

New: 7 functions, 12 cases.

| Test | Covers | At b9a08796 |
|---|---|---|
| `test_a_reply_stores_what_it_requires_..._delivered[text_only / attachment_only / text_and_attachment]` | normal paths; P1-3 | attachment_only: the record stays and a restart presents it again. The other two fail only on the new store |
| `test_a_partly_delivered_reply_..._still_owes[image / document]` | P1-1; partial then restart | the refused attachment is never sent again |
| `test_a_queued_first_reply_whose_attachment_is_refused_...` | P1-2 | the records are lost |
| `test_a_delivered_reply_whose_record_could_not_be_deleted_...[error_notice / answer_with_attachment]` | P1-4; delivered then a failed delete | error_notice: the message runs again |
| `test_a_ledger_row_without_components_is_never_taken_as_a_complete_reply[answer_with_attachment / no_answer]` | compatibility | the image is never sent / the message runs again |
| `test_an_owed_attachment_whose_source_is_gone_is_never_counted_delivered` | a gone source | the record is deleted |
| `test_one_refused_image_of_two_is_owed_alone_...` | a partial batch | the records are lost |

Changed, because B or A changes what is right:
- `test_a_rerun_after_a_restart_is_owed_under_the_run_its_record_fixed` (was `..._never_takes_over_...`):
  under B, an earlier run's row means no rerun. The rerun now starts from an empty ledger. "The row is
  never replaced" moved to the compatibility test and to the insert-only test.
- `test_a_rerun_is_fixed_before_its_hooks_and_a_failed_lookup_or_read_back_never_runs_it` (was
  `..._start_or_lookups_fail_...`): no earlier row. It gains `ledger_lookup` (B3: kept, not run) and
  `read_back` (kept, then cleanup only).
- `test_a_reply_with_an_attachment_..._every_component_was_delivered`: the restart now sends only the
  missing component (it used to resend nothing for the image).
- `test_a_transient_failure_...[transcript_unreadable]`: the failing ledger write now fails the component
  store too, so the case is still "nothing in the ledger".
- `test_a_first_response_neither_ledgered_nor_delivered_...[raises]`: under B, the first message's
  stored answer is delivered instead of run again, and the follow-up's delivered error notice completes
  its obligation.

### Checks

| Check | Result |
|---|---|
| RED at b9a08796 (new and changed tests on its sources) | All 12 new cases fail. Nine fail on behaviour: `attachment_only` (the record stays and a restart presents it again), `partly[image]` and `partly[document]` (the attachment is never resent), queued (records lost), `error_notice` (runs again), legacy (the image is never sent / runs again), gone source (deleted), two images (records lost). Three fail only on the new store: `text_only`, `text_and_attachment`, `answer_with_attachment`. Of the changed tests, `chain[raises]`, `rerun_is_fixed[ledger_lookup]` and both `every_component` cases fail on behaviour; `rerun_is_fixed[read_back]` fails because its injection point does not exist there; the rest pass there |
| The reviewer's four probes (`/private/tmp/rv-s1b-review-probes.py`, on a copy of this tree) | 4 passed |
| Owner/peer file | 120 passed (107 at b9a08796, + 12 new cases, + 1 new case of a changed test) |
| Startup/flush/resume/ledger files (15) | 190 passed |
| Finalizer and queued-ledger files | 107 passed, 2 skipped |
| Extra: ledger, media, state and session-recovery files (34) | 328 passed, 1 skipped (the same as b9a08796) |
| Extra: `test_acp_managed_ingress.py`, `test_state_fence_refusal.py` (both read the ledger internals) | 28 failed, 370 passed; the same 28 `TestReceiptWireAgainstAcp` cases fail at b9a08796 |
| 68-file set | not run (the packet excludes it) |
| Mutations (on a copy of the tree) | 14 of 14 killed, listed below |
| ruff | passes on `gateway/` and the test file |
| commitlore context / guard | context was run on every edited file before editing it. guard is incomplete on every file (notes mirror not fetched) |
| commitlore capture | refused (notes-unfetched), so committed with nothing recorded through `commitlore commit --none`. The guard advisory in the capture named two records (a start-time compare in the settle fence; handing a managed marker to a follow-up). This change does neither |

| Rule | Mutant | Killed by |
|---|---|---|
| A1 | MA1a: an attachment-only reply stores nothing | `stores_what_it_requires[attachment_only]` |
| A1 | MA1b: components are stored after the sends | `stores_what_it_requires[text_and_attachment]` |
| A2 | MA2a: any delivered component completes | `partly` |
| A2 | MA2b: the boundary checks the text row (the b9a08796 check) | `partly`, `attachment_only` |
| A2 | MA2c: the queued lane does not track components | queued |
| A3 | MA3a: no stored components counts as complete | legacy `no_answer` |
| A3 | MA3b: a batch's success proves every image | two images |
| A4 | MA4a: delivered components are sent again | `partly` |
| A4 | MA4b: a gone source counts as delivered | gone source |
| A5 | MA5: a row without components is taken as text only | legacy `answer_with_attachment` |
| B1 | MB1: the transcript is read before the ledger | `error_notice` |
| B2 | MB2: a complete obligation runs again | `delivered_reply_whose` |
| B3 | MB3: a failed lookup is taken as nothing in the ledger | `ledger_lookup` |
| B2 (run end) | MB4: a cut-short run's notice settles the message | `restart_drain_interrupts` |

### Documented uncertainty and open items

- **Not exactly-once.** A crash after a send and before its outcome is written leaves that component
  undelivered, and the next start sends it again: the text with the duplicate marker, an attachment
  without one. Attachment outcomes are written when the attachment sends end (in a `finally`), not
  after each send.
- **No in-process retry for attachments.** A refused attachment stays owed until the next start. Only
  the text has the ledger's runtime redelivery.
- **Album grouping.** A tracked reply with several images sends them one per batch.
- **Unledgered finals.** An ephemeral reply, or the reply to a `/`-prefixed message, stores no
  components, as its text gets no row. It is not proven, and its record stays. Unchanged.
- **Session recovery.** `hermes_cli/session_recovery.py` copies `delivery_obligations` but does not know
  `delivery_components`. A salvaged store keeps the rows without components, which recovery treats as
  untracked (never complete). Registering the table is outside this packet's files.
- **A row without components and no transcript answer** stays, with an ERROR at each start, until it is
  handled by hand. That is the cost of never presuming such a row complete.

## Slice 1 review of ace7c2f5: one settlement boundary

The four reproductions came from per-path exceptions. Those are removed: the owed set (`_inbound_records_owed`,
`_keep_inbound_records`, `_consume_inbound_record_id`), the transcript keep-set (`_inbound_answers_stored`, the
`answered=` argument), the in-turn `_inbound_answer_ledgered`, the ledger run lookup (`_inbound_run_for`,
`obligation_exists`), and the release at ledger time in `send_final_ledgered` for a turn holding records.
Finding 3 (steer) is slice 2 and is not touched.

### The boundary

`gateway/run_inbound.py`, `GatewayInboundMixin._settle_inbound_records`. The decision is `_inbound_release_proven`.
`_settle_inbound` runs it off the loop and then clears the turn's marker. It is the only caller of
`shutdown_flush.discard_inbound_record`. For an owner message's turn it is also the only path that clears the
active-turn marker. It releases only on one of these:
- A `delivery_ledger.InboundObligation` (record, run, destination) whose row reads back `delivered`
  (`obligation_state`), and every required attachment of the reply was delivered (`attachments_ok`). With the
  ledger off, nothing can be read back, so the proof is this process's own sends of every component (`sent`).
- An explicit end that owes nothing:
  - `INBOUND_NO_REPLY`, set by the turn (`run_turn._inbound_owes_nothing`): a muted wake, intentional silence,
    a result a newer run superseded, a refusal before the run. The adapter sets it too, for a reply it suppresses
    because a queued message interrupted it.
  - `INBOUND_DROPPED`: the message was refused, was a command, or was dropped by /stop, /new or /reset.

Anything else keeps the record and the marker:
- no proof, or a row in any state other than `delivered`;
- a failed ledger read (the boundary returns None);
- a delete that failed (None);
- an exception, a cancellation, a turn a restart cut short, or a turn no adapter delivers.

How each path reaches the boundary:
- **Adapter final** (`platforms/base.py`, `_release_turn_marker` with `_final_reply_proof`): the turn hands over
  its records in groups. Each group is paired with the answer it is owed: the turn's row, and each in-band
  follow-up (the follow-up groups come back up the chain in the result, `inbound_settle_groups`). The group the
  final answers gets the final's proof; every other group gets its own obligation. The marker goes only when
  every group is released.
- **In-band first response** (`run_turn._run_agent_queued_followup`): that level's records go when the first
  response's obligation reads back delivered. Otherwise the turn's end tries them again.
- **Startup** (`run_startup._settle_stored_inbound`): the transcript answer is owed through the ledger
  (insert-only), the ledger is redelivered, and the obligation is read back before anything is deleted. A failed
  transcript read raises (`_inbound_record_answer` no longer returns None for it). The record then stays and is
  not run again. A record with no completed answer runs again under a newly fixed run.
- **Startup marker recovery** (`_inbound_turn_markers`): a crash-left marker whose turn a stored record answers
  for is held. It is neither ledgered as a crash-left reply nor resumed. The boundary clears it when that record
  settles.

### Run identity

- The run is a durable counter in the record (`run`). Before a rerun is admitted, `_allocate_inbound_run` writes
  run + 1 (`shutdown_flush.set_inbound_record_run`). A failed write admits nothing and keeps the record. There is
  no ledger lookup and no fallback to run 0.
- The turn's obligation (record and run) is fixed at the claim in run_inbound. That is before agent:start, before
  execution, and before any error reply. The assignment after agent:start is gone.
- Inbound obligations are insert-only (`record_obligation(replace=False)`). Another reply under the same id raises
  `ObligationTaken` and is sent unledgered. The earlier answer's row is never replaced.
- A queued chain that raises after it sent a first response: its error notice answers the follow-up it failed
  on, never the first message.

### Other changes the boundary needs

- **A streamed final.** When the stream consumer confirms the final, it goes into the ledger as delivered
  (`delivery_ledger.record_delivered`), with the outcome "streamed". A voice caption that carried the text is
  recorded the same way. Without this, every streamed reply's record would stay until the next start, which
  would deliver it again.
- **Records stored at the claim.** run_inbound now stores a record there only for a turn an adapter delivers
  (`_turn_marker_handoff`). A nested /retry or a direct `_handle_message` caller has no delivery that could prove
  its reply.
- **An unclaimed message.** When the inbound path raises or is cancelled before a claim, `_settle_unclaimed_inbound`
  now keeps the record. Before, it deleted it unless the gateway was draining.

### Tests (owner/peer file)

The new tests are 6 functions, 11 cases. All 11 fail at ace7c2f5 (run on its sources with the new test file).

| Test | Covers | Fails at ace7c2f5 because |
|---|---|---|
| `..._keeps_its_record_and_marker_until_its_reply_reads_back_delivered` | the normal completion path | no record is on disk during the send |
| `..._cancelled_turn_whose_transcript_cannot_be_read_...` | P1-1 | the record is lost |
| `..._rerun_whose_start_or_lookups_fail_...[agent_start]` | P1-2 | the earlier row is replaced by the notice |
| `...[ledger_read]` | P1-3 | the earlier row is replaced by the rerun's answer |
| `...[run_write]` | invariant 3 | the rerun is admitted without a durable run |
| `..._attachment_..._every_component_...[text_refused_image_sent]` | P1-4 | the records are lost |
| `...[text_sent_image_refused]` | all components | the records are released |
| `..._transient_failure_..._cleaned_up_after_a_restart[send_refused]` | the recovery-to-cleanup cycle | the records are gone |
| `...[delivered_unrecorded]` | the recovery-to-cleanup cycle | the records are gone |
| `...[transcript_unreadable]` | the recovery-to-cleanup cycle | the record is deleted at the failing start |
| `..._inbound_obligation_is_never_taken_over_...` | insert-only | `replace` does not exist |

The `ledger_read` case fails each process's first ledger read. At ace7c2f5 that read is the run lookup; now it
is the read-back after the rerun's delivery, which keeps the record. The next start closes it.

Changed tests:
- `test_a_restart_during_the_final_send_is_settled_by_redelivery_alone`: the record is on disk mid-send.
- `test_a_restart_after_the_adapter_accepted_a_message_before_its_claim_keeps_it[drained_follow_up]`: during the
  first turn's send, the first message's record is still on disk.
- `_crash_now` now patches the boundary.

### Checks at b9a0879622

| Check | Result |
|---|---|
| RED at ace7c2f5 | 11 of 11 new cases fail, with the outcomes in the table above |
| Owner/peer file | 107 passed (96 + 11 new) |
| Startup/flush/resume/ledger files (15) | 190 passed |
| Finalizer and queued-ledger files | 107 passed, 2 skipped |
| 68-file set | not run (final head only, per the packet) |
| Mutations (on a copy of the tree) | 11 of 11 killed, listed below |
| ruff | passes on `gateway/` and the test file |
| commitlore context / guard | context was run on every edited file before editing it. guard is incomplete on every file (notes mirror not fetched) |
| commitlore capture | refused (notes-unfetched); committed with nothing recorded |

Mutations, all killed:

| Invariant | Mutant | Killed by |
|---|---|---|
| 1 | M1a: a failed ledger read counts as proof | `ledger_read` |
| 1 | M1b: an unreadable transcript counts as no answer | `transcript_unreadable` |
| 2 | M2a: any ledger row counts as delivered | `delivered_unrecorded` |
| 2 | M2b: an escaped turn deletes its records | the cancel test |
| 2 | M2c: attachments are ignored | `text_sent_image_refused` |
| 2 | M2d: any component's success releases | `text_refused_image_sent` |
| 2 | M2e: release at ledger time | the normal-path test |
| 3 | M3a: the obligation is set after agent:start | `agent_start` |
| 3 | M3b: a failed run write admits | `run_write` |
| 3 | M3c: inbound rows are replaced | the insert-only test |
| cycle | M4: owed, never read back | all three transient cases |

### Documented uncertainty and open items

- **Not exactly-once.** A crash after the external send, before `mark_delivered`, leaves the row `attempting`.
  The next start redelivers it once, with the duplicate marker (the `delivered_unrecorded` case).
- **Attachments across a restart.** The ledger holds the text only. In process, a failed attachment keeps the
  record and the marker. Across a restart only the text can be recovered, so a delivered text settles the record
  and the attachment is not redelivered.
- **Attachments outside the adapter final.** In-band first responses and streamed finals report no attachment
  outcome (`_deliver_media_from_response` returns nothing), so they are proven on the text alone.
- **Pruned rows.** Delivered rows are pruned after 7 days, or by the 500-row cap, delivered rows first. A record
  that outlives its delivered row, because its delete kept failing, is delivered once more at a later start.
- **A failed turn's notice.** If the notice is delivered and the process dies before the delete, the next start
  runs the message again: its transcript has no completed answer.
- **A message steered into a turn whose row names no record** (an internal turn) has no answer obligation. It
  stays until the next start, which runs it again. This is slice 2 territory.
- **Ledger off.** The proof is this process's own sends. A crash in between runs the message again at the next
  start, as before.

## Review r4 of 36ed681e, slice 1

Scope is findings 1, 2 and 4. Finding 3 (answered steers admitted again) is slice 2 and is not touched.

### 1. A delivery error discarded the only recoverable answer

**What changed:** the adapter keeps the turn marker and the inbound records on disk whenever the reply was neither ledgered nor delivered. It only drops its in-memory handle on them (`_release_turn_marker(event, owed=True)`; `cut_short` is gone). That covers three cases:
- the delivery raised or was cancelled after the handler returned (`reply_owed`, set when the handler returns a reply);
- every send failed with no ledger row, because the ledger write failed or the ledger is off;
- the handler raised or was cancelled after its turn had handed its records over.

A final that was ledgered still releases both at the ledger hand-off. A reply that was delivered, or a turn with nothing to send, releases both as before. The next start delivers the stored answer from the transcript, or runs the message again.

**Test:** `test_a_reply_neither_ledgered_nor_delivered_keeps_its_record_and_marker_for_the_next_process`, parametrized:
- `extraction_raises` is the reviewer's case;
- `ledger_write_and_send_fail`: `record_obligation` raises and the send is refused.

The process lives on, then the next process starts (crash recovery included). At 36ed681e both records are gone and nothing is delivered. Now the record and the marker remain, the answer goes out once, and no second turn runs.

### 2. A delivered reply was sent again when its text changed during delivery

**What changed:** `delivery_ledger.inbound_obligation_id(session_key, recovery_id, platform=, chat_id=, thread_id=)` keys the obligation of a reply to a message with a record on that record and the destination, never the text. It is used by:
- `_record_delivery_obligation`, for the adapter's final (`_inbound_ledger_ref`) and the queued lane's `inbound:<id>` ledger id;
- `_consume_if_answered`.

**Test:** the r3 test `test_a_delivered_answer_whose_record_could_not_be_deleted_is_not_sent_again` is parametrized `plain` / `runtime_footer`. The footer comes from the production hook `_hmwa_runtime_footer_line`. At 36ed681e the footer variant sends the answer a second time; now nothing is sent and no turn runs. `test_two_messages_answered_with_the_same_text_are_each_delivered` (r2) still guards against distinct records merging.

### 4. Stream recovery bypassed the completion mark

**What changed:** the finalizer's branch that fills an existing blank assistant row now marks it `turn_completed` for a completed turn, as the append branch marks its row. In production that row was already flushed, so the flush updates it in place (`agent/transcript_repair.py`), and that update wrote only `content`. It now also writes the message's display metadata (`COALESCE`, so a message without metadata keeps the stored value). `hermes_state_messages.py` passes the encoder.

**Test:** `test_a_stream_recovered_reply_the_finalizer_filled_into_a_blank_row_is_answered_once`:
- the incremental flush writes the user row and a blank assistant row;
- the real `finalize_turn` stream-recovers the reply into that row;
- the production row builder plus `append_messages_batch` updates it in place;
- the record and the marker are kept, and the next process recovers from a crash.

At 36ed681e the marker delivers the answer, the message runs again, and a second answer goes out. Now one answer goes out and no turn runs. A finalizer-only fix still fails, because the mark never reaches the row.

### Checks at f89a57ae

| Check | Result |
|---|---|
| RED at 36ed681e (new tests against the 36ed681e sources) | 4 failed: F1 extraction (records 0), F1 ledger+send (records 0), F2 footer (a second delivery), F4 (the input reran plus a second delivery). The `plain` variant passes, as it should |
| Owner/peer file | 89 passed (85 + 4 new) |
| Startup/flush/resume/ledger files (the 15 files) | 190 passed |
| Finalizer and queued-ledger files | 107 passed, 2 skipped |
| Repair/ledger files (new this round: transcript_repair, the obligation id) | 7 failed, 111 passed. The 7 are the same 7 that fail at 36ed681e (`_WorkItem` TypeError in `test_tool_call_incremental_persistence.py`) |
| 68-file regression set | 32 failed, 1121 passed, 5 skipped. These are the 31 baseline failures plus `test_gateway_silence_tokens.py::test_queued_human_turn_also_gets_the_visible_fallback`, which also fails at 36ed681e: r3's `gateway/run_turn.py:4030` reads `turn_ctx.persist_user_display_metadata`, which the test's SimpleNamespace lacks. It is not caused by this slice |
| Mutation witnesses | 6 of 6 killed: M1a (release after a failed send) kills `ledger_write_and_send_fail`; M1b (the finally releases) kills `extraction_raises`; M2a (both sides keyed on the text again) kills `runtime_footer`; M2b (the id drops the record) kills the same-text test; M4a (the fill branch is not marked) and M4b (the in-place update drops the metadata) kill the F4 test |
| ruff | passes on every changed file |
| commitlore context / guard | context was run on every edited file before editing it. guard is incomplete on every file (notes mirror not fetched) |
| commitlore capture | refused (notes-unfetched); committed with nothing recorded |

### Slice 1 follow-up (5bc45c62)

The coordinator put three of the open items above in scope:

**(a) The in-turn first response deleted its records either way.** `_run_agent_deliver_first_response` now returns whether its answer is settled: nothing was to be sent, it was delivered, or its obligation is in the ledger (`delivery_ledger.obligation_exists`). If the answer is not settled, its records stay on disk. Their ids go into a process-local set (`_keep_inbound_records`), and `_release_turn_marker` skips them when the chain's final is ledgered. That holds whichever event carries them.

Test: `test_a_first_response_neither_ledgered_nor_delivered_keeps_its_record_through_the_chain`, parametrized:
- `as_is`;
- `rewritten`: a pre-dispatch rewrite, so the record lives on the adapter's event;
- `raises`: the chain fails before the follow-up runs.

The failed turn's notice re-asks the message on the transcript, so after `raises` the next process runs it again rather than delivering the stored answer.

**(b) `run_turn.py:4030` read the field directly.** It now reads `getattr(turn_ctx, "persist_user_display_metadata", None)`, the way `acp_admission` is read (lines 2307 and 4191). Both are TurnContext fields with defaults, added after the others, and narrower stand-in contexts lack them. I made the read tolerant rather than changing the stand-in, because that is the existing convention for these fields. `test_queued_human_turn_also_gets_the_visible_fallback` passes.

**(c) A second reply could replace a still-owed earlier obligation.** It can happen: the earlier reply is in the ledger, the record outlives it, the transcript has no completed answer, and the message runs again. So I added a per-run ordinal rather than a proof that it cannot happen. When a record is re-admitted, it starts the first run whose obligation is not in the ledger (`_inbound_run_for`). That run is stamped on the run's user row (`inbound_run`). Delivery and recovery both key on `delivery_ledger.inbound_answer_ref(record, run)`, and the first run is unchanged.

Test: `test_a_rerun_after_a_restart_never_takes_over_the_reply_an_earlier_run_still_owes`. At f89a57ae the earlier row's content becomes the rerun's answer. Now the earlier row is untouched and the rerun has its own.

| Check | Result |
|---|---|
| RED at f89a57ae | 5 failed: (a) as_is, rewritten, raises; (c); (b) |
| Owner/peer file | 93 passed |
| Startup/flush/resume/ledger files | 190 passed |
| Finalizer and queued-ledger files | 107 passed, 2 skipped |
| 68-file set | 31 failed (exactly the baseline list), 1122 passed, 5 skipped |
| Mutations | 4 of 4 killed: M9a (consume regardless of settled) and M9b (the adapter ignores the kept set) kill all three (a) variants; M10 (the ordinal is ignored) kills (c); M11 (direct field read) kills (b) |
| ruff | passes |
| commitlore capture | refused (notes-unfetched); committed with nothing recorded |

### Slice 1 follow-up 2 (ace7c2f5)

**A turn cancelled, or raising, after its answer was stored.** A turn whose own error reply raised too escaped `_handle_message_with_agent`, and so did a cancelled one. `_settle_inbound_record` then deleted its records, even though the agent had stored its completed answer. Before settling a turn that did not end, run_inbound now reads which of its records the transcript holds a completed answer for (`_inbound_answers_stored`, via `_inbound_record_answer`), and those stay. An unreadable transcript counts as stored.

The owed set from (a) now lives in run_inbound (`_inbound_records_owed`). Both `_consume_inbound_record_id` and `_discard_inbound_records` skip it, and the adapter no longer filters. A plain exception inside the turn does not escape: `_hmwa_agent_error_reply` turns it into a failed turn whose notice is delivered and ledgered, and the user is offered /retry. That is the existing design for failed turns.

Test: `test_a_turn_that_raises_after_its_answer_was_stored_keeps_its_record`, with `cancelled` and `raised`.

**The run of a rewritten event.** The wrong run did produce a wrong id. The rerun's row named run 0, so its reply took over the earlier run's still-owed obligation. run_inbound now carries `_inbound_run` from the settled event onto the turn event, with the row ids. Test: the rerun test gains a `rewritten` variant.

| Check | Result |
|---|---|
| RED at 5bc45c62 | 3 failed: cancelled, raised, rerun rewritten |
| Owner/peer file | 96 passed |
| Startup/flush/resume/ledger files | 190 passed |
| Mutations | 3 of 3 killed: M12a (the stored answers are ignored) kills both escapes; M12b (the run is not carried over) kills rerun rewritten; M9b2 (`_consume_inbound_record_id` ignores the owed set) kills all three (a) variants |

### Still open

- **Finding 3** (an answered steer is admitted again after a restart) is slice 2.
- **A failed delivery is answered after the next restart.** A reply that is never ledgered or delivered is delivered after that restart, possibly after later turns. With the ledger off, such a message runs again at each restart until a send succeeds.

## Review r3 of 1524ad04

Scope is the four P1s and nothing else.

### 1. A failed marker write left a window where the answer was lost

**What changed:** when the adapter delivers the turn's reply, the turn no longer deletes its records in its `finally`. `_settle_inbound_record` hands their ids to the adapter's event (`_inbound_records_on_delivery`). `_release_turn_marker` deletes them where it already releases the marker: right after the final reply's obligation is in the ledger, or once nothing more is owed. If the delivery is cut short while the gateway drains, the records stay. A follow-up drained inside the turn no longer deletes its record when its recursive turn returns. Its records ride the outer turn's event, because its answer is the final the adapter sends.

**Test:** `test_a_turn_whose_marker_write_failed_and_whose_reply_never_reached_the_ledger_is_answered` is the reviewer's case. The marker write fails, the agent writes the answer, the turn returns, and the process dies before delivery. At 1524ad04 nothing is delivered; now the answer goes out once and no second turn runs. `_crash_now` now also stops the deferred record delete, since a killed process writes nothing.

### 2. A failed record delete sent a delivered answer again

**What changed:** the reply to a message with a record is now ledgered under `compute_obligation_id(key, "inbound:<recovery id>", text)`, the same id recovery uses. The platform message id is no longer the key. This covers the adapter's final (`_inbound_ledger_ref`, which for a chain is set to the record of the chain's last message) and the in-turn first response. When recovery finds the record answered, its `INSERT OR IGNORE` hits the row that is already there, so nothing is sent again.

**Test:** `test_a_delivered_answer_whose_record_could_not_be_deleted_is_not_sent_again`. `unlink` raises, delivery succeeds, and the process restarts. At 1524ad04 the answer is sent a second time; now nothing is sent and the record goes.

### 3. Folded messages lost their link to the answer across a restart

**What changed:** the user row now also names every record folded into the turn's message before the turn (`display_metadata.inbound_absorbed_ids`). The run_inbound claim takes a snapshot of these ids on `_inbound_row_absorbed`. The in-turn drain falls back to `_inbound_carried`. Recovery matches a record named there and owes the answer under the row's own record, so both records end on one obligation.

**Test:** `test_a_message_merged_into_another_is_answered_by_that_answer_across_a_restart`, the reviewer's reconstruction: the production `merge_pending_message_event` and `_inbound_row_metadata`, two records, then a completed answer. At 1524ad04 the merged message runs as a second turn; now one answer goes out and neither message runs again.

### 4. The completion check rejected completed fallback replies

**What changed:** a recovery or fallback exit has no reply row of its own. When the turn completed (finalize_turn's own `completed`), `_close_transcript_tail` now marks the closing row it appends with `display_metadata.turn_completed`. A failed or interrupted turn's close is not marked. `_inbound_record_answer` accepts `finish_reason == "stop"` or that mark. Rows with neither (a fragment, an interrupted close, a failed reply) still run the message again.

**Tests:**
- `test_a_fallback_reply_the_finalizer_closed_a_completed_turn_with_is_answered_once` runs the real `finalize_turn` with a `fallback_prior_turn_content` exit and keeps the record and the marker. At 1524ad04 the message runs a second time; now the answer goes out once.
- `test_a_failed_turn_the_finalizer_closed_is_not_taken_as_the_answer` is the failed counterpart: the message runs again and the failed reply is never sent. It passes at 1524ad04 too. It guards against false completion.

### Checks at the r3 fix commit

| Check | Result |
|---|---|
| RED at 1524ad04 (the new tests against the 1524ad04 sources) | the four counterexamples fail with each defect's outcome: P1-1 no delivery, P1-2 a second delivery, P1-3 the merged message runs again, P1-4 the message runs again. The failed-turn guard passes, as it should |
| Owner/peer file | 85 passed (80 + 5 new) |
| Startup/flush/resume/ledger files (the 15 files from r2) | 190 passed |
| Finalizer and queued-ledger files (new this round, because agent/turn_finalizer.py changed) | 107 passed, 2 skipped |
| Mutation witnesses | 8 of 8 killed: M1 (delete at turn end), M2 (ledger under the platform id), M3a (the row names only the primary), M3b (the matcher ignores absorbed ids), M4a (no finalizer mark), M4b (the matcher ignores the mark), M4c (the matcher accepts any close, which kills the failed-turn guard), M4d (the finalizer is not told the turn completed) |
| 68-file regression set | not run (the packet excludes it) |
| ruff | passes on every changed file |
| commitlore context / guard | run on every edited file before editing it, except agent/message_metadata.py, whose context was read after the edit. guard flagged nothing |
| commitlore capture | refused (notes-unfetched); committed with nothing recorded |

### Still open after r3

- **Closed by r3 #2:** the r2 item below about a record whose delete fails after its answer was ledgered or delivered.
- **Steered and redirected messages (P1-3):** these join a turn whose user row is already written, so they are linked to its answer only in memory. If a restart lands after that answer is written and before it is in the ledger, such a message runs once more. That gives a duplicate answer, never a lost one. A link written when the message is steered would be wrong for a steer the turn leaves over: that steer runs as its own in-band follow-up, so the link would consume a message nobody answered.
- **Reply text:** the obligation id still includes the reply text. P1-2 holds as long as the transcript's answer (`_crash_left_reply`) and the delivered text match. The marker path's existing same-reply check already depends on the same match.


## Review r2 of 1600122b

### 1. Queued follow-ups lost their record before they completed

**What changed:** queueing and dequeueing never delete a record now. `_settle_unclaimed_inbound` keeps the record of any message still in the pending slot, the overflow FIFO or a busy-text debounce buffer. The shutdown flush skips any event whose record is on disk.

A message with no turn of its own carries its record onto the event that absorbed it (`carry_inbound_records`), and that event's turn settles both. Nothing deletes the record at the hand-over. This covers three cases:
- merged into the queued head;
- replaced by a newer message in the slot;
- steered or redirected into the running turn.

For the in-turn drain (`_run_agent_drain_pending` and `_run_agent_queued_followup`):
- The drained message keeps its record.
- The record of the message whose answer is sent before the follow-up runs is deleted only once that answer has been handed to the ledger.
- The drained message's own record is deleted when its recursive turn returns, unless a restart cut that turn short.

`/stop`, `/new` and `/reset`, which drop the queued head or the debounce buffer, delete those records.

**Tests:**
- `test_a_queued_follow_up_drained_inside_the_running_turn_keeps_its_record_until_its_turn_completes` is the reproduction. It fails at 1600122b and passes now.
- `…settles_its_record_when_it_completes` checks the normal end of the same path.
- `test_a_message_folded_into_another_keeps_its_record_until_that_message_is_settled` and `test_a_message_steered_into_a_running_turn_keeps_its_record_until_that_turn_ends` cover the merge, replace and steer cases.
- The existing busy-mode test now also checks that no record is left once the turn ends.

**Mutation witnesses killed:**
- R1a: delete the record when the message is queued.
- R1b: never settle the drained message's turn.
- R1c: keep the first record past its ledger hand-off.
- R1d: don't carry a steered message's record.
- R1e: don't carry a merged message's record.

### 2. Orphan rescue went ahead after a failed write

**What changed:** before the claim, `_peek_orphaned_fifo` finds the orphan that `_hm_rescue_orphaned_fifo` would run. If it has no record yet, its record is written and awaited. If that write fails, nothing is claimed, the orphan stays in the FIFO, and the owner is told the message was not processed. The best-effort write after the claim is gone.

**Test:** `test_an_orphan_whose_record_cannot_be_written_is_not_claimed`. It fails at 1600122b and passes now. Mutation R2, claiming without the write, is killed.

### 3. Messages without a platform id got duplicate answers

**What changed:** each record's recovery id is now written onto the user transcript row as `display_metadata.inbound_recovery_id`. That happens on the agent's own user row (`persist_user_display_metadata`), on the gateway's fallback user row (`_hmwa_user_transcript_entry`), and on rows for follow-ups drained inside a turn. Recovery matches on that id only, never on `message_id`, which also covers the case where the marker write failed.

**Test:** `test_a_message_without_a_platform_id_is_not_answered_twice`. It fails at 1600122b and passes now.

**Mutation witnesses killed:**
- R3a: match on `message_id` again.
- R3b: don't tag the user row.

### 4. An incomplete fragment counted as a finished answer

**What changed:** `_inbound_record_answer` now counts only an assistant row that ended the model's turn: `finish_reason == "stop"`, no tool calls, non-empty text, and not the failed-turn notice. An assistant row that ends the turn any other way (a truncated `length` fragment, an interrupted or empty reply) means the turn did not complete, so the message is re-admitted.

**Test:** `test_a_partial_answer_in_the_transcript_is_not_taken_as_the_answer`. It fails at 1600122b and passes now. The message runs again and the fragment is never sent. Mutation R4, accepting any assistant text, is killed.

### 5. Two different answers with the same text were delivered once

**What changed:** `record_crash_left_reply` takes an identity key through its existing `obligation_id`, but it also suppressed any reply whose text matched one ledgered since the message was asked. I added `same_reply_dedupe=False`, which drops that text check, so the only deduplication left is the ledger's own primary key on `obligation_id`. The hand-off key is `compute_obligation_id(key, "inbound:<recovery id>", text)`, so two records never collapse into one.

The pass that ledgers answered records now runs in `_start_recover_previous_run`, before the marker recovery. That way the marker path's existing same-reply check finds the reply already owed instead of owing it a second time.

**Test:** `test_two_messages_answered_with_the_same_text_are_each_delivered`. It fails at 1600122b and passes now. Mutation R5, restoring the text check, is killed.

**Changed test:** `test_a_crash_after_the_turn_completed_before_its_record_was_consumed_runs_no_second_turn` was rewritten to the real turn-completion boundary: the turn has returned, and the process dies before the record is deleted and before the reply is in the ledger. The reply is owed and delivered once, and no second turn runs.

## Review r1 of 479a19db, closed in 1600122b and still holding

- **P1-a**, store before acceptance: mutation witness A1 is still killed. I removed the dequeue-time store A2 guarded: a queued message now keeps the record it got on arrival, which made that store a no-op (A2 survived for exactly that reason).
- **P1-b**, hand an answered record's reply to the ledger before deleting it: B1 is still killed.
- **P1-c**, write off the event loop: C1 is still killed.

## Checks at the r2 fix commit

| Check | Result |
|---|---|
| Owner/peer file | 80 passed |
| 68-file regression set | 31 failed, the exact baseline list (none new, none missing); 1122 passed |
| Startup/flush/resume/ledger files | 190 passed |
| Extra adapter files | 167 passed, 1 skipped |
| Mutation witnesses | 17 of 17 killed: the ten above plus A1, B1, C1, M2, M5, M8 and M9 re-pointed at this head (A2 retired with the code it guarded) |
| ruff | passes on every changed file |
| commitlore context | run on `gateway/run_turn.py`, `gateway/run_busy.py` and `gateway/delivery_ledger.py` before editing them |
| commitlore guard / capture | still cannot run: there is no notes mirror |

## Open items

- **Item 1, closed by r2 #1:** a follow-up drained inside a running turn now keeps its record.
- **Item 2, closed by r2 #3:** records are matched by recovery id, so a message without a platform id is no longer answered twice.
- **Item 3, closed in 1600122b:** an answered but undelivered reply is owed through the ledger.
- **Item 4, non-blocking per the review:** the write is off the loop. Deleting a record, reading records at startup and reconnect, and the existence check in the shutdown flush still run on the loop.
- **Item 5, closed by r2 #2:** the orphan is now durable before the claim.
- **Item 6, non-blocking per the review:** unchanged. No re-admission happens on a boot where the restart-loop breaker trips, and records have no age limit.
- **Item 7, non-blocking per the review:** unchanged. The user row now names its record, but skipping a second row needs the agent's turn-start flush to know the first one exists, so the fix does not fall out of item 3.

## Still open

- A record whose delete fails after its answer was already ledgered or delivered is owed once more at the next start, under its own key, with the recovered-reply marker. The rule that only the record's identity deduplicates rules out recognising that the earlier delivery matched.
- A message dropped from a queue implicitly keeps its record and runs again after the next restart instead of disappearing. That covers a stale-lock heal or a session-state reset; the explicit `/stop`, `/new` and `/reset` paths delete their records.
- Queued owner messages now come back after a restart as re-admitted turns instead of transcript rows, because the record outranks the flush copy. This is a change from 420c58a1's behaviour, and it is intended.

## Review of 33bdb24133: retention and the proxy parser

- **Retention, folded messages** (`gateway/delivery_ledger.py`, `_held_by_inbound_records`): a record folded into another is settled from the reply of the message that absorbed it, an id only the transcript names, and the absorbing record may already be deleted. Retention now keeps every obligation of the session of any record still on disk, and every attachment-only reply (components with no row), until the record is gone. A session scope is used because the ledger cannot read the transcript. Test: `test_the_ledger_keeps_the_reply_that_settles_a_record_through_another_messages_answer`.
- **Retention, unreadable records**: `load_inbound_records` omits a file it cannot read. Retention now lists the spool itself and reads every file the loader omitted. A read fault (anything but a parse error) makes retention keep everything this time. A file that reads but is not a record holds nothing, as before. A record whose event cannot be decoded still holds its session. Test: `test_an_unreadable_record_still_holds_the_ledger_evidence_it_settles_from`.
- **Proxy parser** (`gateway/run_turn.py`): `_run_agent_inner` passes the answer reference of the row to `_run_agent_via_proxy`, which calls `show_owner_reply_text()` on its stream consumer under the same `_owes_tracked_reply` test as the normal and queued paths. Tests: `TestProxyOwnerReplyParser` (owner reply loses the declaration, another reply keeps the old display).

RED: all three new owner/peer tests and the proxy owner test fail at 33bdb24133 (the two ledger probes send the answer a second time) and pass now. Evidence is in `owner-durable-v2-evidence/` (`red.*`, `green-*`).

Still open: a record that stays on disk with no age limit holds its session's ledger rows for as long as it stays. The Windows/no-dirfd capability is untouched.

## Review r6 of 6765aee9: a spool that cannot be listed

- **Fix** (`gateway/delivery_ledger.py`, `_held_by_inbound_records`): the spool is listed with `os.listdir`. A missing directory (`FileNotFoundError`) means no records; any other failure to list (a permission fault included) raises into the existing fail-closed path, so retention prunes nothing this time. `Path.glob` and `load_inbound_records` read an unlistable spool as an empty one, which is why neither could be relied on.
- **Test**: `test_a_spool_that_cannot_be_listed_still_holds_the_ledger_evidence_its_records_settle_from`. A delivered message keeps its record (deleting it fails), the spool directory is chmod'd to 0, retention advances 8 days, the mode is restored, and the next start sends nothing and runs no turn. Skipped when run as root.
- **RED** at 6765aee9: the next start sends the recovered-reply duplicate (`red-r6.*`). **GREEN**: `green-r6.*`, 329 passed over the owner/peer file, the two ledger files, the proxy file, housekeeping-profile-scope and completion-delivery, exit 0.
