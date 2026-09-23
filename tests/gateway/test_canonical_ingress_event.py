import json

import pytest

from gateway.canonical_surface import CanonicalIngressEvent


def test_from_json_bytes_parses_only_the_canonical_event_fields_without_normalizing_them():
    raw = (
        b'{"binding":" binding ","event_id":" event ",'
        b'"author_id":" author ","channel_id":" channel ","text":" hello "}'
    )

    assert CanonicalIngressEvent.from_json_bytes(raw) == CanonicalIngressEvent(
        binding=" binding ",
        event_id=" event ",
        author_id=" author ",
        channel_id=" channel ",
        text=" hello ",
    )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"binding":"a","binding":"b","event_id":"event","author_id":"author","channel_id":"channel","text":"text"}',
        json.dumps(
            {
                "binding": "binding",
                "event_id": "event",
                "author_id": "author",
                "channel_id": "channel",
                "text": "text",
                "destination": "not-permitted",
            }
        ).encode(),
        json.dumps(
            {
                "binding": "binding",
                "event_id": "event",
                "author_id": "author",
                "channel_id": "channel",
            }
        ).encode(),
        b'[]',
        b'{"binding":1,"event_id":"event","author_id":"author","channel_id":"channel","text":"text"}',
    ],
)
def test_from_json_bytes_rejects_non_closed_canonical_payloads(raw):
    with pytest.raises(ValueError, match="^canonical_invalid_request$"):
        CanonicalIngressEvent.from_json_bytes(raw)


@pytest.mark.parametrize("field", ("binding", "event_id", "author_id", "channel_id", "text"))
def test_from_json_bytes_rejects_whitespace_only_required_fields(field):
    payload = {
        "binding": "binding",
        "event_id": "event",
        "author_id": "author",
        "channel_id": "channel",
        "text": "text",
    }
    payload[field] = " \t\n "

    with pytest.raises(ValueError, match="^canonical_invalid_request$"):
        CanonicalIngressEvent.from_json_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize(
    ("field", "length", "accepted"),
    [
        pytest.param(field, length, accepted, id=f"{field}-{length}")
        for field in ("binding", "event_id", "author_id", "channel_id")
        for length, accepted in ((256, True), (257, False))
    ]
    + [
        pytest.param("text", length, accepted, id=f"text-{length}")
        for length, accepted in ((16_384, True), (16_385, False))
    ],
)
def test_from_json_bytes_enforces_required_field_length_boundaries(field, length, accepted):
    value = "x" * length
    payload = {
        "binding": "binding",
        "event_id": "event",
        "author_id": "author",
        "channel_id": "channel",
        "text": "text",
    }
    payload[field] = value

    if accepted:
        event = CanonicalIngressEvent.from_json_bytes(json.dumps(payload).encode())
        assert getattr(event, field) == value
    else:
        with pytest.raises(ValueError, match="^canonical_invalid_request$"):
            CanonicalIngressEvent.from_json_bytes(json.dumps(payload).encode())
