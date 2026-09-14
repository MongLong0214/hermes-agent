"""Native Buzz push to the authenticated existing-only canonical surface.

This is a connector, not a gateway adapter/actor. It never creates sessions,
resolves lineage, polls a feed, retries a refused turn, or publishes a reply.
NIP-42 authentication is reused from the existing Buzz transport. Relay event
signature admission remains the authenticated, configured relay's contract.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

import aiohttp
import websockets

from .adapter import BuzzAdapter
from .nostr_auth import public_key_hex


class CanonicalConnector:
    """One native subscription; errors stop without acknowledging delivery."""

    _authenticate_websocket = BuzzAdapter._authenticate_websocket

    def __init__(self, *, relay_url, endpoint, binding, recipient, author,
                 channel, since, private_key, api_key):
        try:
            target, relay = urlsplit(endpoint), urlsplit(relay_url)
            valid = (
                target.scheme == "http" and target.hostname in {"127.0.0.1", "::1"}
                and target.port is not None and target.username is None
                and target.path == "/v1/canonical-surface/events"
                and not target.query and not target.fragment
                and relay.hostname and relay.username is None and not relay.fragment
                and (relay.scheme == "wss" or (relay.scheme == "ws" and relay.hostname in {"127.0.0.1", "::1"}))
                and all(isinstance(v, str) and v.strip() == v and 0 < len(v) <= 256
                        for v in (binding, channel))
                and isinstance(author, str) and re.fullmatch(r"[0-9a-f]{64}", author)
                and isinstance(api_key, str) and api_key.strip()
                and all(32 < ord(char) < 127 for char in api_key)
                and public_key_hex(private_key) == recipient
                and type(since) is int and since >= 0
            )
        except Exception:
            valid = False
        if not valid:
            raise ValueError("canonical_connector_config_invalid")
        self.relay_url = relay_url
        self.endpoint = endpoint
        self.binding = binding
        self.recipient = recipient
        self.author = author
        self.channel = channel
        self.since = since
        self._private_key = private_key
        self._api_key = api_key

    def _websocket_url(self):
        return self.relay_url

    def _admitted(self, event):
        if not isinstance(event, dict):
            return False
        tags = event.get("tags")
        if not isinstance(tags, list) or any(
            not isinstance(tag, list) or len(tag) < 2
            or not all(isinstance(value, str) for value in tag) for tag in tags
        ):
            return False
        return (
            type(event.get("kind")) is int and event["kind"] == 9
            and event.get("pubkey") == self.author
            and isinstance(event.get("id"), str)
            and re.fullmatch(r"[0-9a-f]{64}", event["id"]) is not None
            and type(event.get("created_at")) is int and event["created_at"] >= self.since
            and [tag[1] for tag in tags if tag[0] == "h"] == [self.channel]
            and self.recipient in [tag[1] for tag in tags if tag[0] == "p"]
            and isinstance(event.get("content"), str)
            and 0 < len(event["content"].strip()) <= 16384
        )

    async def run(self, on_receipt):
        """Forward pushed events; only a completed HTTP response is receipted."""
        async with aiohttp.ClientSession(trust_env=False) as http:
            async with websockets.connect(self.relay_url, max_size=262144) as ws:
                await self._authenticate_websocket(ws)
                await ws.send(json.dumps(["REQ", "canonical-buzz", {
                    "kinds": [9], "authors": [self.author], "#p": [self.recipient],
                    "#h": [self.channel], "since": self.since,
                }]))
                async for raw in ws:
                    frame = json.loads(raw)
                    if not isinstance(frame, list) or len(frame) != 3 or frame[:2] != ["EVENT", "canonical-buzz"]:
                        continue
                    event = frame[2]
                    if not self._admitted(event):
                        continue
                    payload = {"binding": self.binding, "event_id": event["id"],
                               "author_id": event["pubkey"], "channel_id": self.channel,
                               "text": event["content"]}
                    async with http.post(self.endpoint, json=payload, allow_redirects=False,
                                         headers={"Authorization": "Bearer " + self._api_key}) as response:
                        if response.status != 200:
                            raise ValueError("canonical_delivery_refused")
                        result = await response.json()
                        if (not isinstance(result, dict) or set(result) != {"event_id", "text"}
                                or result["event_id"] != event["id"] or not isinstance(result["text"], str)):
                            raise ValueError("canonical_delivery_refused")
                    await on_receipt(event["id"])


def main(argv=None):
    """Check or explicitly activate a secret-free connector configuration."""
    import argparse
    import asyncio
    from pathlib import Path
    from agent.secret_scope import get_secret

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text())
        fields = {"relay_url", "endpoint", "binding", "recipient", "author", "channel", "since", "private_key_file"}
        if not isinstance(config, dict) or set(config) != fields:
            raise ValueError("canonical_connector_config_invalid")
        private_key = Path(config.pop("private_key_file")).read_text().strip()
        connector = CanonicalConnector(**config, private_key=private_key,
                                       api_key=get_secret("API_SERVER_KEY", ""))
        if args.check:
            print(json.dumps({"configuration_valid": True, "activated": False}))
            return 0

        async def receipt(event_id):
            print(json.dumps({"canonical_event_id": event_id, "http_receipt": True}), flush=True)

        asyncio.run(connector.run(receipt))
        return 0
    except Exception:
        # No credential, relay error text, event content or upstream body leaks.
        print(json.dumps({"error": "canonical_connector_stopped", "delivery_proven": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
