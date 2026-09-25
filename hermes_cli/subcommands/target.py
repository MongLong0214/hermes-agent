"""``hermes target`` subcommand parser.

Stdlib-only on purpose: ``hermes_cli._early_recovery`` registers this same grammar on the real
top-level parser to recognise ``target bind`` before dotenv resolves external secret sources.
"""

from __future__ import annotations

import argparse

from hermes_cli.subcommands._shared import add_json_flag


def _cmd_target_bind(args: argparse.Namespace) -> int:
    from hermes_cli.target_bind import cmd_target_bind

    return cmd_target_bind(args)


def build_target_parser(subparsers) -> None:
    """Attach the ``target`` subcommand group to ``subparsers``."""
    target_parser = subparsers.add_parser(
        "target",
        help="Run local target binding operations",
        description="Local operations an external controller runs against this profile's session store.",
    )
    target_subparsers = target_parser.add_subparsers(dest="target_command", metavar="<action>")
    bind_parser = target_subparsers.add_parser(
        "bind",
        help="Verify and persist a target-bind receipt from stdin JSON",
        description="Read one JSON request from stdin, bind the caller identity to the session "
            "lineage, and print the receipt (or a closed error) as JSON on stdout.",
    )
    add_json_flag(bind_parser, help="Read exactly one JSON request from standard input")
    bind_parser.set_defaults(func=_cmd_target_bind)
