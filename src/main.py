# =============================================================================
#  main — engram command dispatch
#  why: one entry point behind the run facade; serve is the daemon,
#  the rest are thin operator commands against the live service
# =============================================================================
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def command_check(_args) -> int:
    root = Path(__file__).resolve().parents[1]
    ruff = subprocess.run([sys.executable, "-m", "ruff", "check", "src"], cwd=root)
    tests = subprocess.run([sys.executable, "-m", "pytest"], cwd=root)
    return ruff.returncode or tests.returncode


def command_serve(args) -> int:
    from server.app import run_server

    return run_server(args)


def command_status(args) -> int:
    from server.client import show_status

    return show_status(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engram", description="self-modifying local inference engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="quality gate: ruff + full test suite")
    serve = sub.add_parser("serve", help="start the engram service")
    serve.add_argument("--model", default=None, help="override serving model path")
    status = sub.add_parser("status", help="show live brain status")
    status.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    handlers = {"check": command_check, "serve": command_serve, "status": command_status}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
