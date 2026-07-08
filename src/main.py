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


def command_proof(args) -> int:
    from common.config import load_config
    from common.identity import get_or_create_token
    from evaluation import proof

    config = load_config()
    url = args.url or f"http://{config.server.host}:{config.server.port}"
    result = proof.run_proof(url, get_or_create_token(), rounds=args.rounds)
    return 0 if result.passed else 1


# ##################################################################
# command token
# print the API bearer token (minted in the OS keychain on first boot) so an
# operator can authenticate feedback/brain calls without digging in the keychain
def command_token(_args) -> int:
    from common.identity import get_or_create_token

    print(get_or_create_token())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engram", description="self-modifying local inference engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="quality gate: ruff + full test suite")
    serve = sub.add_parser("serve", help="start the engram service")
    serve.add_argument("--model", default=None, help="override serving model path")
    status = sub.add_parser("status", help="show live brain status")
    status.add_argument("--json", action="store_true")
    proof_cmd = sub.add_parser("proof", help="run the end-to-end proof of life against the live service")
    proof_cmd.add_argument("--url", default=None, help="engram base url (default from config host/port)")
    proof_cmd.add_argument("--rounds", type=int, default=6, help="reward rounds per phase")
    sub.add_parser("token", help="print the API bearer token from the keychain")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    handlers = {"check": command_check, "serve": command_serve, "status": command_status,
                "proof": command_proof, "token": command_token}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
