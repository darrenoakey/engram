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


# ##################################################################
# command name recall
# the dedicated "did it actually learn me" E2E test — spins an isolated 0.8B
# server, tells it a name, waits for the background dream to consolidate, and
# checks a FRESH conversation recalls it. Standalone (not in the pytest suite)
def command_name_recall(_args) -> int:
    from evaluation import name_recall

    result = name_recall.run_name_recall()
    print(name_recall.scoreboard(result))
    return 0 if result.passed else 1


# ##################################################################
# command family recall
# the multi-fact E2E test — tells four family facts in one turn, verifies a fresh
# conversation recalls each relationship (not a confabulation). Standalone
def command_family_recall(_args) -> int:
    from evaluation import family_recall

    result = family_recall.run_family_recall()
    print(family_recall.scoreboard(result))
    return 0 if result.passed else 1


# ##################################################################
# command deploy
# the greenline deploy contract: restart the prod service via `auto` and wait for
# it to come up (the 9B takes ~30-60s to load). Idempotent — a restart of an
# already-running service that comes back healthy is a success. Exits nonzero on
# unhealthy so the gate rolls back. Runs in the canonical checkout (cwd = main)
def command_deploy(args) -> int:
    import subprocess
    import time

    import httpx

    service = args.service
    config = load_config_for_deploy()
    url = f"http://{config.server.host}:{config.server.port}"
    restart = subprocess.run(["auto", "restart", service], capture_output=True, text=True)
    if restart.returncode != 0:
        print(f"auto restart {service} failed: {restart.stderr.strip()}")
        return 1
    print(f"restarted {service}; waiting for {url}/v1/brain ...")
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/v1/brain", timeout=5)
            if r.status_code == 200 and "updates" in r.json():
                print(f"healthy ({r.json().get('model_path', '?')})")
                return 0
        except Exception:
            pass
        time.sleep(2)
    print("service did not become healthy within 180s")
    return 1


# ##################################################################
# command health
# the greenline probe contract: a read-only check that prod is serving. No restart.
# Exits 0 if healthy, 1 otherwise
def command_health(args) -> int:
    import httpx

    config = load_config_for_deploy()
    url = f"http://{config.server.host}:{config.server.port}"
    try:
        r = httpx.get(f"{url}/v1/brain", timeout=10)
        if r.status_code == 200:
            print(f"healthy ({r.json().get('model_path', '?')})")
            return 0
    except Exception as e:
        print(f"unhealthy: {e}")
    return 1


def load_config_for_deploy():
    from common.config import load_config

    return load_config()


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
    sub.add_parser("name-recall", help="dedicated E2E test: tell it a name, fresh chat recalls it")
    sub.add_parser("family-recall", help="multi-fact E2E test: tell it your family, fresh chat recalls each")
    deploy = sub.add_parser("deploy", help="restart the prod service via auto and health-check it")
    deploy.add_argument("--service", default="engram", help="auto service name (default: engram)")
    sub.add_parser("health", help="probe prod health (no restart)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    handlers = {"check": command_check, "serve": command_serve, "status": command_status,
                "proof": command_proof, "token": command_token, "name-recall": command_name_recall,
                "family-recall": command_family_recall, "deploy": command_deploy, "health": command_health}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
