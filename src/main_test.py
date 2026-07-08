# =============================================================================
#  main_test — the CLI dispatch surface
#  why: the operator entry point must parse every subcommand and the token
#  command must return the real keychain token; serve/status/proof need a live
#  service so they are exercised by the server and proof suites instead
# =============================================================================
import io
from contextlib import redirect_stdout

import pytest

import main
from common.identity import get_or_create_token


def test_parser_accepts_every_command():
    parser = main.build_parser()
    for command in ("check", "serve", "status", "proof", "token"):
        args = parser.parse_args([command])
        assert args.command == command


def test_proof_flags_parse():
    args = main.build_parser().parse_args(["proof", "--url", "http://x:1", "--rounds", "3"])
    assert args.url == "http://x:1" and args.rounds == 3


def test_missing_command_exits():
    with pytest.raises(SystemExit):
        main.build_parser().parse_args([])


def test_token_command_prints_keychain_token():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main.command_token(None)
    assert code == 0
    assert buffer.getvalue().strip() == get_or_create_token()
