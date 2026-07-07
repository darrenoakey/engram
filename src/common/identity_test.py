# =============================================================================
#  identity_test — real credential-store round-trip on an isolated service name
#  why: if token storage breaks, the feedback API either locks out or opens up
# =============================================================================
from common import identity

TEST_SERVICE = "engram-selftest"


def test_token_created_and_stable():
    identity.remove_token(TEST_SERVICE)
    first = identity.get_or_create_token(TEST_SERVICE)
    second = identity.get_or_create_token(TEST_SERVICE)
    assert first == second
    assert len(first) >= 32
    identity.remove_token(TEST_SERVICE)


def test_remove_is_idempotent():
    identity.remove_token(TEST_SERVICE)
    identity.remove_token(TEST_SERVICE)
    fresh = identity.get_or_create_token(TEST_SERVICE)
    assert fresh
    identity.remove_token(TEST_SERVICE)
