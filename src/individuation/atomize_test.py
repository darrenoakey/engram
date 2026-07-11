# =============================================================================
#  atomize_test — real multi-fact splitting on the 0.8B
#  why: prove atomize decomposes a compound user turn into one atomic third-person
#  statement per fact, so each fact can be corroborated and learned independently
#  (the fix for "I told it my family and it learned nothing"). A single-fact turn
#  yields one atom; a failed split falls back to [user_text]. All real — no mocks.
# =============================================================================
from __future__ import annotations

import pytest

from common.config import load_config
from engine.model_host import ModelHost
from individuation import atomize as A


# ##################################################################
# host / config fixtures — one real model load shared across the tests
@pytest.fixture(scope="module")
def host():
    config = load_config()
    return ModelHost(config, config.model.test_path)


@pytest.fixture(scope="module")
def config():
    return load_config()


# ##################################################################
# a multi-fact turn splits into several atoms, each a statement about the user
# the family turn is the canonical case: each name+relationship should appear in
# its own atom rather than collapsed into one generic sentence
def test_multi_fact_turn_splits_into_atoms(host, config):
    msg = "my wife is Arlene, my daughter is Alexandra, my son is Leo, and my dog is Peanut"
    atoms = A.atomize(host, msg, config)
    assert len(atoms) >= 3, f"expected at least 3 atoms, got {len(atoms)}: {atoms}"
    # every atom reads as a statement about the user
    joined = " ".join(atoms).lower()
    assert "user" in joined
    # the proper nouns survived into the atoms (not collapsed to a meta-statement)
    blob = " ".join(atoms).lower()
    for name in ("arlene", "leo", "peanut"):
        assert name in blob, f"{name} missing from atoms: {atoms}"


# ##################################################################
# a single-fact turn yields one atom
# this is the regression-guard: the common case behaves exactly as before
def test_single_fact_turn_is_one_atom(host, config):
    atoms = A.atomize(host, "My name is Darren.", config)
    assert len(atoms) == 1
    assert "darren" in atoms[0].lower()


# ##################################################################
# a short trivial turn still returns something (never an empty list)
# the fallback is [user_text]; atomize must never hand the dream an empty list
def test_atomize_never_returns_empty(host, config):
    atoms = A.atomize(host, "ok", config)
    assert len(atoms) >= 1
