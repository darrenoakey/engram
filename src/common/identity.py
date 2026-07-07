# =============================================================================
#  identity — API auth token in the OS credential store
#  why: /v1/feedback mutates model weights; an unauthenticated feedback
#  endpoint is a weight-poisoning API, and secrets never live in env or files
# =============================================================================
from __future__ import annotations

import secrets

import keyring
import keyring.errors

SERVICE = "engram"
ACCOUNT = "api-token"


def get_or_create_token(service: str = SERVICE, account: str = ACCOUNT) -> str:
    existing = keyring.get_password(service, account)
    if existing:
        return existing
    token = secrets.token_urlsafe(32)
    keyring.set_password(service, account, token)
    return token


def remove_token(service: str, account: str = ACCOUNT) -> None:
    try:
        keyring.delete_password(service, account)
    except keyring.errors.PasswordDeleteError:
        return
