# =============================================================================
#  ui — the chat page served straight off the engram service
#  why: a plain chatbot that "just works" and keeps a consistent conversation,
#  same-origin with the OpenAI API so there is no CORS or extra server; talking
#  to it is how engram individuates, so the UI is the front door to the whole
#  thing. The page is a single self-contained file (no external assets).
# =============================================================================
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()
_PAGE = Path(__file__).resolve().parent / "chat.html"


# ##################################################################
# chat page
# serve the self-contained chat UI at the site root, and hand it a same-origin
# httpOnly cookie carrying the API token so its consolidate/verify actions are
# authenticated without ever exposing the secret to page JavaScript
@router.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    response = HTMLResponse(_PAGE.read_text())
    response.set_cookie("engram_token", request.app.state.engram.token, httponly=True, samesite="strict")
    return response
