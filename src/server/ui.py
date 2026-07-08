# =============================================================================
#  ui — the chat page served straight off the engram service
#  why: a plain chatbot that "just works" and keeps a consistent conversation,
#  same-origin with the OpenAI API so there is no CORS or extra server; talking
#  to it is how engram individuates, so the UI is the front door to the whole
#  thing. The page is a single self-contained file (no external assets).
# =============================================================================
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()
_PAGE = Path(__file__).resolve().parent / "chat.html"


# ##################################################################
# chat page
# serve the self-contained chat UI at the site root
@router.get("/", response_class=HTMLResponse)
def chat_page() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text())
