# =============================================================================
#  ui — the chat page served straight off the engram service
#  why: a plain chatbot that "just works" and keeps a consistent conversation,
#  same-origin with the OpenAI API so there is no CORS or extra server; talking
#  to it is how engram individuates, so the UI is the front door to the whole
#  thing. CSS/JS are served as EXTERNAL content-hash-cache-busted static files
#  ({{ static:chat.css }} → /static/chat.css?v=<sha256[:12]>), so a browser never
#  serves a stale asset after a deploy — no hard-refresh is ever needed. The hash
#  changes only on content change, so the assets cache `immutable` (one year).
# =============================================================================
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse

router = APIRouter()
_HERE = Path(__file__).resolve().parent
_PAGE = _HERE / "chat.html"
_STATIC_DIR = _HERE / "static"
_CONTENT_TYPES = {".css": "text/css", ".js": "application/javascript", ".json": "application/json",
                  ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".woff2": "font/woff2"}

# ##################################################################
# static hashes
# content-hash per static file, computed once at startup (or first use). The hash
# changes only when the file content changes, so a URL carrying ?v=<hash> is safe
# to cache as immutable for a year; the moment we edit chat.js, its ?v= changes
# and every browser fetches the new version on the next page load
_static_hashes: dict[str, str] = {}


def _build_static_hashes() -> None:
    if _static_hashes:
        return
    if not _STATIC_DIR.is_dir():
        return
    for f in _STATIC_DIR.iterdir():
        if f.is_file():
            _static_hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:12]


def resolve_static_tags(html: str) -> str:
    _build_static_hashes()

    def _replace(m: re.Match) -> str:
        name = m.group(1).strip()
        h = _static_hashes.get(name, "0")
        return f"/static/{name}?v={h}"

    return re.sub(r"\{\{\s*static:([^}]+)\}\}", _replace, html)


# ##################################################################
# chat page
# serve the chat UI at the site root with its static tags resolved, and hand it a
# same-origin httpOnly cookie carrying the API token so its consolidate/verify
# actions are authenticated without ever exposing the secret to page JavaScript
@router.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    response = HTMLResponse(resolve_static_tags(_PAGE.read_text()))
    response.set_cookie("engram_token", request.app.state.engram.token, httponly=True, samesite="strict")
    return response


# ##################################################################
# static assets
# serve CSS/JS with immutable one-year caching keyed by the content hash in the
# query string; reject path traversal (no "/" or leading "." in the filename)
@router.get("/static/{filename:path}")
def static_asset(filename: str) -> FileResponse:
    if "/" in filename or filename.startswith("."):
        raise HTTPException(404)
    path = _STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(404)
    media_type = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type,
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})
