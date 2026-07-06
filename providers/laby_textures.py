from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from providers.http_client import fetch_with_fallback
from providers.laby_challenge import laby_v3_path_get

_HTML_TAG_RE = re.compile(r"<[^>]+>")

VALID_TEXTURE_TYPES = {"skin", "cape", "cloak", "bandana"}


def _normalize_type(texture_type: str) -> Optional[str]:
    t = (texture_type or "").strip().lower()
    return t if t in VALID_TEXTURE_TYPES else None


def fetch_texture_meta(
    texture_hash: str, texture_type: str, scraper_session: Any
) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    """Public texture metadata: display name, description, difference hash."""
    ttype = _normalize_type(texture_type)
    if not ttype:
        return None, 400, f"Unknown texture type: {texture_type}"
    h = (texture_hash or "").strip().lower()
    if not h:
        return None, 400, "Missing texture hash"

    response = fetch_with_fallback(
        f"https://laby.net/api/v3/texture/{h}/{ttype}/meta",
        session=scraper_session,
        allow=[200, 404],
    )
    if response is None:
        return None, 0, "No response from Laby API"
    if response.status_code == 404:
        return None, 404, "Texture not found"
    if response.status_code != 200:
        return None, response.status_code, response.text[:200]
    try:
        data = response.json()
    except Exception:
        return None, response.status_code, "Invalid JSON from Laby API"

    description = data.get("description")
    if isinstance(description, dict):
        description = description.get("en") or next(iter(description.values()), None)
    if description:
        description = _HTML_TAG_RE.sub("", description)

    return (
        {
            "name": data.get("name"),
            "description": description,
            "difference_hash": data.get("difference_hash"),
        },
        200,
        None,
    )


def fetch_texture_users(
    texture_hash: str, texture_type: str, scraper_session: Any
) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    """Public reverse lookup: players who have worn this exact texture."""
    ttype = _normalize_type(texture_type)
    if not ttype:
        return None, 400, f"Unknown texture type: {texture_type}"
    h = (texture_hash or "").strip().lower()
    if not h:
        return None, 400, "Missing texture hash"

    response = fetch_with_fallback(
        f"https://laby.net/api/v3/texture/{h}/{ttype}/users",
        session=scraper_session,
        allow=[200, 404],
    )
    if response is None:
        return None, 0, "No response from Laby API"
    if response.status_code == 404:
        return None, 404, "Texture not found"
    if response.status_code != 200:
        return None, response.status_code, response.text[:200]
    try:
        data = response.json()
    except Exception:
        return None, response.status_code, "Invalid JSON from Laby API"
    return (
        {"count": data.get("count"), "users": data.get("users") or []},
        200,
        None,
    )


def fetch_texture_tags(
    texture_hash: str, texture_type: str, scraper_session: Any
) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    """Public community tags on a texture."""
    ttype = _normalize_type(texture_type)
    if not ttype:
        return None, 400, f"Unknown texture type: {texture_type}"
    h = (texture_hash or "").strip().lower()
    if not h:
        return None, 400, "Missing texture hash"

    response = fetch_with_fallback(
        f"https://laby.net/api/v3/texture/{h}/{ttype}/tags",
        session=scraper_session,
        allow=[200, 404],
    )
    if response is None:
        return None, 0, "No response from Laby API"
    if response.status_code == 404:
        return None, 404, "Texture not found"
    if response.status_code != 200:
        return None, response.status_code, response.text[:200]
    try:
        data = response.json()
    except Exception:
        return None, response.status_code, "Invalid JSON from Laby API"

    tags = [
        {
            "name": t.get("tag"),
            "emoji": t.get("emoji"),
            "color": t.get("color"),
            "vote_score": t.get("vote_score"),
            "added_at": t.get("added_at"),
        }
        for t in (data.get("tags") or [])
    ]
    return {"tags": tags}, 200, None


def fetch_cape_leaderboard_lookup(
    username: str, scraper_session: Any, ctx: dict
) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    """
    Challenge-gated: this player's position on Laby's cape-count leaderboard.
    Uses the same shared LabyMod token as views/heart/game-stats — a 428
    means the caller should point the user at /api/minecraft/captcha.
    """
    name = (username or "").strip()
    if not name:
        return None, 400, "Missing username"

    data, code, err = laby_v3_path_get(
        f"textures/cape/leaderboard/lookup?name={name}&pageSize=50",
        scraper_session,
        ctx,
    )
    if code != 200 or not isinstance(data, dict):
        return None, code, err

    return data, 200, None
