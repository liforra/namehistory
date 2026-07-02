from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from providers.base import CosmeticsSnapshot, SourceSnapshot
from providers.laby_challenge import laby_v3_get


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        from dateutil import parser as dateutil_parser

        return dateutil_parser.isoparse(value).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _max_iso(a: Optional[str], b: Optional[str]) -> Optional[str]:
    da, db = _parse_ts(a), _parse_ts(b)
    if da and db:
        return (a if da >= db else b)
    return a or b


def _min_iso(a: Optional[str], b: Optional[str]) -> Optional[str]:
    da, db = _parse_ts(a), _parse_ts(b)
    if da and db:
        return (a if da <= db else b)
    return a or b


def fetch_labymod_profile(
    uuid: str, username: str, scraper_session: Any, ctx: Optional[dict] = None
) -> SourceSnapshot:
    source = "labymod"
    ctx = ctx or {}
    challenge_url = "/api/minecraft/captcha"
    try:
        names_r = scraper_session.get(
            f"https://laby.net/api/user/{uuid}/get-names",
            headers={"Accept": "application/json"},
            timeout=12,
        )
        tex_r = scraper_session.get(
            f"https://laby.net/api/user/{uuid}/get-textures",
            headers={"Accept": "application/json"},
            timeout=12,
        )
        if names_r.status_code != 200 and tex_r.status_code != 200:
            return SourceSnapshot(
                source=source,
                available=False,
                error=f"Laby API HTTP {names_r.status_code}/{tex_r.status_code}",
            )

        names: List[Dict[str, Any]] = (
            names_r.json() if names_r.status_code == 200 else []
        )
        textures: Dict[str, Any] = (
            tex_r.json() if tex_r.status_code == 200 else {}
        )

        last_online: Optional[str] = None
        join_date: Optional[str] = None
        monthly_views: Optional[int] = None
        last_server: Optional[str] = None
        likes: Optional[int] = None
        v3_notes: Dict[str, Any] = {}

        for entry in names:
            seen = entry.get("last_seen_at")
            last_online = _max_iso(last_online, seen)
            changed = entry.get("changed_at")
            join_date = _min_iso(join_date, changed)

        for skin in textures.get("SKIN", []):
            last_online = _max_iso(last_online, skin.get("last_seen_at"))
            join_date = _min_iso(join_date, skin.get("first_seen_at"))

        views_data, views_code, views_err = laby_v3_get(
            uuid, "views", scraper_session, ctx
        )
        if views_code == 200 and isinstance(views_data, dict):
            monthly_views = views_data.get("views")
            v3_notes["views"] = views_data
        elif views_err:
            v3_notes["views_error"] = views_err

        heart_data, heart_code, heart_err = laby_v3_get(
            uuid, "heart", scraper_session, ctx
        )
        if heart_code == 200 and isinstance(heart_data, dict):
            likes = heart_data.get("count")
            v3_notes["heart"] = heart_data
        elif heart_err:
            v3_notes["heart_error"] = heart_err

        stats_data, stats_code, stats_err = laby_v3_get(
            uuid, "game-stats", scraper_session, ctx
        )
        if stats_code == 200 and isinstance(stats_data, dict):
            join_date = _min_iso(join_date, stats_data.get("first_joined"))
            last_online = _max_iso(last_online, stats_data.get("last_online"))
            v3_notes["game_stats"] = stats_data
        elif stats_err and stats_code != 204:
            v3_notes["game_stats_error"] = stats_err

        online_data, online_code, online_err = laby_v3_get(
            uuid, "online-status", scraper_session, ctx
        )
        if online_code == 200 and isinstance(online_data, dict):
            v3_notes["online_status"] = online_data
            if online_data.get("online") and online_data.get("server"):
                server = online_data["server"]
                host = server.get("hostname")
                if host:
                    last_server = host
        elif online_err and online_code not in (204, 403):
            v3_notes["online_status_error"] = online_err

        current_name = username
        for entry in reversed(names):
            if entry.get("username"):
                current_name = entry["username"]
                break

        challenge_endpoints: List[str] = []
        if views_code == 428:
            challenge_endpoints.append("views")
        if heart_code == 428:
            challenge_endpoints.append("heart")
        if stats_code == 428:
            challenge_endpoints.append("game-stats")
        if online_code == 428:
            challenge_endpoints.append("online-status")

        challenge_needed = bool(challenge_endpoints)

        return SourceSnapshot(
            source=source,
            available=True,
            username=current_name,
            monthly_views=int(monthly_views) if monthly_views is not None else None,
            last_online=last_online,
            join_date=join_date,
            last_server=last_server,
            raw={
                "names": names,
                "textures": textures,
                "profile_url": f"https://laby.net/@{current_name}",
                "likes": likes,
                "views_per_month": monthly_views,
                "challenge_needed": challenge_needed,
                "challenge_url": challenge_url if challenge_needed else None,
                "challenge_endpoints": challenge_endpoints,
                "v3": v3_notes,
            },
        )
    except Exception as exc:
        return SourceSnapshot(source=source, available=False, error=str(exc))


def fetch_labymod_cosmetics(
    uuid: str, username: str, scraper_session: Any, ctx: Optional[dict] = None
) -> CosmeticsSnapshot:
    source = "labymod"
    try:
        tex_r = scraper_session.get(
            f"https://laby.net/api/user/{uuid}/get-textures",
            headers={"Accept": "application/json"},
            timeout=12,
        )
        if tex_r.status_code != 200:
            return CosmeticsSnapshot(
                source=source,
                available=False,
                error=f"Laby API HTTP {tex_r.status_code}",
            )
        textures = tex_r.json()
        cosmetics: List[Dict[str, Any]] = []
        for skin in textures.get("SKIN", []):
            cosmetics.append(
                {
                    "type": "skin",
                    "hash": skin.get("image_hash"),
                    "file_hash": skin.get("file_hash"),
                    "active": skin.get("active"),
                    "slim": skin.get("slim_skin"),
                    "first_seen_at": skin.get("first_seen_at"),
                    "last_seen_at": skin.get("last_seen_at"),
                }
            )
        for cape in textures.get("CAPE", []):
            cosmetics.append(
                {
                    "type": "cape",
                    "hash": cape.get("image_hash"),
                    "file_hash": cape.get("file_hash"),
                    "first_seen_at": cape.get("first_seen_at"),
                    "last_seen_at": cape.get("last_seen_at"),
                }
            )
        for cloak in textures.get("CLOAK", []):
            cosmetics.append(
                {
                    "type": "cloak",
                    "hash": cloak.get("image_hash"),
                    "file_hash": cloak.get("file_hash"),
                    "first_seen_at": cloak.get("first_seen_at"),
                    "last_seen_at": cloak.get("last_seen_at"),
                }
            )
        return CosmeticsSnapshot(
            source=source,
            available=True,
            cosmetics=cosmetics,
            raw=textures,
        )
    except Exception as exc:
        return CosmeticsSnapshot(source=source, available=False, error=str(exc))
