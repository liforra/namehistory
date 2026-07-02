from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import requests

from providers.base import SourceSnapshot


def _ms_to_iso(value: Optional[int]) -> Optional[str]:
    if not value:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def _extract_playtime_seconds(stats: dict) -> Optional[int]:
    if not isinstance(stats, dict):
        return None
    total = 0
    found = False
    for game_stats in stats.values():
        if not isinstance(game_stats, dict):
            continue
        for key, val in game_stats.items():
            if key in {"playtime", "time_played", "total_playtime"} and isinstance(
                val, (int, float)
            ):
                total += int(val)
                found = True
    return total if found else None


def fetch_hypixel_profile(
    uuid: str, username: str, api_key: str, mojang_session: Any
) -> SourceSnapshot:
    source = "hypixel"
    if not api_key:
        return SourceSnapshot(
            source=source,
            available=False,
            error="Hypixel API key not configured. Set HYPIXEL_API_KEY in .env or providers.hypixel_api_key in config.toml",
        )
    try:
        r = requests.get(
            "https://api.hypixel.net/v2/player",
            params={"uuid": uuid.replace("-", "")},
            headers={"API-Key": api_key},
            timeout=12,
        )
        if r.status_code != 200:
            return SourceSnapshot(
                source=source,
                available=False,
                error=f"Hypixel API HTTP {r.status_code}",
            )
        payload = r.json()
        if not payload.get("success"):
            return SourceSnapshot(
                source=source,
                available=False,
                error=str(payload.get("cause", "Hypixel API error")),
            )
        player = payload.get("player")
        if not player:
            return SourceSnapshot(
                source=source,
                available=False,
                error="Player has never joined Hypixel",
            )

        stats = player.get("stats") or {}
        playtime = _extract_playtime_seconds(stats)

        last_online = (
            _ms_to_iso(player.get("lastLogin"))
            or _ms_to_iso(player.get("lastLogout"))
        )

        return SourceSnapshot(
            source=source,
            available=True,
            username=player.get("displayname") or username,
            playtime_seconds=playtime,
            last_online=last_online,
            join_date=_ms_to_iso(player.get("firstLogin")),
            last_server="hypixel.net",
            raw={
                "rank": player.get("rank"),
                "packageRank": player.get("newPackageRank"),
                "networkLevel": player.get("networkLevel"),
                "achievementPoints": player.get("achievementPoints"),
                "stats_games": list(stats.keys())[:20],
                "profile_url": f"https://plancke.io/hypixel/player/stats/{player.get('displayname', username)}",
            },
        )
    except Exception as exc:
        return SourceSnapshot(source=source, available=False, error=str(exc))
