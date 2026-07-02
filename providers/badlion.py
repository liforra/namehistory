from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from bs4 import BeautifulSoup
from providers.http_client import fetch_with_fallback

from providers.base import CosmeticsSnapshot, SourceSnapshot


def _parse_next_data(html: str) -> dict:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _ms_to_iso(value: Any) -> Optional[str]:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def _ms_playtime_to_seconds(value: Any) -> Optional[int]:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    # Badlion reports overall_playtime in milliseconds.
    return int(value / 1000)


def _parse_badlion_page_props(page_props: dict) -> dict:
    card = page_props.get("card") or {}
    overview = page_props.get("overview") or {}

    last_online_ms = card.get("last_time_online") or card.get("last_online")
    last_server = (card.get("last_server_online") or "").strip() or None

    return {
        "username": card.get("username") or page_props.get("username"),
        "playtime_seconds": _ms_playtime_to_seconds(card.get("overall_playtime")),
        "monthly_views": overview.get("profile_views_30_days"),
        "views_all_time": overview.get("profile_views_all_time"),
        "last_online": _ms_to_iso(last_online_ms),
        "join_date": _ms_to_iso(card.get("joined_date")),
        "last_server": last_server,
        "rank": card.get("rank"),
        "badges": card.get("badges") or [],
        "skins": card.get("skins") or [],
        "names": overview.get("names") or [],
    }


def fetch_badlion_profile(
    uuid: str, username: str, scraper_session: Any
) -> SourceSnapshot:
    source = "badlion"
    url = f"https://www.badlion.net/profile/minecraft/{username}/"
    try:
        r = fetch_with_fallback(url, session=scraper_session, timeout=15, allow=[200])
        if not r or r.status_code != 200:
            return SourceSnapshot(
                source=source,
                available=False,
                error=f"Badlion HTTP {r.status_code} (may be blocked by Cloudflare)",
            )

        page_props = _parse_next_data(r.text).get("props", {}).get("pageProps", {})
        card = page_props.get("card") or {}
        if (
            card.get("found")
            or card.get("success")
            or card.get("username")
            or card.get("overall_playtime")
        ):
            parsed = _parse_badlion_page_props(page_props)
            return SourceSnapshot(
                source=source,
                available=True,
                username=parsed.get("username") or username,
                playtime_seconds=parsed.get("playtime_seconds"),
                monthly_views=parsed.get("monthly_views"),
                last_online=parsed.get("last_online"),
                join_date=parsed.get("join_date"),
                last_server=parsed.get("last_server"),
                raw={
                    "profile_url": url,
                    "rank": parsed.get("rank"),
                    "badges": parsed.get("badges"),
                    "views_all_time": parsed.get("views_all_time"),
                    "names": parsed.get("names"),
                    "skins": parsed.get("skins"),
                },
            )

        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        monthly_views: Optional[int] = None
        views_match = re.search(r"([\d,]+)\s+views", text, re.I)
        if views_match:
            monthly_views = int(views_match.group(1).replace(",", ""))

        return SourceSnapshot(
            source=source,
            available=True,
            username=username,
            monthly_views=monthly_views,
            raw={"profile_url": url, "note": "Limited parse; page may be client-rendered"},
        )
    except Exception as exc:
        return SourceSnapshot(source=source, available=False, error=str(exc))


def fetch_badlion_cosmetics(
    uuid: str, username: str, scraper_session: Any
) -> CosmeticsSnapshot:
    source = "badlion"
    url = f"https://www.badlion.net/profile/minecraft/{username}/"
    try:
        r = fetch_with_fallback(url, session=scraper_session, timeout=15, allow=[200])
        if not r or r.status_code != 200:
            return CosmeticsSnapshot(
                source=source,
                available=False,
                error=f"Badlion HTTP {r.status_code if r else 'no response'}",
            )
        page_props = _parse_next_data(r.text).get("props", {}).get("pageProps", {})
        parsed = _parse_badlion_page_props(page_props)
        skins = parsed.get("skins") or []
        cosmetics = page_props.get("cosmetics") or page_props.get("ownedCosmetics") or []
        items = cosmetics if isinstance(cosmetics, list) else []
        if skins:
            items = items + [
                {
                    "type": "skin",
                    "name": s.get("name"),
                    "skin_hash": s.get("skin_hash"),
                    "skin_url": s.get("skin_url"),
                    "hidden": s.get("hidden"),
                }
                for s in skins
                if isinstance(s, dict)
            ]
        if items:
            return CosmeticsSnapshot(
                source=source,
                available=True,
                cosmetics=items,
                raw={"skins": skins, "cosmetics": cosmetics},
            )
        return CosmeticsSnapshot(
            source=source,
            available=False,
            error="No cosmetics data found on Badlion profile",
        )
    except Exception as exc:
        return CosmeticsSnapshot(source=source, available=False, error=str(exc))
