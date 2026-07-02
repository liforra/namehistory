from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup

from providers.base import SourceSnapshot


def fetch_mineplex_profile(
    uuid: str, username: str, scraper_session: Any
) -> SourceSnapshot:
    source = "mineplex"
    urls = [
        f"https://www.mineplex.com/player/{username}",
        f"https://statssrv.mineplex.com/player/{username}",
        f"https://mineplex.com/stats/player/{username}",
    ]
    errors = []
    for url in urls:
        try:
            r = scraper_session.get(url, timeout=15)
            if r.status_code != 200:
                errors.append(f"{url}: HTTP {r.status_code}")
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(" ", strip=True)

            playtime: Optional[int] = None
            play_match = re.search(
                r"play\s*time[:\s]+([\d,]+)\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)?",
                text,
                re.I,
            )
            if play_match:
                playtime = int(play_match.group(1).replace(",", ""))

            views: Optional[int] = None
            views_match = re.search(r"([\d,]+)\s+views", text, re.I)
            if views_match:
                views = int(views_match.group(1).replace(",", ""))

            last_online: Optional[str] = None
            online_match = re.search(
                r"last\s+(?:online|seen)[:\s]+([0-9T:\-+Z\s]+)", text, re.I
            )
            if online_match:
                last_online = online_match.group(1).strip()

            join_date: Optional[str] = None
            join_match = re.search(
                r"(?:joined|first\s+joined)[:\s]+([0-9T:\-+Z\s]+)", text, re.I
            )
            if join_match:
                join_date = join_match.group(1).strip()

            if playtime or views or last_online or join_date:
                return SourceSnapshot(
                    source=source,
                    available=True,
                    username=username,
                    playtime_seconds=playtime,
                    monthly_views=views,
                    last_online=last_online,
                    join_date=join_date,
                    last_server="mineplex.com",
                    raw={"profile_url": url},
                )
            errors.append(f"{url}: no stats parsed")
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    return SourceSnapshot(
        source=source,
        available=False,
        error=(
            "Mineplex player stats could not be retrieved from public pages. "
            + ("; ".join(errors[:3]) if errors else "")
        ),
    )
