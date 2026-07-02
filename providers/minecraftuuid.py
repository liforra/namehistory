from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup

from providers.base import SourceSnapshot


def fetch_minecraftuuid_profile(
    uuid: str, username: str, scraper_session: Any
) -> SourceSnapshot:
    source = "minecraftuuid"
    url = f"https://minecraftuuid.com/player/{username}"
    try:
        r = scraper_session.get(url, timeout=15)
        if r.status_code != 200:
            return SourceSnapshot(
                source=source,
                available=False,
                error=f"minecraftuuid HTTP {r.status_code}",
            )

        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        monthly_views: Optional[int] = None
        views_match = re.search(r"([\d,]+)\s+profile views", text, re.I)
        if not views_match:
            views_match = re.search(r"([\d,]+)\s+views", text, re.I)
        if views_match:
            monthly_views = int(views_match.group(1).replace(",", ""))

        last_online: Optional[str] = None
        for pattern in [
            r"last\s+online[:\s]+([0-9T:\-+Z\s]+)",
            r"last\s+seen[:\s]+([0-9T:\-+Z\s]+)",
        ]:
            m = re.search(pattern, text, re.I)
            if m:
                last_online = m.group(1).strip()
                break

        join_date: Optional[str] = None
        join_match = re.search(
            r"(?:joined|registered|created)[:\s]+([0-9T:\-+Z\s]+)", text, re.I
        )
        if join_match:
            join_date = join_match.group(1).strip()

        return SourceSnapshot(
            source=source,
            available=True,
            username=username,
            monthly_views=monthly_views,
            last_online=last_online,
            join_date=join_date,
            raw={"profile_url": url, "uuid": uuid},
        )
    except Exception as exc:
        return SourceSnapshot(source=source, available=False, error=str(exc))
