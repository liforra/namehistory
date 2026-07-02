from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup

from providers.base import SourceSnapshot


def fetch_namemc_profile_stats(
    uuid: str, username: str, fetch_profile_html: Any
) -> SourceSnapshot:
    source = "namemc"
    url = f"https://namemc.com/profile/{username}"
    try:
        html = fetch_profile_html(f"https://namemc.com/profile/{username}")
        if not html:
            return SourceSnapshot(
                source=source,
                available=False,
                error="NameMC profile unavailable (blocked or not cached)",
            )

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)

        monthly_views: Optional[int] = None
        for pattern in [
            r"([\d,]+)\s+profile views",
            r"([\d,]+)\s+views\s*/\s*month",
            r"([\d,]+)\s+views",
        ]:
            m = re.search(pattern, text, re.I)
            if m:
                monthly_views = int(m.group(1).replace(",", ""))
                break

        return SourceSnapshot(
            source=source,
            available=True,
            username=username,
            monthly_views=monthly_views,
            raw={"profile_url": url},
        )
    except Exception as exc:
        return SourceSnapshot(source=source, available=False, error=str(exc))
