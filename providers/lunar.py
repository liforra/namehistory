from __future__ import annotations

from typing import Any, List

from providers.base import CosmeticsSnapshot


def fetch_lunar_cosmetics(
    uuid: str, username: str, scraper_session: Any
) -> CosmeticsSnapshot:
    source = "lunar"
    endpoints = [
        f"https://api.lunarclientprod.com/game/cosmetics/user/{username}",
        f"https://api.lunarclientprod.com/game/profile/{username}",
        f"https://api.lunarclientprod.com/game/store/cosmetics/user/{username}",
    ]
    errors: List[str] = []
    for url in endpoints:
        try:
            r = scraper_session.get(
                url,
                headers={"Accept": "application/json"},
                timeout=12,
            )
            if r.status_code == 404:
                errors.append(f"{url}: 404")
                continue
            if r.status_code != 200:
                errors.append(f"{url}: HTTP {r.status_code}")
                continue
            payload = r.json()
            items = (
                payload.get("cosmetics")
                or payload.get("owned")
                or payload.get("items")
                or []
            )
            if isinstance(items, dict):
                items = list(items.values())
            return CosmeticsSnapshot(
                source=source,
                available=True,
                cosmetics=items if isinstance(items, list) else [items],
                raw=payload if isinstance(payload, dict) else {"data": payload},
            )
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    return CosmeticsSnapshot(
        source=source,
        available=False,
        error=(
            "Lunar Client cosmetics lookup is not publicly available. "
            f"Tried {len(endpoints)} endpoints. "
            + ("; ".join(errors[:3]) if errors else "")
        ),
    )
