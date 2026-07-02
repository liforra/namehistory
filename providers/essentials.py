from __future__ import annotations

from typing import Any, List

from providers.base import CosmeticsSnapshot


def fetch_essentials_cosmetics(
    uuid: str, username: str, scraper_session: Any
) -> CosmeticsSnapshot:
    source = "essentials"
    endpoints = [
        f"https://api.essential.gg/v1/cosmetics/user/{username}",
        f"https://api.essential.gg/v1/wardrobe/{username}",
        f"https://api.essential.gg/v1/users/by-minecraft-username/{username}/cosmetics",
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
                or payload.get("items")
                or payload.get("wardrobe")
                or payload
            )
            if isinstance(items, dict):
                items = items.get("cosmetics") or items.get("items") or []
            if not isinstance(items, list):
                items = [items] if items else []
            return CosmeticsSnapshot(
                source=source,
                available=True,
                cosmetics=items,
                raw=payload if isinstance(payload, dict) else {"data": payload},
            )
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    return CosmeticsSnapshot(
        source=source,
        available=False,
        error=(
            "Essential.gg cosmetics are not exposed via a public API. "
            f"Tried {len(endpoints)} endpoints. "
            + ("; ".join(errors[:3]) if errors else "")
        ),
    )
