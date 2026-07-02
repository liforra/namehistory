from __future__ import annotations

from typing import Any, Dict, List

import requests

from providers.votes.base import ServerVoteResult, VoteSiteResult


def check_namemc_likes(
    uuid: str,
    username: str,
    servers: List[Dict[str, Any]],
) -> List[ServerVoteResult]:
    results: List[ServerVoteResult] = []
    uuid_compact = uuid.replace("-", "")

    for entry in servers:
        if not entry.get("namemc", True):
            continue
        address = str(entry.get("address") or "").strip()
        if not address:
            continue

        server_id = str(entry.get("id") or address)
        row = ServerVoteResult(
            server_id=server_id,
            name=str(entry.get("name") or address),
            address=address,
        )

        try:
            r = requests.get(
                f"https://api.namemc.com/server/{address}/likes",
                params={"profile": uuid_compact},
                timeout=12,
                headers={"Accept": "application/json", "User-Agent": "namehistory/1.0"},
            )
            if r.status_code != 200:
                row.sites["namemc"] = VoteSiteResult(
                    site="namemc",
                    available=False,
                    error=f"NameMC HTTP {r.status_code}",
                )
            else:
                liked = r.json()
                if isinstance(liked, bool):
                    liked_val = liked
                elif isinstance(liked, dict):
                    liked_val = bool(liked.get("liked", liked.get("success")))
                else:
                    liked_val = bool(liked)

                row.sites["namemc"] = VoteSiteResult(
                    site="namemc",
                    available=True,
                    liked=liked_val,
                    voted=liked_val,
                    status="liked" if liked_val else "not_liked",
                    raw={"response": liked},
                )
        except Exception as exc:
            row.sites["namemc"] = VoteSiteResult(
                site="namemc",
                available=False,
                error=str(exc),
            )

        results.append(row)

    return results
