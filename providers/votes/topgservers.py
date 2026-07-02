from __future__ import annotations

from typing import Any, Dict, List

import requests

from providers.votes.base import ServerVoteResult, VoteSiteResult


def check_topgservers(
    username: str,
    servers: List[Dict[str, Any]],
) -> List[ServerVoteResult]:
    by_id: Dict[str, ServerVoteResult] = {}

    for entry in servers:
        api_key = str(entry.get("topgservers_key") or "").strip()
        address = str(entry.get("address") or "").strip()
        server_id = str(entry.get("id") or address)
        name = str(entry.get("name") or address)

        if server_id not in by_id:
            by_id[server_id] = ServerVoteResult(
                server_id=server_id, name=name, address=address
            )

        row = by_id[server_id]
        if not api_key:
            row.sites["topgservers"] = VoteSiteResult(
                site="topgservers",
                available=False,
                error="TopGServers API key not configured for this server",
            )
            continue

        try:
            r = requests.get(
                "https://topgservers.net/api/v1/vote-check",
                params={"username": username},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                    "User-Agent": "namehistory/1.0",
                },
                timeout=12,
            )
            if r.status_code == 401:
                row.sites["topgservers"] = VoteSiteResult(
                    site="topgservers",
                    available=False,
                    error="TopGServers API key invalid",
                )
                continue
            if r.status_code != 200:
                row.sites["topgservers"] = VoteSiteResult(
                    site="topgservers",
                    available=False,
                    error=f"TopGServers HTTP {r.status_code}",
                    raw={"body": r.text[:200]},
                )
                continue

            payload = r.json()
            voted = bool(payload.get("voted"))
            row.sites["topgservers"] = VoteSiteResult(
                site="topgservers",
                available=True,
                voted=voted,
                status="voted_today" if voted else "not_voted_today",
                raw=payload if isinstance(payload, dict) else {"response": payload},
            )
        except Exception as exc:
            row.sites["topgservers"] = VoteSiteResult(
                site="topgservers",
                available=False,
                error=str(exc),
            )

    return list(by_id.values())
