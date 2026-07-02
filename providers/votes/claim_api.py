from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import requests

from providers.votes.base import ServerVoteResult, VoteSiteResult

# Sites using ?object=votes&element=claim&key=...&username=...
# Response: 0 = not voted (24h), 1 = voted unclaimed, 2 = voted claimed
CLAIM_API_SITES = {
    "minecraft_mp": {
        "name": "Minecraft-MP",
        "base_url": "https://minecraft-mp.com/api/",
        "config_key": "minecraft_mp_key",
    },
    "minecraftpocket_servers": {
        "name": "Minecraft Pocket Servers",
        "base_url": "https://minecraftpocket-servers.com/api/",
        "config_key": "minecraftpocket_servers_key",
    },
    "trustmyserver": {
        "name": "TrustMyServer",
        "base_url": "https://trustmyserver.com/api/",
        "config_key": "trustmyserver_key",
    },
}

_STATUS_MAP = {
    "0": "not_voted",
    "1": "voted_unclaimed",
    "2": "voted_claimed",
}


def _parse_claim_response(text: str) -> Tuple[Optional[bool], str]:
    raw = (text or "").strip()
    if raw in _STATUS_MAP:
        status = _STATUS_MAP[raw]
        return raw != "0", status
    if raw.isdigit():
        return raw != "0", _STATUS_MAP.get(raw, raw)
    return None, raw[:80] or "unknown"


def check_claim_api_site(
    site_id: str,
    site_cfg: Dict[str, Any],
    username: str,
    servers: List[Dict[str, Any]],
) -> List[ServerVoteResult]:
    config_key = site_cfg["config_key"]
    base_url = site_cfg["base_url"]
    by_id: Dict[str, ServerVoteResult] = {}

    for entry in servers:
        api_key = str(entry.get(config_key) or "").strip()
        address = str(entry.get("address") or "").strip()
        server_id = str(entry.get("id") or address)
        name = str(entry.get("name") or address)

        if server_id not in by_id:
            by_id[server_id] = ServerVoteResult(
                server_id=server_id, name=name, address=address
            )

        row = by_id[server_id]
        if not api_key:
            row.sites[site_id] = VoteSiteResult(
                site=site_id,
                available=False,
                error=f"{site_cfg['name']} API key not configured for this server",
            )
            continue

        try:
            r = requests.get(
                base_url,
                params={
                    "object": "votes",
                    "element": "claim",
                    "key": api_key,
                    "username": username,
                },
                timeout=12,
                headers={"User-Agent": "namehistory/1.0"},
            )
            if r.status_code != 200:
                row.sites[site_id] = VoteSiteResult(
                    site=site_id,
                    available=False,
                    error=f"{site_cfg['name']} HTTP {r.status_code}",
                    raw={"body": r.text[:200]},
                )
                continue

            voted, status = _parse_claim_response(r.text)
            row.sites[site_id] = VoteSiteResult(
                site=site_id,
                available=True,
                voted=voted,
                status=status,
                raw={"response": r.text.strip()},
            )
        except Exception as exc:
            row.sites[site_id] = VoteSiteResult(
                site=site_id,
                available=False,
                error=str(exc),
            )

    return list(by_id.values())


def check_all_claim_api_sites(
    username: str, servers: List[Dict[str, Any]]
) -> Dict[str, List[ServerVoteResult]]:
    out: Dict[str, List[ServerVoteResult]] = {}
    for site_id, site_cfg in CLAIM_API_SITES.items():
        out[site_id] = check_claim_api_site(site_id, site_cfg, username, servers)
    return out
