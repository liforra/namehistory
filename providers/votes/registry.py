from __future__ import annotations

from typing import Any, Dict, List, Optional

from providers.votes.base import PlayerVotesSnapshot, ServerVoteResult
from providers.votes.claim_api import CLAIM_API_SITES, check_all_claim_api_sites
from providers.votes.namemc import check_namemc_likes
from providers.votes.topgservers import check_topgservers

VOTE_SITES = {
    "namemc": {
        "name": "NameMC",
        "description": "Server likes (not daily vote sites)",
        "needs_server_key": False,
    },
    **{
        site_id: {
            "name": cfg["name"],
            "description": "Daily vote check (last ~24h)",
            "needs_server_key": True,
        }
        for site_id, cfg in CLAIM_API_SITES.items()
    },
    "topgservers": {
        "name": "TopGServers",
        "description": "Daily vote check (today)",
        "needs_server_key": True,
    },
}

VOTE_SITE_ALIASES = {
    "namemc": "namemc",
    "minecraft-mp": "minecraft_mp",
    "minecraft_mp": "minecraft_mp",
    "mcmp": "minecraft_mp",
    "minecraftpocket": "minecraftpocket_servers",
    "minecraftpocket_servers": "minecraftpocket_servers",
    "mps": "minecraftpocket_servers",
    "trustmyserver": "trustmyserver",
    "tms": "trustmyserver",
    "topgservers": "topgservers",
    "topg": "topgservers",
}


def _merge_server_results(
    buckets: List[List[ServerVoteResult]],
) -> List[ServerVoteResult]:
    merged: Dict[str, ServerVoteResult] = {}
    for bucket in buckets:
        for row in bucket:
            existing = merged.get(row.server_id)
            if not existing:
                merged[row.server_id] = ServerVoteResult(
                    server_id=row.server_id,
                    name=row.name,
                    address=row.address,
                    sites=dict(row.sites),
                )
                continue
            existing.sites.update(row.sites)
    return list(merged.values())


def _configured_servers(votes_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    servers = votes_config.get("servers") or []
    if not isinstance(servers, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        address = str(entry.get("address") or "").strip()
        if not address:
            continue
        out.append(entry)
    return out


def fetch_player_votes(
    uuid: str,
    username: str,
    votes_config: Dict[str, Any],
) -> PlayerVotesSnapshot:
    servers = _configured_servers(votes_config)
    if not servers:
        return PlayerVotesSnapshot(
            username=username,
            uuid=uuid,
            servers=[],
            note=(
                "No vote servers configured. Add [[votes.servers]] entries in "
                "config.toml — vote sites only support per-server checks, not "
                "a global list of everywhere a player voted."
            ),
        )

    enabled = votes_config.get("sites") or {}
    buckets: List[List[ServerVoteResult]] = []

    if enabled.get("namemc", True):
        buckets.append(check_namemc_likes(uuid, username, servers))

    claim_results = check_all_claim_api_sites(username, servers)
    for site_id, rows in claim_results.items():
        if enabled.get(site_id, True):
            buckets.append(rows)

    if enabled.get("topgservers", True):
        buckets.append(check_topgservers(username, servers))

    merged = _merge_server_results(buckets)
    return PlayerVotesSnapshot(
        username=username,
        uuid=uuid,
        servers=merged,
        note=(
            "Vote sites do not publish a full history per player. This checks "
            "each configured server on each supported listing site."
        ),
    )


def fetch_votes_for_site(
    site: str,
    uuid: str,
    username: str,
    votes_config: Dict[str, Any],
) -> PlayerVotesSnapshot:
    canonical = VOTE_SITE_ALIASES.get(site.lower(), site.lower())
    if canonical not in VOTE_SITES:
        return PlayerVotesSnapshot(
            username=username,
            uuid=uuid,
            servers=[],
            note=f"Unknown vote site: {site}",
        )

    servers = _configured_servers(votes_config)
    if not servers:
        return PlayerVotesSnapshot(
            username=username,
            uuid=uuid,
            servers=[],
            note="No vote servers configured in config.toml",
        )

    buckets: List[List[ServerVoteResult]] = []
    if canonical == "namemc":
        buckets.append(check_namemc_likes(uuid, username, servers))
    elif canonical in CLAIM_API_SITES:
        from providers.votes.claim_api import check_claim_api_site

        buckets.append(
            check_claim_api_site(
                canonical, CLAIM_API_SITES[canonical], username, servers
            )
        )
    elif canonical == "topgservers":
        buckets.append(check_topgservers(username, servers))

    return PlayerVotesSnapshot(
        username=username,
        uuid=uuid,
        servers=_merge_server_results(buckets),
        note=None,
    )
