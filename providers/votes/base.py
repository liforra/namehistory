from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class VoteSiteResult:
    site: str
    available: bool = True
    error: Optional[str] = None
    voted: Optional[bool] = None
    liked: Optional[bool] = None
    status: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ServerVoteResult:
    server_id: str
    name: str
    address: str
    sites: Dict[str, VoteSiteResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server_id": self.server_id,
            "name": self.name,
            "address": self.address,
            "sites": {k: v.to_dict() for k, v in self.sites.items()},
            "any_voted": any(
                (s.voted is True or s.liked is True) for s in self.sites.values()
            ),
        }


@dataclass
class PlayerVotesSnapshot:
    username: str
    uuid: str
    servers: List[ServerVoteResult] = field(default_factory=list)
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        sites_checked = set()
        sites_ok: List[str] = []
        sites_skipped: List[str] = []
        for server in self.servers:
            for site, result in server.sites.items():
                sites_checked.add(site)
                if result.available and not result.error:
                    if site not in sites_ok:
                        sites_ok.append(site)
                elif result.error and "not configured" in (result.error or "").lower():
                    if site not in sites_skipped:
                        sites_skipped.append(site)

        voted_servers = [s.to_dict() for s in self.servers if s.to_dict().get("any_voted")]

        return {
            "username": self.username,
            "uuid": self.uuid,
            "note": self.note,
            "configured_servers": len(self.servers),
            "voted_or_liked_servers": voted_servers,
            "servers": [s.to_dict() for s in self.servers],
            "sites_ok": sorted(sites_ok),
            "sites_skipped": sorted(sites_skipped),
        }
