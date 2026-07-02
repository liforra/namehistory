from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SourceSnapshot:
    source: str
    available: bool = True
    error: Optional[str] = None
    playtime_seconds: Optional[int] = None
    monthly_views: Optional[int] = None
    last_online: Optional[str] = None
    join_date: Optional[str] = None
    last_server: Optional[str] = None
    username: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class CosmeticsSnapshot:
    source: str
    available: bool = True
    error: Optional[str] = None
    cosmetics: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
