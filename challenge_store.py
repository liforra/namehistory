from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, String


class ChallengeTokenStore:
    """Challenge token cache for provider APIs protected by edge captchas."""

    def __init__(self, default_ttl_minutes: int = 45):
        self._default_ttl = default_ttl_minutes
        self._lock = threading.Lock()
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._needed: Dict[str, bool] = {}

    def set(
        self,
        provider: str,
        token: str,
        ttl_minutes: Optional[int] = None,
        source: str = "manual",
    ) -> None:
        provider = provider.lower()
        ttl = ttl_minutes if ttl_minutes is not None else self._default_ttl
        expires = datetime.now(timezone.utc) + timedelta(minutes=ttl)
        with self._lock:
            self._tokens[provider] = {
                "token": token.strip(),
                "expires_at": expires,
                "source": source,
                "updated_at": datetime.now(timezone.utc),
            }
            self._needed[provider] = False

    def get(self, provider: str) -> Optional[str]:
        provider = provider.lower()
        with self._lock:
            entry = self._tokens.get(provider)
            if not entry:
                return None
            if entry["expires_at"] <= datetime.now(timezone.utc):
                self._tokens.pop(provider, None)
                return None
            return entry["token"]

    def is_valid(self, provider: str) -> bool:
        return bool(self.get(provider))

    def clear(self, provider: str) -> None:
        provider = provider.lower()
        with self._lock:
            self._tokens.pop(provider, None)

    def mark_needed(self, provider: str) -> None:
        with self._lock:
            self._needed[provider.lower()] = True

    def clear_needed(self, provider: str) -> None:
        with self._lock:
            self._needed[provider.lower()] = False

    def needs_challenge(self, provider: str) -> bool:
        if self.is_valid(provider):
            return False
        with self._lock:
            return self._needed.get(provider.lower(), False)

    def status(self, provider: str) -> Dict[str, Any]:
        provider = provider.lower()
        with self._lock:
            entry = self._tokens.get(provider)
            valid = bool(entry and entry["expires_at"] > datetime.now(timezone.utc))
            return {
                "provider": provider,
                "valid": valid,
                "needed": self._needed.get(provider, False) and not valid,
                "expires_at": entry["expires_at"].isoformat() if valid else None,
                "source": entry.get("source") if valid else None,
                "updated_at": entry.get("updated_at").isoformat()
                if valid and entry.get("updated_at")
                else None,
            }

    def all_status(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            providers = set(self._needed) | set(self._tokens)
        return {provider: self.status(provider) for provider in sorted(providers)}


def challenge_token_models(Base):
    class ChallengeToken(Base):
        __tablename__ = "challenge_tokens"
        provider = Column(String(32), primary_key=True)
        token = Column(String(2048), nullable=False)
        expires_at = Column(DateTime, nullable=False, index=True)
        source = Column(String(32), nullable=False, default="manual")
        updated_at = Column(DateTime, nullable=False)

    return ChallengeToken


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_tokens_from_db(session, ChallengeToken, store: ChallengeTokenStore) -> None:
    now = datetime.now(timezone.utc)
    for row in session.query(ChallengeToken).filter(ChallengeToken.expires_at > now.replace(tzinfo=None)).all():
        expires = _as_utc(row.expires_at)
        if expires <= now:
            continue
        store.set(
            row.provider,
            row.token,
            ttl_minutes=max(1, int((expires - now).total_seconds() // 60)),
            source=row.source or "database",
        )


def delete_token_from_db(session, ChallengeToken, provider: str) -> None:
    row = session.get(ChallengeToken, provider.lower())
    if row:
        session.delete(row)


def persist_token_to_db(
    session, ChallengeToken, store: ChallengeTokenStore, provider: str, source: str
) -> None:
    provider = provider.lower()
    token = store.get(provider)
    if not token:
        return
    status = store.status(provider)
    expires = _as_utc(datetime.fromisoformat(status["expires_at"]))
    row = session.get(ChallengeToken, provider)
    if not row:
        row = ChallengeToken(provider=provider, token=token)
        session.add(row)
    row.token = token
    row.expires_at = expires.replace(tzinfo=None)
    row.source = source
    row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
