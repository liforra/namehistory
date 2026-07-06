from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import abort, jsonify, request
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from providers.registry import (
    COSMETIC_PROVIDERS,
    PROFILE_PROVIDERS,
    fetch_all_profiles,
    fetch_cosmetics_source,
    fetch_profile_source,
)
from providers.base import CosmeticsSnapshot, SourceSnapshot
from providers.laby_textures import (
    VALID_TEXTURE_TYPES,
    fetch_cape_leaderboard_lookup,
    fetch_texture_meta,
    fetch_texture_tags,
    fetch_texture_users,
)


def client_profile_models(Base):
    class ClientProfileSnapshot(Base):
        __tablename__ = "client_profile_snapshots"
        id = Column(Integer, primary_key=True, autoincrement=True)
        uuid = Column(
            String(36),
            ForeignKey("profiles.uuid", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
        source = Column(String(32), nullable=False, index=True)
        playtime_seconds = Column(Integer)
        monthly_views = Column(Integer)
        last_online = Column(String(40))
        join_date = Column(String(40))
        last_server = Column(String(128))
        username = Column(String(32))
        payload = Column(Text)
        fetched_at = Column(DateTime, nullable=False, index=True)
        profile = relationship("Profile", back_populates="client_profiles")
        __table_args__ = (UniqueConstraint("uuid", "source"),)

    class ClientCosmeticSnapshot(Base):
        __tablename__ = "client_cosmetic_snapshots"
        id = Column(Integer, primary_key=True, autoincrement=True)
        uuid = Column(
            String(36),
            ForeignKey("profiles.uuid", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
        source = Column(String(32), nullable=False, index=True)
        payload = Column(Text, nullable=False)
        fetched_at = Column(DateTime, nullable=False, index=True)
        profile = relationship("Profile", back_populates="client_cosmetics")
        __table_args__ = (UniqueConstraint("uuid", "source"),)

    return ClientProfileSnapshot, ClientCosmeticSnapshot


def parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        from dateutil import parser as dateutil_parser

        return dateutil_parser.isoparse(ts).replace(tzinfo=timezone.utc)
    except Exception:
        return None


VIEW_SOURCE_PRIORITY = ("labymod", "badlion", "namemc")


def _monthly_views_from_sources(available: List[SourceSnapshot]) -> Optional[int]:
    by_source = {snap.source: snap for snap in available}
    for source in VIEW_SOURCE_PRIORITY:
        snap = by_source.get(source)
        if snap and snap.monthly_views is not None:
            return snap.monthly_views
    total = sum(snap.monthly_views or 0 for snap in available)
    return total or None


def _labymod_views_missing(snapshots: List[SourceSnapshot]) -> bool:
    for snap in snapshots:
        if snap.source == "labymod" and snap.available and snap.monthly_views is None:
            return True
    return False


def aggregate_profile_snapshots(
    snapshots: List[SourceSnapshot],
    *,
    challenge_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    available = [s for s in snapshots if s.available]
    est_playtime = sum(s.playtime_seconds or 0 for s in available)
    monthly_views = _monthly_views_from_sources(available)
    monthly_views_source = None
    if monthly_views is not None:
        by_source = {snap.source: snap for snap in available}
        for source in VIEW_SOURCE_PRIORITY:
            snap = by_source.get(source)
            if snap and snap.monthly_views == monthly_views:
                monthly_views_source = source
                break

    last_online_dt: Optional[datetime] = None
    last_online_source: Optional[str] = None
    for snap in available:
        dt = parse_timestamp(snap.last_online)
        if dt and (last_online_dt is None or dt > last_online_dt):
            last_online_dt = dt
            last_online_source = snap.source

    last_server = None
    last_server_source = None
    for snap in available:
        if snap.last_server:
            last_server = snap.last_server
            last_server_source = snap.source
            break

    sources: Dict[str, Any] = {}
    sources_ok: List[str] = []
    sources_failed: List[str] = []
    for snap in snapshots:
        sources[snap.source] = {
            "available": snap.available,
            "error": snap.error,
            "playtime_seconds": snap.playtime_seconds,
            "monthly_views": snap.monthly_views,
            "last_online": snap.last_online,
            "join_date": snap.join_date,
            "last_server": snap.last_server,
            "username": snap.username,
            "raw": snap.raw,
        }
        if snap.available:
            sources_ok.append(snap.source)
        else:
            sources_failed.append(snap.source)

    labymod_challenge_endpoints: List[str] = []
    labymod_challenge_needed = False
    for snap in snapshots:
        if snap.source == "labymod":
            raw = snap.raw or {}
            labymod_challenge_endpoints = list(raw.get("challenge_endpoints") or [])
            labymod_challenge_needed = bool(raw.get("challenge_needed"))
            break

    provider_challenges = dict(challenge_status or {})
    if labymod_challenge_endpoints or labymod_challenge_needed:
        labymod_status = dict(provider_challenges.get("labymod") or {})
        labymod_status["endpoints"] = labymod_challenge_endpoints
        labymod_status["needed"] = labymod_challenge_needed
        provider_challenges["labymod"] = labymod_status

    return {
        "est_playtime": est_playtime or None,
        "est_playtime_seconds": est_playtime or None,
        "monthly_views": monthly_views,
        "monthly_views_source": monthly_views_source,
        "est_last_online": last_online_dt.isoformat() if last_online_dt else None,
        "est_last_online_source": last_online_source,
        "last_server": last_server,
        "last_server_source": last_server_source,
        "sources_ok": sorted(sources_ok),
        "sources_failed": sorted(sources_failed),
        "sources": sources,
        "provider_challenges": provider_challenges,
    }


def persist_profile_snapshots(
    session,
    ClientProfileSnapshot,
    uuid: str,
    snapshots: List[SourceSnapshot],
) -> None:
    now = datetime.now(timezone.utc)
    for snap in snapshots:
        row = (
            session.query(ClientProfileSnapshot)
            .filter_by(uuid=uuid, source=snap.source)
            .first()
        )
        if not row:
            row = ClientProfileSnapshot(uuid=uuid, source=snap.source)
            session.add(row)
        row.playtime_seconds = snap.playtime_seconds
        row.monthly_views = snap.monthly_views
        row.last_online = snap.last_online
        row.join_date = snap.join_date
        row.last_server = snap.last_server
        row.username = snap.username
        row.payload = json.dumps(snap.to_dict(), ensure_ascii=False)
        row.fetched_at = now


def persist_cosmetics_snapshot(
    session,
    ClientCosmeticSnapshot,
    uuid: str,
    snap: CosmeticsSnapshot,
) -> None:
    now = datetime.now(timezone.utc)
    row = (
        session.query(ClientCosmeticSnapshot)
        .filter_by(uuid=uuid, source=snap.source)
        .first()
    )
    if not row:
        row = ClientCosmeticSnapshot(uuid=uuid, source=snap.source)
        session.add(row)
    row.payload = json.dumps(snap.to_dict(), ensure_ascii=False)
    row.fetched_at = now


def load_cached_profile_snapshots(
    session, ClientProfileSnapshot, uuid: str
) -> List[SourceSnapshot]:
    rows = session.query(ClientProfileSnapshot).filter_by(uuid=uuid).all()
    out: List[SourceSnapshot] = []
    for row in rows:
        data = {}
        if row.payload:
            try:
                data = json.loads(row.payload)
            except json.JSONDecodeError:
                data = {}
        if data:
            out.append(
                SourceSnapshot(
                    source=data.get("source", row.source),
                    available=bool(data.get("available", True)),
                    error=data.get("error"),
                    playtime_seconds=data.get("playtime_seconds", row.playtime_seconds),
                    monthly_views=data.get("monthly_views", row.monthly_views),
                    last_online=data.get("last_online", row.last_online),
                    join_date=data.get("join_date", row.join_date),
                    last_server=data.get("last_server", row.last_server),
                    username=data.get("username", row.username),
                    raw=data.get("raw", {}),
                )
            )
            continue
        out.append(
            SourceSnapshot(
                source=row.source,
                available=True,
                playtime_seconds=row.playtime_seconds,
                monthly_views=row.monthly_views,
                last_online=row.last_online,
                join_date=row.join_date,
                last_server=row.last_server,
                username=row.username,
                raw={},
            )
        )
    return out


def register_minecraft_routes(app, deps: Dict[str, Any]) -> None:
    ClientProfileSnapshot = deps["ClientProfileSnapshot"]
    ClientCosmeticSnapshot = deps["ClientCosmeticSnapshot"]
    tx = deps["tx"]
    ensure_profile = deps["ensure_profile"]
    is_source_stale = deps["is_source_stale"]
    update_source_timestamp = deps["update_source_timestamp"]
    resolve_player = deps["resolve_player"]
    log_request = deps["log_request"]
    query_history_public = deps["query_history_public"]
    query_skin_history_public = deps["query_skin_history_public"]
    _update_profile_from_sources = deps["_update_profile_from_sources"]
    _update_skin_profile_from_sources = deps["_update_skin_profile_from_sources"]
    SCRAPER_STALE_HOURS = deps["SCRAPER_STALE_HOURS"]
    provider_ctx = deps["provider_ctx"]
    jitter_sleep = deps.get("jitter_sleep", lambda: None)

    PROFILE_SUBSOURCES = set(PROFILE_PROVIDERS.keys())
    COSMETIC_SUBSOURCES = set(COSMETIC_PROVIDERS.keys())

    def _log_hit(endpoint: str, requested_username: str, status: int = 200) -> None:
        ip_address = (
            request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
            .split(",")[0]
            .strip()
        )
        req_name = request.args.get("req_name")
        requester_username = req_name.strip() if req_name and req_name.strip() else None
        try:
            with tx() as s:
                log_request(
                    session=s,
                    ip_address=ip_address,
                    requester_username=requester_username,
                    requested_username=requested_username,
                    source=request.args.get("source"),
                    version=request.args.get("version"),
                    endpoint=endpoint,
                    response_status=status,
                    mc_version=request.args.get("mc_version"),
                )
        except Exception as exc:
            deps["log"].error(f"Failed to log request: {exc}")

    def _challenge_status() -> Dict[str, Any]:
        store = provider_ctx.get("challenge_store")
        if not store:
            return {}
        return {"labymod": store.status("labymod")}

    def _fetch_or_cache_profiles(
        uuid: str, username: str, force: bool = False
    ) -> List[SourceSnapshot]:
        sync = provider_ctx.get("sync_challenge_tokens")
        if sync:
            sync()

        stale = True
        store = provider_ctx.get("challenge_store")
        with tx() as s:
            stale = force or is_source_stale(
                s, uuid, "client_profiles", SCRAPER_STALE_HOURS
            )
            if not stale:
                cached = load_cached_profile_snapshots(s, ClientProfileSnapshot, uuid)
                if cached:
                    if not (
                        store
                        and store.is_valid("labymod")
                        and _labymod_views_missing(cached)
                    ):
                        return cached

        snapshots = fetch_all_profiles(uuid, username, provider_ctx)
        with tx() as s:
            ensure_profile(s, uuid, username)
            persist_profile_snapshots(s, ClientProfileSnapshot, uuid, snapshots)
            update_source_timestamp(s, uuid, "client_profiles")
        for _ in snapshots:
            jitter_sleep()
        return snapshots

    def _player_namehistory(uuid: str, username: str) -> Dict[str, Any]:
        current_name = deps["get_profile_by_uuid_from_mojang"](uuid) or username
        with tx() as s:
            if not is_source_stale(s, uuid, "scraper", SCRAPER_STALE_HOURS):
                return query_history_public(s, uuid)
        return _update_profile_from_sources(uuid, current_name)

    def _player_skinhistory(
        uuid: str, username: str, *, force: bool = False
    ) -> Dict[str, Any]:
        current_name = deps["get_profile_by_uuid_from_mojang"](uuid) or username
        if not force:
            with tx() as s:
                if not is_source_stale(s, uuid, "skin_scraper", SCRAPER_STALE_HOURS):
                    return query_skin_history_public(s, uuid)
        return _update_skin_profile_from_sources(uuid, current_name)

    @app.route("/api/minecraft/<entity>", methods=["GET"])
    @app.route("/api/minecraft/<entity>/", methods=["GET"])
    def minecraft_entity_docs(entity: str):
        if entity == "texture":
            return jsonify(
                {
                    "entity": entity,
                    "texture_types": sorted(VALID_TEXTURE_TYPES),
                    "actions": ["<type>/<hash>/meta", "<type>/<hash>/users", "<type>/<hash>/tags"],
                    "example": "/api/minecraft/texture/cape/5b37a01fde6a3e075f3bc5694c18e667/meta",
                }
            )
        if entity != "player":
            abort(404, description=f"Unsupported entity type: {entity}")
        return jsonify(
            {
                "entity": entity,
                "actions": [
                    "profile",
                    "namehistory",
                    "skinhistory",
                    "update",
                    "votes",
                    "cape-rank",
                    "profile/<source>",
                    "cosmetics/<source>",
                    "votes/<site>",
                ],
                "profile_sources": sorted(PROFILE_SUBSOURCES),
                "cosmetic_sources": sorted(COSMETIC_SUBSOURCES),
                "legacy": {
                    "namehistory": "/api/namehistory?username=<name>",
                    "skinhistory": "/api/skinhistory?username=<name>",
                },
                "example": "/api/minecraft/player/profile?username=Notch",
            }
        )

    @app.route("/api/minecraft/<entity>/<path:action>", methods=["GET", "POST"])
    def minecraft_api(entity: str, action: str):
        if entity == "texture":
            parts = action.split("/")
            if len(parts) != 3:
                abort(404, description="Expected /api/minecraft/texture/<type>/<hash>/<meta|users|tags>")
            ttype, texture_hash, sub = parts
            if ttype.lower() not in VALID_TEXTURE_TYPES:
                abort(404, description=f"Unknown texture type: {ttype}")
            fetchers = {
                "meta": fetch_texture_meta,
                "users": fetch_texture_users,
                "tags": fetch_texture_tags,
            }
            fetcher = fetchers.get(sub)
            if not fetcher:
                abort(404, description=f"Unknown texture action: {sub}")
            data, status, err = fetcher(
                texture_hash, ttype, provider_ctx.get("scraper_session")
            )
            if data is None:
                return jsonify({"error": err or "Unknown error"}), status if status else 502
            return jsonify(
                {"type": ttype.lower(), "hash": texture_hash.lower(), **data}
            )

        if entity != "player":
            abort(404, description=f"Unsupported entity type: {entity}")

        username = request.args.get("username", "").strip()
        uuid_param = request.args.get("uuid", "").strip()
        force = request.args.get("refresh", "").lower() in {"1", "true", "yes"}

        if action in {"", "docs"} and not username and not uuid_param:
            return jsonify(
                {
                    "entity": entity,
                    "actions": [
                        "profile",
                        "namehistory",
                        "skinhistory",
                        "votes",
                        "profile/<source>",
                        "cosmetics/<source>",
                        "votes/<site>",
                    ],
                    "profile_sources": sorted(PROFILE_SUBSOURCES),
                    "cosmetic_sources": sorted(COSMETIC_SUBSOURCES),
                    "example": "/api/minecraft/player/profile?username=Notch",
                }
            )

        uuid, username, current_name = resolve_player(username, uuid_param)
        if not uuid:
            abort(404, "Profile not found")

        display_name = current_name or username

        if action == "namehistory":
            data = _player_namehistory(uuid, display_name)
            _log_hit(request.endpoint or action, username or uuid)
            return jsonify(data)

        if action == "skinhistory":
            data = _player_skinhistory(uuid, display_name, force=force)
            _log_hit(request.endpoint or action, username or uuid)
            return jsonify(data)

        if action == "cape-rank":
            data, status, err = fetch_cape_leaderboard_lookup(
                display_name, provider_ctx.get("scraper_session"), provider_ctx
            )
            _log_hit(request.endpoint or action, username or uuid)
            if data is None:
                return jsonify(
                    {
                        "query": display_name,
                        "uuid": uuid,
                        "available": False,
                        "error": err or "Unknown error",
                        "needs_challenge": status == 428,
                    }
                )
            return jsonify(
                {"query": display_name, "uuid": uuid, "available": True, **data}
            )

        if action == "update":
            if request.method != "POST":
                abort(405, "POST required")
            checker = deps.get("check_api_auth")
            if checker:
                checker()
            snapshots = _fetch_or_cache_profiles(uuid, display_name, force=True)
            _player_namehistory(uuid, display_name)
            _player_skinhistory(uuid, display_name, force=True)
            aggregated = aggregate_profile_snapshots(
                snapshots, challenge_status=_challenge_status()
            )
            _log_hit(request.endpoint or action, username or uuid)
            return jsonify(
                {
                    "message": "Profile refreshed",
                    "query": display_name,
                    "uuid": uuid,
                    "est_playtime": aggregated["est_playtime"],
                    "monthly_views": aggregated["monthly_views"],
                    "monthly_views_source": aggregated["monthly_views_source"],
                    "est_last_online": aggregated["est_last_online"],
                    "provider_challenges": aggregated["provider_challenges"],
                    "sources_refreshed": sorted({s.source for s in snapshots}),
                }
            )

        if action == "profile":
            snapshots = _fetch_or_cache_profiles(uuid, display_name, force=force)
            aggregated = aggregate_profile_snapshots(
                snapshots, challenge_status=_challenge_status()
            )
            response = {
                "query": display_name,
                "uuid": uuid,
                "est_playtime": aggregated["est_playtime"],
                "est_playtime_seconds": aggregated["est_playtime_seconds"],
                "monthly_views": aggregated["monthly_views"],
                "monthly_views_source": aggregated["monthly_views_source"],
                "est_last_online": aggregated["est_last_online"],
                "est_last_online_source": aggregated["est_last_online_source"],
                "last_server": aggregated["last_server"],
                "last_server_source": aggregated["last_server_source"],
                "sources_ok": aggregated["sources_ok"],
                "sources_failed": aggregated["sources_failed"],
                "sources": aggregated["sources"],
                "provider_challenges": aggregated["provider_challenges"],
            }
            _log_hit(request.endpoint or action, username or uuid)
            return jsonify(response)

        if action.startswith("profile/"):
            source = action.split("/", 1)[1].lower()
            if source not in PROFILE_SUBSOURCES and source not in {
                "laby",
                "labynet",
                "mcuuid",
            }:
                abort(404, f"Unknown profile source: {source}")
            snap = fetch_profile_source(source, uuid, display_name, provider_ctx)
            with tx() as s:
                ensure_profile(s, uuid, display_name)
                persist_profile_snapshots(s, ClientProfileSnapshot, uuid, [snap])
                update_source_timestamp(s, uuid, f"profile:{snap.source}")
            _log_hit(request.endpoint or action, username or uuid)
            return jsonify(
                {
                    "query": display_name,
                    "uuid": uuid,
                    "source": snap.source,
                    "profile": snap.to_dict(),
                }
            )

        if action.startswith("cosmetics/"):
            source = action.split("/", 1)[1].lower()
            if source not in COSMETIC_SUBSOURCES:
                abort(404, f"Unknown cosmetics source: {source}")
            snap = fetch_cosmetics_source(source, uuid, display_name, provider_ctx)
            with tx() as s:
                ensure_profile(s, uuid, display_name)
                persist_cosmetics_snapshot(s, ClientCosmeticSnapshot, uuid, snap)
                update_source_timestamp(s, uuid, f"cosmetics:{snap.source}")
            _log_hit(request.endpoint or action, username or uuid)
            return jsonify(
                {
                    "query": display_name,
                    "uuid": uuid,
                    "source": snap.source,
                    "cosmetics": snap.to_dict(),
                }
            )

        abort(404, f"Unknown action: {action}")
