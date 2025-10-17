#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import random
import threading
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dateutil import parser as dateutil_parser

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
    relationship,
    scoped_session,
)
from curl_cffi.requests import Session
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, abort
from werkzeug.exceptions import HTTPException

try:
    import tomli as toml
except ImportError:
    toml = None

import logging
from logging.handlers import RotatingFileHandler

# --- CONFIGURATION ---
CONFIG: Dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 8000},
    "rate_limit": {"rpm": 15, "window_seconds": 60},
    # Optional auth: if unset, all endpoints remain open (backward compatible)
    # If auth.api_key is set, write-like actions require "Authorization: API-Key <value>"
    # If auth.admin_key is set, admin actions require "Authorization: Admin-Key <value>"
    # Admin-Key also satisfies API-Key checks if both are set.
    "auth": {"api_key": "", "admin_key": ""},
    "fetch": {
        "default_sleep": 0.5,
        "bulk_min_delay": 0.8,
        "bulk_max_delay": 1.5,
    },
    "database": {"type": "sqlite", "path": "namehistory.db"},
    "logging": {
        "level": "INFO",
        "json": False,
        "file": "",
        "max_bytes": 10_485_760,
        "backup_count": 3,
    },
    "auto_update": {
        "enabled": True,
        "check_interval_minutes": 60,
        "scraper_stale_hours": 24,
        "batch_size": 10,
    },
}


def load_config():
    global CONFIG
    path = "config.toml"
    if toml and os.path.exists(path):
        with open(path, "rb") as f:
            data = toml.load(f)
        for k, v in data.items():
            if isinstance(v, dict) and k in CONFIG:
                CONFIG[k].update(v)
            else:
                CONFIG[k] = v


load_config()

HOST = CONFIG["server"]["host"]
PORT = int(CONFIG["server"]["port"])
DB_TYPE = CONFIG["database"].get("type", "sqlite")
DB_PATH = CONFIG["database"].get("path", "namehistory.db")
DB_URL = CONFIG["database"].get("url", None)
DEFAULT_SLEEP = float(CONFIG["fetch"]["default_sleep"])
BULK_MIN_DELAY = float(CONFIG["fetch"]["bulk_min_delay"])
BULK_MAX_DELAY = float(CONFIG["fetch"]["bulk_max_delay"])
RATE_LIMIT_RPM = int(CONFIG["rate_limit"]["rpm"])
RATE_LIMIT_WINDOW = float(CONFIG["rate_limit"]["window_seconds"])
AUTO_UPDATE_ENABLED = bool(CONFIG["auto_update"]["enabled"])
AUTO_UPDATE_CHECK_INTERVAL = int(CONFIG["auto_update"]["check_interval_minutes"]) * 60
SCRAPER_STALE_HOURS = int(CONFIG["auto_update"]["scraper_stale_hours"])
AUTO_UPDATE_BATCH_SIZE = int(CONFIG["auto_update"]["batch_size"])
API_KEY = str(CONFIG.get("auth", {}).get("api_key", "") or "").strip()
ADMIN_KEY = str(CONFIG.get("auth", {}).get("admin_key", "") or "").strip()

# --- LOGGING SETUP ---
def setup_logging():
    lvl = str(CONFIG.get("logging", {}).get("level", "INFO")).upper()
    json_mode = bool(CONFIG.get("logging", {}).get("json", False))
    log_file = CONFIG.get("logging", {}).get("file") or ""
    max_bytes = int(CONFIG.get("logging", {}).get("max_bytes", 10_485_760))
    backup_count = int(CONFIG.get("logging", {}).get("backup_count", 3))

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "level": record.levelname,
                "ts": datetime.now(timezone.utc).isoformat(),
                "msg": record.getMessage(),
            }
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=False)

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    root.setLevel(getattr(logging, lvl, logging.INFO))
    fmt = (
        JsonFormatter()
        if json_mode
        else logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_file:
        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


setup_logging()
log = logging.getLogger("namehistory")

# --- AUTH HELPERS ---
class AuthLevel:
    NONE = 0
    API = 1
    ADMIN = 2


def _extract_auth_token() -> Tuple[str, str]:
    """
    Returns tuple (scheme, token) where scheme is one of "API-Key", "Admin-Key", or "".
    Accepted header formats:
      Authorization: API-Key <token>
      Authorization: Admin-Key <token>
    """
    hdr = request.headers.get("Authorization", "").strip()
    if not hdr:
        return "", ""
    try:
        scheme, token = hdr.split(None, 1)
        return scheme.strip(), token.strip()
    except ValueError:
        return "", ""


def require_auth(level: int):
    """
    Decorator to enforce optional auth.
    - If API key not configured and level is API: allow.
    - If Admin key not configured and level is ADMIN: allow.
    - If both configured, Admin-Key satisfies both ADMIN and API.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            api_needed = (level >= AuthLevel.API) and bool(API_KEY)
            admin_needed = (level >= AuthLevel.ADMIN) and bool(ADMIN_KEY)
            # If neither required due to missing config -> open access
            if not api_needed and not admin_needed:
                return fn(*args, **kwargs)

            scheme, token = _extract_auth_token()

            # Admin level required
            if admin_needed:
                if scheme == "Admin-Key" and token == ADMIN_KEY:
                    return fn(*args, **kwargs)
                abort(401, description="Admin authorization required")

            # API level required
            if api_needed:
                # Admin key also satisfies API requirement
                if (scheme == "API-Key" and token == API_KEY) or (
                    ADMIN_KEY and scheme == "Admin-Key" and token == ADMIN_KEY
                ):
                    return fn(*args, **kwargs)
                abort(401, description="API authorization required")

            return fn(*args, **kwargs)

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator


# --- HTTP & UTILS ---
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
]
scraper_session = Session(impersonate="chrome110")
mojang_session = requests.Session()
MCS_LOOKUP = (
    "https://api.minecraftservices.com/minecraft/profile/lookup/name/{name}"
)
MCS_PROFILE = (
    "https://sessionserver.mojang.com/session/minecraft/profile/{uuid}"
)


def jitter_sleep(base: float = DEFAULT_SLEEP):
    time.sleep(base + random.uniform(0.05, 0.2))


def bulk_sleep():
    time.sleep(random.uniform(BULK_MIN_DELAY, BULK_MAX_DELAY))


def dashed_uuid(raw: str) -> str:
    raw = raw.replace("-", "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{32}", raw):
        s = raw.lower()
        return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"
    return raw


def parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse a timestamp string to datetime object, return None if invalid or None."""
    if not ts:
        return None
    try:
        return dateutil_parser.isoparse(ts).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def normalize_timestamp(ts: Optional[str]) -> Optional[str]:
    """Normalize a timestamp to a consistent ISO format."""
    dt = parse_timestamp(ts)
    return dt.isoformat() if dt else None


# --- DATA FETCHING ---
def normalize_username(name: str) -> Optional[Tuple[str, str]]:
    try:
        r = mojang_session.get(MCS_LOOKUP.format(name=name), timeout=10.0)
        if r.status_code != 200:
            return None
        data = r.json()
        return (data.get("name"), dashed_uuid(data.get("id")))
    except Exception:
        return None


def get_profile_by_uuid_from_mojang(uuid: str) -> Optional[str]:
    try:
        r = mojang_session.get(
            MCS_PROFILE.format(uuid=uuid.replace("-", "")), timeout=10.0
        )
        return r.json().get("name") if r.status_code == 200 else None
    except Exception:
        return None


def fetch_profile_html(url: str) -> str:
    try:
        r = scraper_session.get(
            url, headers={"User-Agent": random.choice(UA_POOL)}, timeout=15.0
        )
        return r.text if r.status_code == 200 else ""
    except Exception:
        return ""


def fetch_namemc_data(username: str) -> List[Dict[str, Optional[str]]]:
    html = fetch_profile_html(f"https://namemc.com/profile/{username}")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")

    # Updated selector - try both old and new table formats
    table = soup.select_one("table.table-borderless") or soup.select_one(
        "table.table-striped"
    )
    if not table:
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    out = []
    for row in tbody.find_all("tr"):
        # Skip mobile-only duplicate rows
        if "d-lg-none" in row.get("class", []):
            continue

        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # Find name cell - look for <a> with /search?q= link
        name = None
        for cell in cells:
            link = cell.find("a", href=re.compile(r"/search\?q="))
            if link:
                name = link.get_text(strip=True)
                break

        # Find time cell
        changed_at = None
        for cell in cells:
            time_tag = cell.find("time")
            if time_tag and time_tag.get("datetime"):
                changed_at = time_tag["datetime"].strip()
                break

        if name:
            out.append(
                {"name": name, "changedAt": changed_at, "source": "namemc"}
            )

    return out


def fetch_laby_api_data(uuid: str) -> List[Dict[str, Optional[str]]]:
    url = f"https://laby.net/api/user/{uuid}/get-names"
    try:
        r = scraper_session.get(
            url, headers={"Accept": "application/json"}, timeout=10
        )
        if r.status_code != 200:
            return []
        return [
            {
                "name": e.get("name"),
                "changedAt": e.get("changed_at"),
                "source": "laby",
            }
            for e in r.json()
            if e.get("name")
        ]
    except Exception:
        return []


def gather_remote_rows_by_uuid(
    uuid: str, current_name: str
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch from BOTH Laby and NameMC, return all results.
    """
    all_rows = []

    # Fetch from Laby API
    log.debug(f"Fetching from Laby API for UUID {uuid}")
    laby_rows = fetch_laby_api_data(uuid)
    log.debug(f"Laby returned {len(laby_rows)} entries")
    all_rows.extend(laby_rows)
    jitter_sleep()

    # Fetch from NameMC
    log.debug(f"Fetching from NameMC for username {current_name}")
    namemc_rows = fetch_namemc_data(current_name)
    log.debug(f"NameMC returned {len(namemc_rows)} entries")
    all_rows.extend(namemc_rows)

    return all_rows


def merge_remote_sources(
    rows: List[Dict[str, Optional[str]]], current_name: str
) -> List[Tuple[str, Optional[str]]]:
    """
    Intelligently merge data from multiple sources.

    Key insights:
    1. You can't change from "Name" to "Name" - consecutive duplicates are the same event
    2. A name with null timestamp might not actually be the original - if we have a dated
       occurrence of the same name, the null is likely incomplete data
    3. Name reuse is detected when OTHER names appear in between
    """

    # Step 1: Normalize all timestamps
    normalized_entries = []
    for row in rows:
        name = row.get("name")
        changed_at = row.get("changedAt")
        source = row.get("source", "unknown")

        if name:
            normalized_ts = normalize_timestamp(changed_at)
            normalized_entries.append(
                {
                    "name": name,
                    "timestamp": normalized_ts,
                    "datetime": parse_timestamp(changed_at),
                    "source": source,
                }
            )

    log.debug(f"Processing {len(normalized_entries)} entries from all sources")

    # Step 2: Group by name to detect conflicting null vs dated timestamps
    by_name = defaultdict(list)
    for entry in normalized_entries:
        by_name[entry["name"].lower()].append(entry)

    # Step 3: For each name, if it has BOTH null and dated timestamps, remove the null
    # (the null is likely incomplete data)
    cleaned_entries = []
    for name_lower, entries in by_name.items():
        has_null = any(e["timestamp"] is None for e in entries)
        has_dated = any(e["timestamp"] is not None for e in entries)

        if has_null and has_dated:
            log.debug(
                f"  Name '{entries[0]['name']}' has both null and dated timestamps - discarding null"
            )
            cleaned_entries.extend([e for e in entries if e["timestamp"] is not None])
        else:
            cleaned_entries.extend(entries)

    log.debug(f"After removing incomplete nulls: {len(cleaned_entries)} entries")

    # Step 4: Sort chronologically (None first, then by datetime)
    cleaned_entries.sort(
        key=lambda e: e["datetime"]
        or datetime.min.replace(tzinfo=timezone.utc)
    )

    # Step 5: Remove consecutive duplicates (same name appearing multiple times in a row)
    merged = []
    for entry in cleaned_entries:
        name = entry["name"]
        timestamp = entry["timestamp"]
        source = entry["source"]

        if merged:
            last_name, last_timestamp = merged[-1]
            if name.lower() == last_name.lower():
                last_dt = parse_timestamp(last_timestamp)
                curr_dt = parse_timestamp(timestamp)

                if curr_dt and last_dt and curr_dt < last_dt:
                    log.debug(
                        f"  Replacing {last_name} @ {last_timestamp} with earlier {timestamp} (from {source})"
                    )
                    merged[-1] = (name, timestamp)
                else:
                    log.debug(
                        f"  Skipping consecutive duplicate {name} @ {timestamp} (from {source})"
                    )
                continue

        log.debug(f"  Adding {name} @ {timestamp} (from {source})")
        merged.append((name, timestamp))

    # Step 6: Ensure current name is in the list
    if merged:
        last_name, _ = merged[-1]
        if last_name.lower() != current_name.lower():
            log.debug(f"  Adding current name '{current_name}' (not in history)")
            merged.append((current_name, None))
    else:
        log.debug(f"  Adding current name '{current_name}' (empty history)")
        merged.append((current_name, None))

    log.info(f"Final merged result: {len(merged)} unique name changes")

    return merged


# --- DATABASE ---
Base = declarative_base()


class Profile(Base):
    __tablename__ = "profiles"
    uuid = Column(String(36), primary_key=True)
    query = Column(String(32))
    last_seen_at = Column(DateTime)
    history = relationship(
        "History", back_populates="profile", cascade="all, delete-orphan"
    )
    source_updates = relationship(
        "SourceUpdate",
        back_populates="profile",
        cascade="all, delete-orphan",
    )


class History(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(
        String(36),
        ForeignKey("profiles.uuid", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(32), nullable=False, index=True)
    changed_at = Column(String(32), index=True)
    observed_at = Column(DateTime, nullable=False)
    profile = relationship("Profile", back_populates="history")
    __table_args__ = (UniqueConstraint("uuid", "name", "changed_at"),)


class SourceUpdate(Base):
    __tablename__ = "source_updates"
    uuid = Column(
        String(36), ForeignKey("profiles.uuid", ondelete="CASCADE"), primary_key=True
    )
    source = Column(String(32), primary_key=True)
    last_updated_at = Column(DateTime, nullable=False, index=True)
    profile = relationship("Profile", back_populates="source_updates")


db_url = f"sqlite:///{DB_PATH}" if DB_TYPE == "sqlite" else DB_URL
engine = create_engine(db_url, echo=False, future=True)
SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False)
)


def init_db():
    Base.metadata.create_all(bind=engine)


@contextmanager
def tx():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --- CORE LOGIC ---
def ensure_profile(session, uuid: str, query: Optional[str]):
    profile = session.get(Profile, uuid)
    now = datetime.now(timezone.utc)
    if not profile:
        session.add(Profile(uuid=uuid, query=query, last_seen_at=now))
    else:
        profile.query = query
        profile.last_seen_at = now


def get_uuid_from_history_by_name(session, username: str) -> Optional[str]:
    row = (
        session.query(History.uuid)
        .filter(History.name.ilike(username))
        .order_by(History.observed_at.desc())
        .first()
    )
    return row[0] if row else None


def update_source_timestamp(session, uuid: str, source: str):
    now = datetime.now(timezone.utc)
    su = session.query(SourceUpdate).filter_by(uuid=uuid, source=source).first()
    if not su:
        session.add(
            SourceUpdate(uuid=uuid, source=source, last_updated_at=now)
        )
    else:
        su.last_updated_at = now


def is_source_stale(session, uuid: str, source: str, stale_hours: int) -> bool:
    su = session.query(SourceUpdate).filter_by(uuid=uuid, source=source).first()
    if not su or not su.last_updated_at:
        return True
    last_updated = (
        su.last_updated_at.replace(tzinfo=timezone.utc)
        if su.last_updated_at.tzinfo is None
        else su.last_updated_at
    )
    return (datetime.now(timezone.utc) - last_updated).total_seconds() > (
        stale_hours * 3600
    )


def query_history_public(session, uuid: str) -> Dict[str, Any]:
    profile = session.get(Profile, uuid)
    if not profile:
        return {
            "query": None,
            "uuid": uuid,
            "last_seen_at": None,
            "history": [],
        }

    rows = session.query(History).filter_by(uuid=uuid).all()

    # Sort chronologically: None (original name) first, then by timestamp
    def sort_key(r):
        if r.changed_at is None:
            return ""  # Empty string sorts before ISO dates
        return r.changed_at

    rows.sort(key=sort_key)

    history = [
        {
            "id": i + 1,
            "name": r.name,
            "changed_at": r.changed_at,
            "observed_at": r.observed_at.isoformat(),
            "censored": r.name in {"-"},
        }
        for i, r in enumerate(rows)
    ]
    return {
        "query": profile.query,
        "uuid": profile.uuid,
        "last_seen_at": profile.last_seen_at.isoformat(),
        "history": history,
    }


def _update_profile_from_sources(uuid: str, current_name: str) -> Dict[str, Any]:
    log.info(
        f"Updating profile from sources: uuid={uuid}, current_name={current_name}"
    )

    # Fetch from BOTH sources
    rows = gather_remote_rows_by_uuid(uuid, current_name)
    log.debug(f"Total fetched: {len(rows)} entries from all sources")

    # Intelligently merge the data with timestamp normalization
    pairs = merge_remote_sources(rows, current_name)
    log.info(f"Merged into {len(pairs)} final name history entries")

    with tx() as con:
        # Do NOT delete existing history anymore. Preserve manual/censored entries.
        ensure_profile(con, uuid, current_name)
        now = datetime.now(timezone.utc)

        # Build a set of existing composite keys to avoid duplicates
        existing = set(
            (
                r.name.lower(),
                r.changed_at if r.changed_at is not None else None,
            )
            for r in con.query(History).filter_by(uuid=uuid).all()
        )

        inserted = 0
        for name, changed_at in pairs:
            key = (name.lower(), changed_at if changed_at is not None else None)
            if key in existing:
                continue
            con.add(
                History(
                    uuid=uuid,
                    name=name,
                    changed_at=changed_at,
                    observed_at=now,
                )
            )
            existing.add(key)
            inserted += 1
        log.info(f"Inserted {inserted} new history rows (preserved existing).")

        update_source_timestamp(con, uuid, "scraper")
        con.flush()  # Ensure changes are visible to the query
        return query_history_public(con, uuid)


def delete_profile(session, uuid: str) -> bool:
    profile = session.get(Profile, uuid)
    if profile:
        session.delete(profile)
        return True
    return False


# --- FLASK API ---
app = Flask(__name__)
RATE_BUCKET, RATE_LOCK = {}, threading.Lock()


@app.errorhandler(HTTPException)
def handle_exception(e: HTTPException):
    response = e.get_response()
    response.data = json.dumps(
        {"code": e.code, "name": e.name, "description": e.description}
    )
    response.content_type = "application/json"
    return response


@app.before_request
def _apply_rate_limit():
    ip = (
        request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        .split(",")[0]
        .strip()
    )
    now = time.time()
    with RATE_LOCK:
        bucket = RATE_BUCKET.setdefault(ip, [])
        bucket[:] = [t for t in bucket if now - t <= RATE_LIMIT_WINDOW]
        if len(bucket) >= RATE_LIMIT_RPM:
            abort(429)
        bucket.append(now)


@app.route("/api/namehistory", methods=["GET"])
def api_namehistory():
    username = request.args.get("username", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_]{1,16}", username):
        abort(400, "username required")

    normalized = normalize_username(username)
    uuid, current_name = (None, None)
    if normalized:
        current_name, uuid = normalized
    else:
        with tx() as s:
            uuid = get_uuid_from_history_by_name(s, username)
        if uuid:
            current_name = get_profile_by_uuid_from_mojang(uuid)

    if not uuid or not current_name:
        abort(404, "Username not found")
    with tx() as s:
        if not is_source_stale(s, uuid, "scraper", SCRAPER_STALE_HOURS):
            return jsonify(query_history_public(s, uuid))
    return jsonify(_update_profile_from_sources(uuid, current_name))


@app.route("/api/namehistory/uuid/<uuid>", methods=["GET"])
def api_namehistory_by_uuid(uuid):
    uuid = dashed_uuid(uuid)
    current_name = get_profile_by_uuid_from_mojang(uuid)
    if not current_name:
        abort(404, "UUID not found")
    with tx() as s:
        if not is_source_stale(s, uuid, "scraper", SCRAPER_STALE_HOURS):
            return jsonify(query_history_public(s, uuid))
    return jsonify(_update_profile_from_sources(uuid, current_name))


@app.route("/api/namehistory", methods=["DELETE"])
@require_auth(AuthLevel.ADMIN)
def api_delete():
    username = request.args.get("username", "").strip()
    uuid_param = request.args.get("uuid", "").strip()
    if not username and not uuid_param:
        abort(400, "username or uuid required")

    uuid = dashed_uuid(uuid_param) if uuid_param else None
    if not uuid and username:
        norm = normalize_username(username)
        if norm:
            _, uuid = norm
        else:
            with tx() as s:
                uuid = get_uuid_from_history_by_name(s, username)

    if not uuid:
        abort(404, "Profile not found")
    with tx() as s:
        if not delete_profile(s, uuid):
            abort(404, "Profile not found in database")
    return jsonify({"message": "Profile deleted", "uuid": uuid})


@app.route("/api/namehistory/update", methods=["POST"])
@require_auth(AuthLevel.API)
def api_update():
    body = request.get_json(force=True, silent=False) or {}
    results = {"updated": [], "errors": []}

    usernames = body.get("usernames", [])
    if "username" in body:
        usernames.append(body["username"])
    uuids = body.get("uuids", [])
    if "uuid" in body:
        uuids.append(body["uuid"])

    for name in usernames:
        norm = normalize_username(name)
        if not norm:
            results["errors"].append({"username": name, "error": "Not found"})
        else:
            results["updated"].append(
                _update_profile_from_sources(norm[1], norm[0])
            )
            bulk_sleep()

    for u in uuids:
        u = dashed_uuid(u)
        name = get_profile_by_uuid_from_mojang(u)
        if not name:
            results["errors"].append({"uuid": u, "error": "Not found"})
        else:
            results["updated"].append(_update_profile_from_sources(u, name))
            bulk_sleep()

    return jsonify(results)


# --- MAIN ---
if __name__ == "__main__":
    init_db()
    log.info("DB initialized")
    if AUTO_UPDATE_ENABLED:
        # Auto-update worker can be started in a separate thread here if desired
        pass
    log.info(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT)