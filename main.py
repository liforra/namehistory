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

from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, ForeignKey, 
    UniqueConstraint
)
from sqlalchemy.orm import (
    sessionmaker, declarative_base, relationship, scoped_session
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
    "fetch": {
        "default_sleep": 0.5,
        "bulk_min_delay": 0.8,
        "bulk_max_delay": 1.5,
    },
    "database": {"type": "sqlite", "path": "namehistory.db"},
    "logging": {
        "level": "INFO", "json": False, "file": "", 
        "max_bytes": 10_485_760, "backup_count": 3
    },
    "auto_update": {
        "enabled": True, "check_interval_minutes": 60,
        "scraper_stale_hours": 24, "batch_size": 10,
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

# --- LOGGING SETUP ---
def setup_logging():
    lvl = str(CONFIG.get("logging", {}).get("level", "INFO")).upper()
    json_mode = bool(CONFIG.get("logging", {}).get("json", False))
    log_file = CONFIG.get("logging", {}).get("file") or ""
    max_bytes = int(CONFIG.get("logging", {}).get("max_bytes", 10_485_760))
    backup_count = int(CONFIG.get("logging", {}).get("backup_count", 3))

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {"level": record.levelname, "ts": datetime.now(timezone.utc).isoformat(), "msg": record.getMessage()}
            if record.exc_info: payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=False)

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers: root.removeHandler(handler)
    root.setLevel(getattr(logging, lvl, logging.INFO))
    fmt = JsonFormatter() if json_mode else logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        fh.setFormatter(fmt)
        root.addHandler(fh)

setup_logging()
log = logging.getLogger("namehistory")

# --- HTTP & UTILS ---
UA_POOL = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"]
scraper_session = Session(impersonate="chrome110")
mojang_session = requests.Session()
MCS_LOOKUP = "https://api.minecraftservices.com/minecraft/profile/lookup/name/{name}"
MCS_PROFILE = "https://sessionserver.mojang.com/session/minecraft/profile/{uuid}"

def jitter_sleep(base: float = DEFAULT_SLEEP): time.sleep(base + random.uniform(0.05, 0.2))
def bulk_sleep(): time.sleep(random.uniform(BULK_MIN_DELAY, BULK_MAX_DELAY))

def dashed_uuid(raw: str) -> str:
    raw = raw.replace("-", "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{32}", raw):
        s = raw.lower()
        return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"
    return raw

# --- DATA FETCHING ---
def normalize_username(name: str) -> Optional[Tuple[str, str]]:
    try:
        r = mojang_session.get(MCS_LOOKUP.format(name=name), timeout=10.0)
        if r.status_code != 200: return None
        data = r.json()
        return (data.get("name"), dashed_uuid(data.get("id")))
    except Exception: return None

def get_profile_by_uuid_from_mojang(uuid: str) -> Optional[str]:
    try:
        r = mojang_session.get(MCS_PROFILE.format(uuid=uuid.replace("-", "")), timeout=10.0)
        return r.json().get("name") if r.status_code == 200 else None
    except Exception: return None

def fetch_profile_html(url: str) -> str:
    try:
        r = scraper_session.get(url, headers={"User-Agent": random.choice(UA_POOL)}, timeout=15.0)
        return r.text if r.status_code == 200 else ""
    except Exception: return ""

def fetch_namemc_data(username: str) -> List[Dict[str, Optional[str]]]:
    html = fetch_profile_html(f"https://namemc.com/profile/{username}")
    if not html: return []
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one('table.table-striped')
    if not table: return []
    out = []
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) > 0:
            name = cells[0].get_text(strip=True)
            time_tag = cells[1].find("time") if len(cells) > 1 else None
            changed_at = time_tag["datetime"].strip() if time_tag else None
            out.append({"name": name, "changedAt": changed_at})
    return out

def fetch_laby_api_data(uuid: str) -> List[Dict[str, Optional[str]]]:
    url = f"https://laby.net/api/user/{uuid}/get-names"
    try:
        r = scraper_session.get(url, headers={"Accept": "application/json"}, timeout=10)
        if r.status_code != 200: return []
        return [{"name": e.get("name"), "changedAt": e.get("changed_at")} for e in r.json() if e.get("name")]
    except Exception: return []

def gather_remote_rows_by_uuid(uuid: str) -> List[Dict[str, Optional[str]]]:
    rows = []
    current_name = get_profile_by_uuid_from_mojang(uuid)
    if current_name:
        rows.extend(fetch_namemc_data(current_name))
        jitter_sleep()
    rows.extend(fetch_laby_api_data(uuid))
    return rows

def merge_remote_sources(rows: List[Dict[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
    # De-duplicates based on (name, changedAt) tuple, preserving name reuses
    unique_entries = set()
    for row in rows:
        name = row.get("name")
        if name:
            unique_entries.add((name, row.get("changedAt")))
    return list(unique_entries)

# --- DATABASE ---
Base = declarative_base()
class Profile(Base):
    __tablename__ = "profiles"
    uuid = Column(String(36), primary_key=True)
    query = Column(String(32)); last_seen_at = Column(DateTime)
    history = relationship("History", back_populates="profile", cascade="all, delete-orphan")
    source_updates = relationship("SourceUpdate", back_populates="profile", cascade="all, delete-orphan")

class History(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), ForeignKey("profiles.uuid", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(32), nullable=False, index=True)
    changed_at = Column(String(32), index=True); observed_at = Column(DateTime, nullable=False)
    profile = relationship("Profile", back_populates="history")
    __table_args__ = (UniqueConstraint("uuid", "name", "changed_at"),)

class SourceUpdate(Base):
    __tablename__ = "source_updates"
    uuid = Column(String(36), ForeignKey("profiles.uuid", ondelete="CASCADE"), primary_key=True)
    source = Column(String(32), primary_key=True)
    last_updated_at = Column(DateTime, nullable=False, index=True)
    profile = relationship("Profile", back_populates="source_updates")

db_url = f"sqlite:///{DB_PATH}" if DB_TYPE == "sqlite" else DB_URL
engine = create_engine(db_url, echo=False, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))

def init_db(): Base.metadata.create_all(bind=engine)

@contextmanager
def tx():
    session = SessionLocal()
    try: yield session; session.commit()
    except: session.rollback(); raise
    finally: session.close()

# --- CORE LOGIC ---
def ensure_profile(session, uuid: str, query: Optional[str]):
    profile = session.get(Profile, uuid)
    now = datetime.now(timezone.utc)
    if not profile: session.add(Profile(uuid=uuid, query=query, last_seen_at=now))
    else: profile.query = query; profile.last_seen_at = now

def get_uuid_from_history_by_name(session, username: str) -> Optional[str]:
    row = session.query(History.uuid).filter(History.name.ilike(username)).order_by(History.observed_at.desc()).first()
    return row[0] if row else None

def update_source_timestamp(session, uuid: str, source: str):
    now = datetime.now(timezone.utc)
    su = session.query(SourceUpdate).filter_by(uuid=uuid, source=source).first()
    if not su: session.add(SourceUpdate(uuid=uuid, source=source, last_updated_at=now))
    else: su.last_updated_at = now

def is_source_stale(session, uuid: str, source: str, stale_hours: int) -> bool:
    su = session.query(SourceUpdate).filter_by(uuid=uuid, source=source).first()
    if not su or not su.last_updated_at: return True
    last_updated = su.last_updated_at.replace(tzinfo=timezone.utc) if su.last_updated_at.tzinfo is None else su.last_updated_at
    return (datetime.now(timezone.utc) - last_updated).total_seconds() > (stale_hours * 3600)

def query_history_public(session, uuid: str) -> Dict[str, Any]:
    profile = session.get(Profile, uuid)
    if not profile: return {"query": None, "uuid": uuid, "last_seen_at": None, "history": []}
    
    rows = session.query(History).filter_by(uuid=uuid).all()
    rows.sort(key=lambda r: r.changed_at or "0") # Sort chronologically, nulls first
    
    history = [{"id": i+1, "name": r.name, "changed_at": r.changed_at, "observed_at": r.observed_at.isoformat(), "censored": r.name in {"-"}} for i, r in enumerate(rows)]
    return {"query": profile.query, "uuid": profile.uuid, "last_seen_at": profile.last_seen_at.isoformat(), "history": history}

def _update_profile_from_sources(uuid: str, current_name: str) -> Dict[str, Any]:
    log.info("Updating profile from sources", extra={"uuid": uuid, "current_name": current_name})
    
    rows = gather_remote_rows_by_uuid(uuid)
    rows.append({"name": current_name, "changedAt": None})
    
    pairs = merge_remote_sources(rows)

    with tx() as con:
        # Clean slate: Delete old history to ensure data is fresh and correct
        con.query(History).filter_by(uuid=uuid).delete()
        
        ensure_profile(con, uuid, current_name)
        for name, changed_at in pairs:
            con.add(History(uuid=uuid, name=name, changed_at=changed_at, observed_at=datetime.now(timezone.utc)))
        
        update_source_timestamp(con, uuid, "scraper")
        return query_history_public(con, uuid)

def delete_profile(session, uuid: str) -> bool:
    profile = session.get(Profile, uuid)
    if profile: session.delete(profile); return True
    return False

# --- FLASK API ---
app = Flask(__name__)
RATE_BUCKET, RATE_LOCK = {}, threading.Lock()

@app.errorhandler(HTTPException)
def handle_exception(e: HTTPException):
    response = e.get_response()
    response.data = json.dumps({"code": e.code, "name": e.name, "description": e.description})
    response.content_type = "application/json"
    return response

@app.before_request
def _apply_rate_limit():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    now = time.time()
    with RATE_LOCK:
        bucket = RATE_BUCKET.setdefault(ip, [])
        bucket[:] = [t for t in bucket if now - t <= RATE_LIMIT_WINDOW]
        if len(bucket) >= RATE_LIMIT_RPM: abort(429)
        bucket.append(now)

@app.route("/api/namehistory", methods=["GET"])
def api_namehistory():
    username = request.args.get("username", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_]{1,16}", username): abort(400, "username required")
    
    normalized = normalize_username(username)
    uuid, current_name = (None, None)
    if normalized: current_name, uuid = normalized
    else:
        with tx() as s: uuid = get_uuid_from_history_by_name(s, username)
        if uuid: current_name = get_profile_by_uuid_from_mojang(uuid)

    if not uuid or not current_name: abort(404, "Username not found")
    with tx() as s:
        if not is_source_stale(s, uuid, "scraper", SCRAPER_STALE_HOURS):
            return jsonify(query_history_public(s, uuid))
    return jsonify(_update_profile_from_sources(uuid, current_name))

@app.route("/api/namehistory/uuid/<uuid>", methods=["GET"])
def api_namehistory_by_uuid(uuid):
    uuid = dashed_uuid(uuid)
    current_name = get_profile_by_uuid_from_mojang(uuid)
    if not current_name: abort(404, "UUID not found")
    with tx() as s:
        if not is_source_stale(s, uuid, "scraper", SCRAPER_STALE_HOURS):
            return jsonify(query_history_public(s, uuid))
    return jsonify(_update_profile_from_sources(uuid, current_name))

@app.route("/api/namehistory", methods=["DELETE"])
def api_delete():
    username = request.args.get("username", "").strip()
    uuid_param = request.args.get("uuid", "").strip()
    if not username and not uuid_param: abort(400, "username or uuid required")
    
    uuid = dashed_uuid(uuid_param) if uuid_param else None
    if not uuid and username:
        norm = normalize_username(username)
        if norm: _, uuid = norm
        else:
            with tx() as s: uuid = get_uuid_from_history_by_name(s, username)
    
    if not uuid: abort(404, "Profile not found")
    with tx() as s:
        if not delete_profile(s, uuid): abort(404, "Profile not found in database")
    return jsonify({"message": "Profile deleted", "uuid": uuid})

@app.route("/api/namehistory/update", methods=["POST"])
def api_update():
    body = request.get_json(force=True, silent=False) or {}
    results = {"updated": [], "errors": []}
    
    usernames = body.get("usernames", [])
    if "username" in body: usernames.append(body["username"])
    uuids = body.get("uuids", [])
    if "uuid" in body: uuids.append(body["uuid"])

    for name in usernames:
        norm = normalize_username(name)
        if not norm: results["errors"].append({"username": name, "error": "Not found"})
        else: 
            results["updated"].append(_update_profile_from_sources(norm[1], norm[0]))
            bulk_sleep()
        
    for u in uuids:
        u = dashed_uuid(u)
        name = get_profile_by_uuid_from_mojang(u)
        if not name: results["errors"].append({"uuid": u, "error": "Not found"})
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