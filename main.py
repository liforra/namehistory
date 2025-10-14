#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import random
import sqlite3
import threading
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from curl_cffi.requests import Session
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, abort
from werkzeug.exceptions import HTTPException, ServiceUnavailable

try:
    import tomli as toml
except Exception:
    toml = None

import logging
from logging.handlers import RotatingFileHandler

CONFIG: Dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 8000},
    "rate_limit": {"rpm": 5, "window_seconds": 60},
    "fetch": {
        "fuzzy_window_days": 1,
        "default_sleep": 0.5,
        "bulk_min_delay": 0.8,
        "bulk_max_delay": 1.5,
    },
    "database": {"path": "namehistory.db"},
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
        "mojang_stale_hours": 6,
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
DB_PATH = CONFIG["database"]["path"]
FUZZY_WINDOW = timedelta(days=float(CONFIG["fetch"]["fuzzy_window_days"]))
DEFAULT_SLEEP = float(CONFIG["fetch"]["default_sleep"])
BULK_MIN_DELAY = float(CONFIG["fetch"]["bulk_min_delay"])
BULK_MAX_DELAY = float(CONFIG["fetch"]["bulk_max_delay"])
RATE_LIMIT_RPM = int(CONFIG["rate_limit"]["rpm"])
RATE_LIMIT_WINDOW = float(CONFIG["rate_limit"]["window_seconds"])

AUTO_UPDATE_ENABLED = bool(CONFIG["auto_update"]["enabled"])
AUTO_UPDATE_CHECK_INTERVAL = int(CONFIG["auto_update"]["check_interval_minutes"]) * 60
MOJANG_STALE_HOURS = int(CONFIG["auto_update"]["mojang_stale_hours"])
SCRAPER_STALE_HOURS = int(CONFIG["auto_update"]["scraper_stale_hours"])
AUTO_UPDATE_BATCH_SIZE = int(CONFIG["auto_update"]["batch_size"])


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
                "logger": record.name,
            }
            for key, value in record.__dict__.items():
                if key not in logging.LogRecord.__dict__ and key not in payload:
                    payload[key] = value
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
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        fh.setFormatter(fmt)
        root.addHandler(fh)


setup_logging()
log = logging.getLogger("namehistory")

UA_POOL = (
    [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:131.0) Gecko/20100101 Firefox/131.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) CriOS/124.0.0.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X; rv:131.0) Gecko/20100101 Firefox/131.0 Mobile/15E148",
        "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (Linux; Android 14; SM-S921B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/23.0 Chrome/120.0.6099.231 Mobile Safari/537.36",
        "Mozilla/5.0 (Android 14; Mobile; rv:131.0) Gecko/131.0 Firefox/131.0",
    ]
    + [
        f"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{rv}.0) Gecko/20100101 Firefox/{rv}.0"
        for rv in range(90, 120)
    ]
    + [
        f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{v}.0.0.0 Safari/537.36"
        for v in range(100, 120)
    ]
    + [
        f"Mozilla/5.0 (Macintosh; Intel Mac OS X 14_{m}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.{m} Safari/605.1.15"
        for m in range(0, 10)
    ]
)
UA_POOL = UA_POOL[:100]

scraper_session = Session(impersonate="chrome110")
mojang_session = requests.Session()


def jitter_sleep(base: float = DEFAULT_SLEEP):
    time.sleep(base + random.uniform(0.05, 0.2))


def bulk_sleep():
    time.sleep(random.uniform(BULK_MIN_DELAY, BULK_MAX_DELAY))


MCS_LOOKUP = "https://api.minecraftservices.com/minecraft/profile/lookup/name/{name}"
MCS_PROFILE = "https://sessionserver.mojang.com/session/minecraft/profile/{uuid}"


def dashed_uuid(raw: str) -> str:
    raw = raw.replace("-", "").strip()
    m = re.fullmatch(r"[0-9a-fA-F]{32}", raw)
    if not m:
        return raw
    s = raw.lower()
    return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"


def normalize_username(name: str) -> Optional[Tuple[str, Optional[str]]]:
    """Get properly capitalized username and UUID from Mojang API"""
    try:
        r = mojang_session.get(MCS_LOOKUP.format(name=name), timeout=10.0)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        proper_name = data.get("name")
        uuid = data.get("id")
        return (proper_name, dashed_uuid(uuid) if uuid else None)
    except Exception as e:
        log.warning(
            "Failed to normalize username", extra={"username": name, "error": str(e)}
        )
        return (name, None)


def get_profile_by_uuid_from_mojang(uuid: str) -> Optional[str]:
    """Gets the current username for a given UUID from Mojang."""
    try:
        url = MCS_PROFILE.format(uuid=uuid.replace("-", ""))
        r = mojang_session.get(url, timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            return data.get("name")
        return None
    except Exception as e:
        log.error(
            "Failed to get profile by UUID from Mojang",
            extra={"uuid": uuid, "error": str(e)},
        )
        return None


def is_censored(name: Optional[str]) -> bool:
    if name is None:
        return False
    n = name.strip()
    return n in {"-", "—", "–", "‑"} or n == ""


def fetch_profile_html(url: str, timeout: float = 15.0) -> str:
    t0 = time.time()
    headers = {"User-Agent": random.choice(UA_POOL)}
    try:
        r = scraper_session.get(url, headers=headers, timeout=timeout)
        if r.status_code >= 400:
            log.error("Fetch error", extra={"url": url, "status": r.status_code})
            return ""
        log.info(
            "Fetch ok",
            extra={
                "url": url,
                "status": r.status_code,
                "ms": int((time.time() - t0) * 1000),
            },
        )
        return r.text
    except Exception as e:
        log.error("Fetch exception", extra={"url": url, "error": str(e)})
        return ""


def parse_namemc_html(html: str) -> List[Dict[str, Optional[str]]]:
    soup = BeautifulSoup(html, "html.parser")
    header = soup.find(string=re.compile(r"^\s*Name History\s*$"))
    if not header:
        return []
    card = header.find_parent(class_="card")
    if not card:
        return []
    table = card.find("table")
    if not table:
        return []
    out = []
    for row in table.find("tbody").find_all("tr"):
        if "d-lg-none" in row.get("class", []):
            continue
        name_tag = row.select_one("td a")
        name = name_tag.get_text(strip=True) if name_tag else None
        time_tag = row.find("time", datetime=True)
        changed_at = time_tag["datetime"].strip() if time_tag else None
        if name:
            out.append({"name": name, "changedAt": changed_at})
    return out


def fetch_laby_api_data(uuid: str) -> List[Dict[str, Optional[str]]]:
    url = f"https://laby.net/api/user/{uuid}/name-history"
    log.info("Fetching Laby.net API", extra={"url": url})
    try:
        r = scraper_session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data:
            ts = item.get("changed_at")
            iso_ts = (
                datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                if isinstance(ts, int)
                else None
            )
            out.append({"name": item.get("name"), "changedAt": iso_ts})
        return out
    except Exception as e:
        log.error("Laby.net API fetch failed", extra={"url": url, "error": str(e)})
        return []


def gather_remote_rows_by_uuid(uuid: str) -> List[Dict[str, Optional[str]]]:
    rows: List[Dict[str, Optional[str]]] = []
    current_name = get_profile_by_uuid_from_mojang(uuid)

    # Scrape NameMC using the current name
    if current_name:
        try:
            html_a = fetch_profile_html(f"https://namemc.com/profile/{current_name}")
            if html_a:
                rows.extend(parse_namemc_html(html_a))
        except Exception:
            log.warning("Source A (namemc) failed", exc_info=False)
        jitter_sleep()

    # Scrape Laby.net using UUID
    try:
        rows.extend(fetch_laby_api_data(uuid))
    except Exception:
        log.warning("Source B (laby) failed", exc_info=False)

    return rows


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt:
        return None
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        return None


DB_LOCK = threading.Lock()
SCHEMA = """
PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS profiles (uuid TEXT PRIMARY KEY, query TEXT, last_seen_at TEXT);
CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, uuid TEXT NOT NULL, name TEXT NOT NULL, changed_at TEXT, observed_at TEXT NOT NULL, UNIQUE(uuid, name, changed_at), FOREIGN KEY (uuid) REFERENCES profiles(uuid) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS provider_details (id INTEGER PRIMARY KEY AUTOINCREMENT, history_id INTEGER NOT NULL, provider TEXT NOT NULL, provider_changed_at TEXT, UNIQUE(history_id, provider, provider_changed_at), FOREIGN KEY (history_id) REFERENCES history(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS source_updates (uuid TEXT NOT NULL, source TEXT NOT NULL, last_updated_at TEXT NOT NULL, PRIMARY KEY (uuid, source), FOREIGN KEY (uuid) REFERENCES profiles(uuid) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS idx_history_uuid ON history(uuid);
CREATE INDEX IF NOT EXISTS idx_history_uuid_changed ON history(uuid, changed_at);
CREATE INDEX IF NOT EXISTS idx_history_name ON history(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_source_updates_time ON source_updates(source, last_updated_at);
"""


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect_db() as con:
        con.executescript(SCHEMA)


@contextmanager
def tx() -> sqlite3.Connection:
    with DB_LOCK:
        con = connect_db()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.execute("COMMIT")
        except:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()


def ensure_profile(con: sqlite3.Connection, uuid: str, query: Optional[str]) -> None:
    cur = con.execute("SELECT uuid FROM profiles WHERE uuid = ?", (uuid,))
    if cur.fetchone() is None:
        con.execute(
            "INSERT INTO profiles (uuid, query, last_seen_at) VALUES (?, ?, ?)",
            (uuid, query, now_iso()),
        )
    else:
        con.execute(
            "UPDATE profiles SET query = ?, last_seen_at = ? WHERE uuid = ?",
            (query, now_iso(), uuid),
        )


def get_profile_by_uuid(con: sqlite3.Connection, uuid: str) -> Optional[sqlite3.Row]:
    return con.execute("SELECT * FROM profiles WHERE uuid = ?", (uuid,)).fetchone()


def get_uuid_from_history_by_name(
    con: sqlite3.Connection, username: str
) -> Optional[str]:
    """Find a UUID from the history table based on a past username."""
    row = con.execute(
        "SELECT uuid FROM history WHERE name = ? COLLATE NOCASE ORDER BY observed_at DESC LIMIT 1",
        (username,),
    ).fetchone()
    return row["uuid"] if row else None


def update_source_timestamp(con: sqlite3.Connection, uuid: str, source: str) -> None:
    """Track when a source was last updated for a profile"""
    con.execute(
        "INSERT OR REPLACE INTO source_updates (uuid, source, last_updated_at) VALUES (?, ?, ?)",
        (uuid, source, now_iso()),
    )


def is_source_stale(
    con: sqlite3.Connection, uuid: str, source: str, stale_hours: int
) -> bool:
    """Check if a source needs updating"""
    row = con.execute(
        "SELECT last_updated_at FROM source_updates WHERE uuid = ? AND source = ?",
        (uuid, source),
    ).fetchone()
    if not row or not row["last_updated_at"]:
        return True
    last_updated = parse_iso(row["last_updated_at"])
    if not last_updated:
        return True
    age = datetime.now(timezone.utc) - last_updated
    return age.total_seconds() > (stale_hours * 3600)


def insert_or_merge_history(
    con: sqlite3.Connection,
    uuid: str,
    name: str,
    changed_at: Optional[str],
    provider: str,
    provider_changed_at: Optional[str],
) -> None:
    if not is_censored(name) and changed_at is not None:
        cur = con.execute(
            "SELECT id, name FROM history WHERE uuid = ? AND changed_at = ?",
            (uuid, changed_at),
        )
        for row in cur.fetchall():
            if is_censored(row["name"]):
                con.execute(
                    "UPDATE history SET name = ? WHERE id = ?", (name, row["id"])
                )
                con.execute(
                    "INSERT OR IGNORE INTO provider_details (history_id, provider, provider_changed_at) VALUES (?,?,?)",
                    (row["id"], provider, provider_changed_at),
                )
                logging.getLogger("merge").info(
                    "Uncensor upgrade",
                    extra={"uuid": uuid, "ts": changed_at, "username": name},
                )
                return
    ta = parse_iso(changed_at)
    if ta:
        cur = con.execute(
            "SELECT id, changed_at FROM history WHERE uuid = ? AND name = ?",
            (uuid, name),
        )
        for row in cur.fetchall():
            tb = parse_iso(row["changed_at"])
            if tb and abs(ta - tb) <= FUZZY_WINDOW:
                if row["changed_at"] is None:
                    con.execute(
                        "UPDATE history SET changed_at = ? WHERE id = ?",
                        (changed_at, row["id"]),
                    )
                con.execute(
                    "INSERT OR IGNORE INTO provider_details (history_id, provider, provider_changed_at) VALUES (?,?,?)",
                    (row["id"], provider, provider_changed_at),
                )
                logging.getLogger("merge").info(
                    "Fuzzy-merged",
                    extra={
                        "uuid": uuid,
                        "username": name,
                        "from": row["changed_at"],
                        "to": changed_at,
                    },
                )
                return
    con.execute(
        "INSERT OR IGNORE INTO history (uuid, name, changed_at, observed_at) VALUES (?,?,?,?)",
        (uuid, name, changed_at, now_iso()),
    )
    cur = con.execute(
        "SELECT id FROM history WHERE uuid = ? AND name = ? AND (changed_at = ? OR (changed_at IS NULL AND ? IS NULL))",
        (uuid, name, changed_at, changed_at),
    )
    result = cur.fetchone()
    if result:
        hid = result["id"]
        con.execute(
            "INSERT OR IGNORE INTO provider_details (history_id, provider, provider_changed_at) VALUES (?,?,?)",
            (hid, provider, provider_changed_at),
        )
        logging.getLogger("merge").info(
            "Inserted history",
            extra={"uuid": uuid, "username": name, "changed_at": changed_at},
        )


def ensure_current_entry(con: sqlite3.Connection, uuid: str, current_name: str) -> None:
    """
    Ensures that an undated entry for the current name exists. This prevents duplicates.
    """
    # Check if an undated entry for the current name already exists.
    cur = con.execute(
        "SELECT id FROM history WHERE uuid = ? AND name = ? AND changed_at IS NULL",
        (uuid, current_name),
    )
    if cur.fetchone():
        # It exists, do nothing.
        return

    # It does not exist, so we must add it.
    log.info(
        "Adding missing 'Current' entry", extra={"uuid": uuid, "username": current_name}
    )
    insert_or_merge_history(con, uuid, current_name, None, "current", None)


def query_history_public(con: sqlite3.Connection, uuid: str) -> Dict[str, Any]:
    p = get_profile_by_uuid(con, uuid)
    if not p:
        return {"query": None, "uuid": uuid, "last_seen_at": None, "history": []}

    cur = con.execute(
        "SELECT name, changed_at, observed_at FROM history WHERE uuid = ?", (uuid,)
    )
    rows = cur.fetchall()

    dated = []
    undated = []

    for r in rows:
        if r["changed_at"] is not None:
            dated.append(dict(r))
        else:
            undated.append(dict(r))

    dated.sort(key=lambda x: x["changed_at"])
    undated.sort(key=lambda x: x["observed_at"])

    # Identify the true original name (earliest observed, undated)
    original_name_entry = undated[0] if undated else None

    # Identify the true current name (latest observed, undated)
    current_name_entry = undated[-1] if undated else None

    # Build the final list
    final_list = []
    if original_name_entry:
        final_list.append(original_name_entry)

    # Add dated history, ensuring no overlap with the original name
    for entry in dated:
        if not original_name_entry or (entry["name"] != original_name_entry["name"]):
            final_list.append(entry)

    # Add current name if it's different from original and last dated
    if current_name_entry:
        is_original = (
            original_name_entry
            and current_name_entry["observed_at"] == original_name_entry["observed_at"]
        )
        is_last_dated = dated and current_name_entry["name"] == dated[-1]["name"]

        if not is_original and not is_last_dated:
            final_list.append(current_name_entry)

    # Deduplicate the final list based on name and changed_at
    seen = set()
    deduplicated_list = []
    for item in final_list:
        key = (item["name"], item["changed_at"])
        if key not in seen:
            seen.add(key)
            deduplicated_list.append(item)

    items = []
    for idx, r in enumerate(deduplicated_list):
        items.append(
            {
                "id": idx + 1,
                "name": r["name"],
                "changed_at": r["changed_at"],
                "observed_at": r["observed_at"],
                "censored": is_censored(r["name"]),
            }
        )

    return {
        "query": p["query"],
        "uuid": p["uuid"],
        "last_seen_at": p["last_seen_at"],
        "history": items,
    }


def merge_remote_sources(
    rows: List[Dict[str, Optional[str]]],
) -> List[Tuple[str, Optional[str]]]:
    """
    Intelligently merge entries from multiple sources.
    Prefer entries with timestamps, deduplicate by name.
    """
    # Group by name (case-insensitive)
    name_map: Dict[str, List[Tuple[str, Optional[str]]]] = {}

    for row in rows:
        name = row.get("name")
        changed_at = row.get("changedAt")

        if not name:
            continue

        name_lower = name.lower()
        if name_lower not in name_map:
            name_map[name_lower] = []

        name_map[name_lower].append((name, changed_at))

    # For each name, pick the best entry
    merged: Dict[str, Tuple[str, Optional[str]]] = {}

    for name_lower, entries in name_map.items():
        # Prefer entries with timestamps
        dated = [e for e in entries if e[1] is not None]
        undated = [e for e in entries if e[1] is None]

        if dated:
            # Sort by timestamp to get consistent results
            dated.sort(key=lambda x: x[1])

            # Group by similar timestamps (within fuzzy window)
            timestamp_groups: List[List[Tuple[str, Optional[str]]]] = []

            for entry in dated:
                placed = False
                entry_time = parse_iso(entry[1])

                if entry_time:
                    for group in timestamp_groups:
                        group_time = parse_iso(group[0][1])
                        if group_time and abs(entry_time - group_time) <= FUZZY_WINDOW:
                            group.append(entry)
                            placed = True
                            break

                if not placed:
                    timestamp_groups.append([entry])

            # For each timestamp group, use the best capitalization
            # (prefer the first entry from the most reliable source)
            for group in timestamp_groups:
                # Use the first entry's timestamp and name
                best_entry = group[0]
                # Store with the timestamp as key to avoid duplicates
                key = f"{name_lower}_{best_entry[1]}"
                if key not in merged:
                    merged[key] = best_entry

        # If we have undated entries and no dated entries, add one undated
        if undated and not dated:
            # Use the first undated entry
            key = f"{name_lower}_undated"
            if key not in merged:
                merged[key] = undated[0]

    # Convert back to list and sort by timestamp
    result = list(merged.values())

    # Separate dated and undated
    dated_entries = [(n, t) for n, t in result if t is not None]
    undated_entries = [(n, t) for n, t in result if t is None]

    # Sort dated by timestamp
    dated_entries.sort(key=lambda x: x[1])

    # Return undated first, then dated
    return undated_entries + dated_entries


def _update_profile_from_sources(uuid: str, current_name: str) -> Dict[str, Any]:
    """The authoritative function to fetch, merge, and save a profile's history."""
    log.info(
        "Updating profile from sources",
        extra={"uuid": uuid, "current_name": current_name},
    )

    rows = gather_remote_rows_by_uuid(uuid)

    if not rows:
        log.warning(
            "No historical data found from any source for profile", extra={"uuid": uuid}
        )
        # Still proceed to save the profile info we have

    pairs = merge_remote_sources(rows)

    with tx() as con:
        # Update the main profile entry with the latest known current name
        ensure_profile(con, uuid, current_name)

        # Insert all historical names
        for n, t in pairs:
            if n is None:
                continue
            insert_or_merge_history(con, uuid, n, t, "profiles", t)

        # Ensure the current name is correctly marked as the last entry
        ensure_current_entry(con, uuid, current_name)

        # Update source timestamps
        update_source_timestamp(con, uuid, "mojang")
        update_source_timestamp(con, uuid, "scraper")

        return query_history_public(con, uuid)


def clean_database():
    """Remove duplicate names from history, keeping only unique entries"""
    log.info("=" * 60)
    log.info("Starting database cleanup...")
    log.info("=" * 60)

    with tx() as con:
        profiles = con.execute("SELECT uuid, query FROM profiles").fetchall()
        total_deleted = 0

        for profile in profiles:
            uuid = profile["uuid"]

            # Find all undated entries for the current name
            current_name = profile["query"]
            cur = con.execute(
                "SELECT id, observed_at FROM history WHERE uuid = ? AND name = ? AND changed_at IS NULL ORDER BY observed_at DESC",
                (uuid, current_name),
            )
            duplicate_currents = cur.fetchall()

            if len(duplicate_currents) > 1:
                # Keep the newest one, delete the rest
                ids_to_delete = [row["id"] for row in duplicate_currents[1:]]
                for row_id in ids_to_delete:
                    con.execute("DELETE FROM history WHERE id = ?", (row_id,))
                total_deleted += len(ids_to_delete)
                log.info(
                    f"Removed {len(ids_to_delete)} duplicate 'Current' entries for {current_name}",
                    extra={"uuid": uuid},
                )

        log.info("=" * 60)
        log.info(
            f"Database cleanup complete! Deleted {total_deleted} duplicate entries."
        )
        log.info("=" * 60)


# Auto-update background thread
def auto_update_worker():
    """Background worker that updates stale profiles"""
    log.info("Auto-update worker started")

    while True:
        try:
            time.sleep(AUTO_UPDATE_CHECK_INTERVAL)

            if not AUTO_UPDATE_ENABLED:
                continue

            log.info("Running auto-update check...")

            with tx() as con:
                # Find profiles that need scraper updates
                scraper_cutoff = (
                    datetime.now(timezone.utc) - timedelta(hours=SCRAPER_STALE_HOURS)
                ).isoformat()
                scraper_stale = con.execute(
                    """
                    SELECT p.uuid, p.query
                    FROM profiles p
                    LEFT JOIN source_updates su ON p.uuid = su.uuid AND su.source = 'scraper'
                    WHERE su.last_updated_at IS NULL OR su.last_updated_at < ?
                    ORDER BY su.last_updated_at ASC NULLS FIRST
                    LIMIT ?
                """,
                    (scraper_cutoff, AUTO_UPDATE_BATCH_SIZE),
                ).fetchall()

            # Update scraper data
            for profile in scraper_stale:
                try:
                    uuid = profile["uuid"]
                    current_name = profile["query"]
                    log.info(
                        "Auto-updating profile",
                        extra={"uuid": uuid, "username": current_name},
                    )
                    _update_profile_from_sources(uuid, current_name)
                    bulk_sleep()
                except Exception as e:
                    log.error(
                        "Auto-update failed",
                        extra={"uuid": profile["uuid"], "error": str(e)},
                    )

            log.info(f"Auto-update complete. Processed {len(scraper_stale)} profiles.")

        except Exception as e:
            log.error("Auto-update worker error", exc_info=True)


RATE_BUCKET: Dict[str, List[float]] = {}
RATE_LOCK = threading.Lock()


def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    with RATE_LOCK:
        bucket = RATE_BUCKET.setdefault(ip, [])
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW:
            bucket.pop(0)
        if len(bucket) >= RATE_LIMIT_RPM:
            return False
        bucket.append(now)
        return True


app = Flask(__name__)


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
    if not rate_limit_ok(ip):
        logging.getLogger("api").warning(
            "Rate limit", extra={"ip": ip, "path": request.path}
        )
        abort(429)
    logging.getLogger("api").info(
        "Request", extra={"ip": ip, "method": request.method, "path": request.path}
    )


@app.get("/api/namehistory")
def api_namehistory_by_username():
    username = request.args.get("username", "").strip()
    if not username or not re.fullmatch(r"[A-Za-z0-9_]{1,16}", username):
        abort(400, "username query parameter required (1–16 chars)")

    # Check Mojang API first
    normalized = normalize_username(username)

    if normalized:
        # Name exists on Mojang
        current_name, uuid = normalized
        log.info(
            "Mojang check: name exists", extra={"username": current_name, "uuid": uuid}
        )

        with tx() as con:
            if not is_source_stale(con, uuid, "scraper", SCRAPER_STALE_HOURS):
                log.info("Cache is fresh, returning cached data", extra={"uuid": uuid})
                return jsonify(query_history_public(con, uuid))

        # Cache is stale, fetch fresh data
        log.info("Cache is stale, fetching fresh data", extra={"uuid": uuid})
        data = _update_profile_from_sources(uuid, current_name)
        return jsonify(data)
    else:
        # Name does not exist on Mojang, check if we have it cached as an old name
        log.info("Mojang check: name does not exist", extra={"username": username})

        with tx() as con:
            uuid_from_history = get_uuid_from_history_by_name(con, username)

        if uuid_from_history:
            log.info(
                "Found old profile in history cache",
                extra={"username": username, "uuid": uuid_from_history},
            )
            # We found an old record. Let's find the user's new name.
            new_current_name = get_profile_by_uuid_from_mojang(uuid_from_history)

            if new_current_name:
                log.info(
                    "Migrating profile to new name",
                    extra={"old_name": username, "new_name": new_current_name},
                )
                # Fetch fresh data for the new profile
                fresh_data = _update_profile_from_sources(
                    uuid_from_history, new_current_name
                )
                return jsonify(fresh_data)
            else:
                # The UUID is no longer valid, we cannot find the user.
                log.warning(
                    "UUID for old name no longer valid",
                    extra={"uuid": uuid_from_history},
                )
                abort(404, "User associated with this old name could not be found.")
        else:
            # No current Mojang profile and no cached profile
            abort(404, "Username not found")


@app.get("/api/namehistory/uuid/<uuid>")
def api_namehistory_by_uuid(uuid):
    uuid = dashed_uuid(uuid)
    current_name = get_profile_by_uuid_from_mojang(uuid)
    if not current_name:
        abort(404, "UUID not found")

    with tx() as con:
        if not is_source_stale(con, uuid, "scraper", SCRAPER_STALE_HOURS):
            log.info(
                "Cache is fresh for UUID, returning cached data", extra={"uuid": uuid}
            )
            return jsonify(query_history_public(con, uuid))

    log.info("Cache is stale for UUID, fetching fresh data", extra={"uuid": uuid})
    data = _update_profile_from_sources(uuid, current_name)
    return jsonify(data)


@app.post("/api/namehistory/update")
def api_update():
    try:
        body = request.get_json(force=True, silent=False) or {}
        if not isinstance(body, dict):
            abort(400, "JSON object expected")

        # Bulk update
        if "usernames" in body or "uuids" in body:
            usernames = body.get("usernames") or []
            uuids = body.get("uuids") or []
            if not isinstance(usernames, list) or not isinstance(uuids, list):
                abort(400, "'usernames' and 'uuids' must be arrays")

            usernames = [
                u.strip()
                for u in usernames
                if isinstance(u, str) and re.fullmatch(r"[A-Za-z0-9_]{1,16}", u.strip())
            ]
            uuids = [
                dashed_uuid(u)
                for u in uuids
                if isinstance(u, str)
                and re.fullmatch(r"[0-9a-fA-F-]{32,36}", u.strip())
            ]

            results = {"updated": [], "errors": []}

            for idx, name in enumerate(usernames, 1):
                log.info(
                    "Bulk username",
                    extra={"index": idx, "total": len(usernames), "username": name},
                )
                normalized = normalize_username(name)
                if not normalized:
                    results["errors"].append(
                        {"username": name, "error": "Username not found"}
                    )
                    log.warning("Bulk username not found", extra={"username": name})
                    continue

                try:
                    current_name, uuid = normalized
                    results["updated"].append(
                        _update_profile_from_sources(uuid, current_name)
                    )
                except Exception as e:
                    error_desc = (
                        e.description if isinstance(e, HTTPException) else str(e)
                    )
                    results["errors"].append({"username": name, "error": error_desc})
                    log.error(
                        "Bulk username failed",
                        extra={"username": name, "error": error_desc},
                    )
                bulk_sleep()

            for idx, u in enumerate(uuids, 1):
                log.info(
                    "Bulk uuid", extra={"index": idx, "total": len(uuids), "uuid": u}
                )
                current_name = get_profile_by_uuid_from_mojang(u)
                if not current_name:
                    results["errors"].append({"uuid": u, "error": "UUID not found"})
                    log.warning("Bulk UUID not found", extra={"uuid": u})
                    continue

                try:
                    results["updated"].append(
                        _update_profile_from_sources(u, current_name)
                    )
                except Exception as e:
                    error_desc = (
                        e.description if isinstance(e, HTTPException) else str(e)
                    )
                    results["errors"].append({"uuid": u, "error": error_desc})
                    log.error(
                        "Bulk uuid failed", extra={"uuid": u, "error": error_desc}
                    )
                bulk_sleep()
            return jsonify(results)

        # Single update
        elif "username" in body or "uuid" in body:
            username = body.get("username")
            uuid = body.get("uuid")
            if username:
                normalized = normalize_username(username)
                if not normalized:
                    abort(404, "Username not found")
                current_name, uuid = normalized
                data = _update_profile_from_sources(uuid, current_name)
                return jsonify(data)
            if uuid:
                current_name = get_profile_by_uuid_from_mojang(uuid)
                if not current_name:
                    abort(404, "UUID not found")
                data = _update_profile_from_sources(uuid, current_name)
                return jsonify(data)

        abort(
            400,
            "Provide 'username'/'uuid' for single update or 'usernames'/'uuids' for bulk update",
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        log.error("API error in update", exc_info=True)
        abort(500)


@app.post("/api/namehistory/refresh-all")
def api_refresh_all():
    """Trigger a refresh of all profiles over 24 hours"""
    try:
        with tx() as con:
            profiles = con.execute("SELECT uuid FROM profiles").fetchall()
            total = len(profiles)

            if total == 0:
                return jsonify({"message": "No profiles to update", "total": 0})

            seconds_in_day = 86400
            delay_per_profile = seconds_in_day / total

            current_time = datetime.now(timezone.utc)
            for idx, profile in enumerate(profiles):
                stale_time = (
                    current_time
                    - timedelta(hours=SCRAPER_STALE_HOURS + 1)
                    + timedelta(seconds=idx * delay_per_profile)
                )
                con.execute(
                    "INSERT OR REPLACE INTO source_updates (uuid, source, last_updated_at) VALUES (?, 'mojang', ?)",
                    (profile["uuid"], stale_time.isoformat()),
                )
                con.execute(
                    "INSERT OR REPLACE INTO source_updates (uuid, source, last_updated_at) VALUES (?, 'scraper', ?)",
                    (profile["uuid"], stale_time.isoformat()),
                )

            log.info(f"Scheduled {total} profiles for refresh over 24 hours")
            return jsonify(
                {
                    "message": f"Scheduled {total} profiles for gradual refresh",
                    "total": total,
                    "estimated_completion": "24 hours",
                    "delay_per_profile": f"{delay_per_profile:.2f}s",
                }
            )
    except Exception as e:
        log.error("API error in refresh-all", exc_info=True)
        abort(500)


def run_debugger(username: str):
    log.info(f"--- Running Debugger for {username} ---")
    try:
        html_a = fetch_profile_html(f"https://namemc.com/profile/{username}")
        if html_a:
            with open(f"debug_namemc_{username}.html", "w") as f:
                f.write(html_a)
            log.info(f"Saved debug_namemc_{username}.html")
    except Exception as e:
        log.error(f"Failed to fetch NameMC: {e}")
    try:
        normalized = normalize_username(username)
        if normalized:
            _, uuid = normalized
            if uuid:
                api_data = fetch_laby_api_data(uuid)
                with open(f"debug_laby_api_{username}.json", "w") as f:
                    json.dump(api_data, f, indent=2)
                log.info(f"Saved debug_laby_api_{username}.json")
    except Exception as e:
        log.error(f"Failed to fetch Laby.net: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft Name History API")
    parser.add_argument(
        "--dump-html",
        metavar="USERNAME",
        help="Run in debug mode to download raw HTML/API data for a user instead of starting the server.",
    )
    parser.add_argument(
        "command", nargs="?", help="Command to run: 'clean' to clean database"
    )
    args = parser.parse_args()

    if args.dump_html:
        run_debugger(args.dump_html)
    elif args.command == "clean":
        init_db()
        clean_database()
    else:
        init_db()
        log.info("DB initialized")

        if AUTO_UPDATE_ENABLED:
            update_thread = threading.Thread(target=auto_update_worker, daemon=True)
            update_thread.start()
            log.info("Auto-update worker enabled")

        log.info("Starting Flask", extra={"host": HOST, "port": PORT})
        app.run(host=HOST, port=PORT)