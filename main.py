#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import random
import hashlib
import threading
import argparse
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
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
from providers.http_client import create_scraper_session, fetch_with_fallback
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, abort
from werkzeug.exceptions import HTTPException

from minecraft_api import client_profile_models, register_minecraft_routes
from config_loader import load_config
from challenge_store import (
    ChallengeTokenStore,
    challenge_token_models,
    delete_token_from_db,
    load_tokens_from_db,
    persist_token_to_db,
)
from captcha_routes import register_captcha_routes
from votes_api import register_votes_routes
from skin_assets import (
    CapeAssetStore,
    SkinAssetStore,
    register_cape_routes,
    register_skin_routes,
    skin_public_urls,
)

import logging
from logging.handlers import RotatingFileHandler

# --- CONFIGURATION ---
CONFIG: Dict[str, Any] = load_config()

HOST = CONFIG["server"]["host"]
PORT = int(CONFIG["server"]["port"])
DB_TYPE = CONFIG["database"].get("type", "sqlite")
DB_PATH = CONFIG["database"].get("path", "namehistory.db")
DB_URL = CONFIG["database"].get("url", None)
DEFAULT_SLEEP = float(CONFIG["fetch"]["default_sleep"])
# Same-name entries whose timestamps differ by less than this are treated as
# the same change event reported with different clocks (sources disagree by
# minutes to weeks). Mojang enforces 30 days between changes and ~37 days
# before a dropped name is reusable, so any window below 30 days cannot
# swallow a genuine reuse.
FUZZY_WINDOW_DAYS = float(CONFIG["fetch"].get("fuzzy_window_days", 1))
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
HYPIXEL_API_KEY = str(CONFIG.get("providers", {}).get("hypixel_api_key", "") or "").strip()
CHALLENGE_TTL_MINUTES = int(CONFIG.get("providers", {}).get("challenge_ttl_minutes", 45))
VOTES_CONFIG = CONFIG.get("votes") or {}
SKIN_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    str(CONFIG.get("skins", {}).get("cache_dir", "skin_cache")),
)
LABY_TURNSTILE_SITE_KEY = str(
    CONFIG.get("providers", {}).get("laby_turnstile_site_key", "") or ""
).strip()
LABY_CHALLENGE_HEADER = str(
    CONFIG.get("providers", {}).get("laby_challenge_header", "X-Challenge-Token") or ""
).strip()
LABY_CHALLENGE_TOKEN = str(
    os.environ.get("LABY_CHALLENGE_TOKEN")
    or CONFIG.get("providers", {}).get("laby_challenge_token", "")
    or ""
).strip()

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

if not HYPIXEL_API_KEY:
    log.warning(
        "HYPIXEL_API_KEY not set — hypixel profile data will be unavailable. "
        "Set it in .env or providers.hypixel_api_key in config.toml"
    )

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


def check_api_auth():
    """Raise 401 if API key is configured and request is not authorized."""
    if not API_KEY:
        return
    scheme, token = _extract_auth_token()
    if (scheme == "API-Key" and token == API_KEY) or (
        ADMIN_KEY and scheme == "Admin-Key" and token == ADMIN_KEY
    ):
        return
    abort(401, description="API authorization required")


# --- HTTP & UTILS ---
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
]
scraper_session = create_scraper_session("chrome124")
skin_store = SkinAssetStore(
    Path(SKIN_CACHE_DIR),
    lambda url, **kwargs: scraper_session.get(url, **kwargs),
)
cape_store = CapeAssetStore(
    Path(SKIN_CACHE_DIR),
    lambda url, **kwargs: scraper_session.get(url, **kwargs),
)


def resolve_skin_content_key(
    skin_hash: str, file_hash: Optional[str] = None
) -> Optional[str]:
    """
    SHA256 of the decoded, normalized texture — a stable identity across
    sources that use incompatible hash schemes for the same pixels (Laby's
    MD5-style hash vs NameMC's opaque id). Backed by skin_store's on-disk
    cache, so repeat calls for an already-prefetched hash are local reads,
    not new downloads.
    """
    try:
        path = skin_store.ensure_skin(skin_hash, file_hash)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None
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
    """Parse a timestamp (ISO string or epoch seconds/millis) to an aware UTC
    datetime, return None if invalid or None."""
    if ts is None or ts == "":
        return None
    try:
        if isinstance(ts, (int, float)) or str(ts).strip().isdigit():
            value = float(ts)
            if value >= 1e12:  # epoch milliseconds
                value /= 1000.0
            return datetime.fromtimestamp(value, tz=timezone.utc)
        dt = dateutil_parser.isoparse(str(ts))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
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
        r = fetch_with_fallback(
            url,
            session=scraper_session,
            headers={"User-Agent": random.choice(UA_POOL)},
            timeout=15.0,
            allow=[200],
        )
        return r.text if r and r.status_code == 200 else ""
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


def fetch_laby_skin_data(uuid: str) -> List[Dict[str, Optional[str]]]:
    url = f"https://laby.net/api/user/{uuid}/get-textures"
    try:
        r = scraper_session.get(
            url, headers={"Accept": "application/json"}, timeout=10
        )
        if r.status_code != 200:
            return []
        return [
            {
                "skin_hash": e.get("image_hash"),
                "file_hash": e.get("file_hash"),
                "changedAt": e.get("first_seen_at"),
                "slim": e.get("slim_skin"),
                "source": "laby",
            }
            for e in r.json().get("SKIN", [])
            if e.get("image_hash")
        ]
    except Exception:
        return []


def _parse_namemc_skin_rows(soup: BeautifulSoup) -> List[Dict[str, Optional[str]]]:
    """Extract skin history from a NameMC profile or skins page."""
    out: List[Dict[str, Optional[str]]] = []
    seen: set[str] = set()

    def append_row(skin_hash: str, changed_at: Optional[str], slim: Optional[bool]):
        if not skin_hash or skin_hash in seen:
            return
        seen.add(skin_hash)
        out.append(
            {
                "skin_hash": skin_hash,
                "file_hash": None,
                "changedAt": changed_at,
                "slim": slim,
                "source": "namemc",
            }
        )

    skins_card = None
    for header in soup.find_all("div", class_="card-header"):
        if "Skins" in header.get_text():
            skins_card = header.find_parent("div", class_="card")
            break

    scope = skins_card or soup
    for link in scope.find_all("a", href=re.compile(r"^/skin/[a-f0-9]+$")):
        skin_hash = link["href"].strip("/").split("/")[-1]
        canvas = link.find("canvas", class_=re.compile(r"skin-2d|skin-button"))
        changed_at = None
        slim = None
        if canvas:
            changed_at = (canvas.get("title") or "").strip() or None
            model = (canvas.get("data-model") or "").lower()
            if model == "slim":
                slim = True
            elif model == "classic":
                slim = False
        append_row(skin_hash, changed_at, slim)

    if out:
        return out

    table = soup.select_one("table.table-borderless") or soup.select_one(
        "table.table-striped"
    )
    if not table:
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    for row in tbody.find_all("tr"):
        if "d-lg-none" in row.get("class", []):
            continue

        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        skin_hash = None
        slim = None
        for cell in cells:
            link = cell.find("a", href=re.compile(r"/skin/"))
            if link:
                skin_hash = link["href"].strip("/").split("/")[-1]
                img = cell.find("img")
                if img and "slim" in (img.get("alt", "") + img.get("title", "")).lower():
                    slim = True
                break

        changed_at = None
        for cell in cells:
            time_tag = cell.find("time")
            if time_tag and time_tag.get("datetime"):
                changed_at = time_tag["datetime"].strip()
                break

        if skin_hash:
            append_row(skin_hash, changed_at, slim)

    return out


def fetch_namemc_skin_data(username: str) -> List[Dict[str, Optional[str]]]:
    # NameMC removed /profile/{user}/skins; history lives on the main profile page.
    for url in (
        f"https://namemc.com/profile/{username}",
        f"https://namemc.com/minecraft-skins/profile/{username}.1",
        f"https://namemc.com/profile/{username}/skins",
    ):
        html = fetch_profile_html(url)
        if not html:
            continue
        rows = _parse_namemc_skin_rows(BeautifulSoup(html, "html.parser"))
        if rows:
            return rows
    return []


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
                # Laby marks approximated change dates with accurate=false;
                # those can be weeks off the real event.
                "accurate": bool(e.get("accurate", True)),
            }
            for e in r.json()
            if e.get("name")
        ]
    except Exception:
        return []


def fetch_crafty_name_data(username: str) -> List[Dict[str, Optional[str]]]:
    """
    Crafty.gg (crafty.gg/players/{username}.json) sometimes has a real
    changed_at for the very first name on an account, which Mojang never
    exposed and Laby/NameMC often report as null — useful to fill that gap.
    """
    try:
        r = scraper_session.get(
            f"https://crafty.gg/players/{username}.json",
            headers={"Accept": "application/json"},
            timeout=12,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, dict):
            return []
        return [
            {
                "name": e.get("username"),
                "changedAt": e.get("changed_at"),
                "source": "crafty",
            }
            for e in data.get("usernames") or []
            if isinstance(e, dict) and e.get("username")
        ]
    except Exception:
        return []


def gather_remote_rows_by_uuid(
    uuid: str, current_name: str
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch from Laby, NameMC, and Crafty.gg, return all results.
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
    jitter_sleep()

    # Fetch from Crafty.gg — mainly useful for filling in the original
    # name's date, which Laby/NameMC frequently report as null.
    log.debug(f"Fetching from Crafty.gg for username {current_name}")
    crafty_rows = fetch_crafty_name_data(current_name)
    log.debug(f"Crafty.gg returned {len(crafty_rows)} entries")
    all_rows.extend(crafty_rows)

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
                    "accurate": bool(row.get("accurate", True)),
                }
            )

    log.debug(f"Processing {len(normalized_entries)} entries from all sources")

    # Step 2: Group by name to detect conflicting null vs dated timestamps
    by_name = defaultdict(list)
    for entry in normalized_entries:
        by_name[entry["name"].lower()].append(entry)

    # Step 3: For each name with BOTH null and dated timestamps, decide what
    # the null means. If the name's earliest dated entry is the earliest date
    # in the whole history, the dated entry describes the same original-name
    # event and the null is incomplete data — drop it. Otherwise another name
    # predates it, so the null is the original name and the dated entry is a
    # genuine reuse — keep both.
    all_dated = [e["datetime"] for e in normalized_entries if e["datetime"]]
    earliest_overall = min(all_dated) if all_dated else None

    cleaned_entries = []
    for name_lower, entries in by_name.items():
        nulls = [e for e in entries if e["timestamp"] is None]
        dated = [e for e in entries if e["timestamp"] is not None]

        if nulls and dated and min(e["datetime"] for e in dated) == earliest_overall:
            log.debug(
                f"  Name '{entries[0]['name']}' has both null and dated timestamps "
                "and is the oldest name - discarding null as incomplete"
            )
            cleaned_entries.extend(dated)
        else:
            cleaned_entries.extend(entries)

    log.debug(f"After removing incomplete nulls: {len(cleaned_entries)} entries")

    # Step 4: Sort chronologically (None first, then by datetime)
    cleaned_entries.sort(
        key=lambda e: e["datetime"]
        or datetime.min.replace(tzinfo=timezone.utc)
    )

    # Step 5: Collapse entries that describe the same change event. Sources
    # report the same change with different clocks (seconds to weeks apart),
    # so besides consecutive duplicates we also merge any same-name entries
    # within FUZZY_WINDOW_DAYS of each other — Mojang's 30-day change cooldown
    # means a genuine reuse can never fall inside that window. When merging,
    # an accurate timestamp beats an approximated one; among equals the
    # earlier wins (approximations always trail the real event).
    fuzzy_window = timedelta(days=FUZZY_WINDOW_DAYS)
    merged: List[Dict[str, Any]] = []

    def same_event(existing: Dict[str, Any], entry: Dict[str, Any], consecutive: bool) -> bool:
        if existing["name"].lower() != entry["name"].lower():
            return False
        if consecutive:
            return True
        prev_dt, curr_dt = existing["datetime"], entry["datetime"]
        return (
            prev_dt is not None
            and curr_dt is not None
            and curr_dt - prev_dt <= fuzzy_window
        )

    def better_estimate(candidate: Dict[str, Any], existing: Dict[str, Any]) -> bool:
        if candidate["accurate"] != existing["accurate"]:
            return candidate["accurate"]
        cand_dt, exist_dt = candidate["datetime"], existing["datetime"]
        return bool(cand_dt and exist_dt and cand_dt < exist_dt)

    for entry in cleaned_entries:
        match_idx = None
        for i in range(len(merged) - 1, -1, -1):
            if same_event(merged[i], entry, consecutive=(i == len(merged) - 1)):
                match_idx = i
                break

        if match_idx is not None:
            existing = merged[match_idx]
            if better_estimate(entry, existing):
                log.debug(
                    f"  Replacing {existing['name']} @ {existing['timestamp']} with "
                    f"{entry['timestamp']} (from {entry['source']}, accurate={entry['accurate']})"
                )
                merged[match_idx] = entry
            else:
                log.debug(
                    f"  Skipping duplicate event {entry['name']} @ {entry['timestamp']} (from {entry['source']})"
                )
            continue

        log.debug(f"  Adding {entry['name']} @ {entry['timestamp']} (from {entry['source']})")
        merged.append(entry)

    # Timestamps may have moved during merging; restore chronological order.
    merged.sort(
        key=lambda e: e["datetime"] or datetime.min.replace(tzinfo=timezone.utc)
    )
    result = [(e["name"], e["timestamp"]) for e in merged]

    # Step 6: Ensure current name is in the list
    if not result or result[-1][0].lower() != current_name.lower():
        log.debug(f"  Adding current name '{current_name}'")
        result.append((current_name, None))

    log.info(f"Final merged result: {len(result)} unique name changes")

    return result


def gather_remote_skin_rows_by_uuid(
    uuid: str, current_name: str
) -> List[Dict[str, Optional[str]]]:
    all_rows = []

    log.debug(f"Fetching skin data from Laby API for UUID {uuid}")
    laby_rows = fetch_laby_skin_data(uuid)
    log.debug(f"Laby returned {len(laby_rows)} skin entries")
    all_rows.extend(laby_rows)
    jitter_sleep()

    log.debug(f"Fetching skin data from NameMC for username {current_name}")
    namemc_rows = fetch_namemc_skin_data(current_name)
    log.debug(f"NameMC returned {len(namemc_rows)} skin entries")
    all_rows.extend(namemc_rows)

    return all_rows


def get_current_skin_from_laby(uuid: str) -> Optional[Tuple[str, Optional[bool]]]:
    try:
        r = scraper_session.get(
            f"https://laby.net/api/user/{uuid}/get-textures",
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        skins = r.json().get("SKIN", [])
        for entry in skins:
            if entry.get("active") and entry.get("image_hash"):
                return entry["image_hash"], entry.get("slim_skin")
        if skins:
            latest = max(
                skins,
                key=lambda e: parse_timestamp(e.get("last_seen_at"))
                or datetime.min.replace(tzinfo=timezone.utc),
            )
            if latest.get("image_hash"):
                return latest["image_hash"], latest.get("slim_skin")
    except Exception:
        return None
    return None


def merge_remote_skin_sources(
    rows: List[Dict[str, Optional[str]]],
    current_skin_hash: str,
    resolve_content_key: Optional[Callable[[str, Optional[str]], Optional[str]]] = None,
) -> List[Tuple[str, Optional[str], Optional[str], Optional[bool]]]:
    """
    `resolve_content_key` maps (skin_hash, file_hash) -> a stable identity for
    the actual pixel content (e.g. a hash of the decoded texture), when it can
    be resolved. Laby (MD5-style hash) and NameMC (opaque id) use incompatible
    identifiers for the exact same texture, and their crawl timestamps for the
    same real change can differ by weeks — far more than any fixed time
    window — so without real content identity the same skin gets recorded
    twice under two different "hashes". Falls back to the source-reported
    hash string when the resolver is omitted or fails for a given entry.
    """

    def content_key(skin_hash: str, file_hash: Optional[str]) -> Tuple[str, bool]:
        """Returns (key, verified) — verified means the key came from actual
        pixel-content resolution, not a same-string fallback."""
        if resolve_content_key:
            resolved = resolve_content_key(skin_hash, file_hash)
            if resolved:
                return resolved, True
        return skin_hash.lower(), False

    normalized_entries = []
    for row in rows:
        skin_hash = row.get("skin_hash")
        changed_at = row.get("changedAt")
        source = row.get("source", "unknown")
        file_hash = row.get("file_hash")

        if skin_hash:
            normalized_ts = normalize_timestamp(changed_at)
            key, verified = content_key(skin_hash, file_hash)
            normalized_entries.append(
                {
                    "skin_hash": skin_hash,
                    "file_hash": file_hash,
                    "slim": row.get("slim"),
                    "timestamp": normalized_ts,
                    "datetime": parse_timestamp(changed_at),
                    "source": source,
                    "content_key": key,
                    "content_verified": verified,
                }
            )

    log.debug(f"Processing {len(normalized_entries)} skin entries from all sources")

    # Same null-vs-dated resolution as name history: only drop a content
    # key's null timestamp if its earliest dated occurrence is also the
    # earliest dated occurrence overall — otherwise the null means this skin
    # was worn again later without a source telling us when (a genuine
    # reuse), not that the dated occurrence's timestamp is incomplete data
    # for the same event.
    by_key = defaultdict(list)
    for entry in normalized_entries:
        by_key[entry["content_key"]].append(entry)

    all_dated = [e["datetime"] for e in normalized_entries if e["datetime"]]
    earliest_overall = min(all_dated) if all_dated else None

    cleaned_entries = []
    for entries in by_key.values():
        nulls = [e for e in entries if e["timestamp"] is None]
        dated = [e for e in entries if e["timestamp"] is not None]

        if nulls and dated and min(e["datetime"] for e in dated) == earliest_overall:
            cleaned_entries.extend(dated)
        else:
            cleaned_entries.extend(entries)

    cleaned_entries.sort(
        key=lambda e: e["datetime"]
        or datetime.min.replace(tzinfo=timezone.utc)
    )

    # Collapse entries describing the same skin-change event: within
    # FUZZY_WINDOW_DAYS of each other is one signal, but cross-source crawl
    # drift on the exact same content key can span weeks, so we also treat
    # same-content-key entries as one event whenever no *different* content
    # key's occurrence falls between them (you can't have "wear it, wear
    # something else, wear it again" without that middle event showing up
    # somewhere). Among matches the earliest timestamp wins.
    fuzzy_window = timedelta(days=FUZZY_WINDOW_DAYS)

    def has_intermediate(key: str, dt_a: datetime, dt_b: datetime) -> bool:
        lo, hi = min(dt_a, dt_b), max(dt_a, dt_b)
        return any(
            e["content_key"] != key and e["datetime"] is not None and lo < e["datetime"] < hi
            for e in cleaned_entries
        )

    merged: List[Dict[str, Any]] = []

    def same_event(existing: Dict[str, Any], entry: Dict[str, Any], consecutive: bool) -> bool:
        if existing["content_key"] != entry["content_key"]:
            return False
        # A verified pixel-content match is authoritative — Laby/NameMC crawl
        # dates for the exact same texture can drift by weeks, so no time-gap
        # or intermediate-activity heuristic should override proven identity.
        if existing["content_verified"] and entry["content_verified"]:
            return True
        if consecutive:
            return True
        prev_dt, curr_dt = existing["datetime"], entry["datetime"]
        if prev_dt is None or curr_dt is None:
            return False
        if abs(curr_dt - prev_dt) <= fuzzy_window:
            return True
        return not has_intermediate(entry["content_key"], prev_dt, curr_dt)

    for entry in cleaned_entries:
        match_idx = None
        for i in range(len(merged) - 1, -1, -1):
            if same_event(merged[i], entry, consecutive=(i == len(merged) - 1)):
                match_idx = i
                break

        if match_idx is not None:
            existing = merged[match_idx]
            cand_dt, exist_dt = entry["datetime"], existing["datetime"]
            if cand_dt and exist_dt and cand_dt < exist_dt:
                log.debug(
                    f"  Replacing skin {existing['skin_hash']} @ {existing['timestamp']} with "
                    f"earlier {entry['timestamp']} (from {entry['source']})"
                )
                merged[match_idx] = entry
            else:
                log.debug(
                    f"  Skipping duplicate skin event {entry['skin_hash']} @ {entry['timestamp']} (from {entry['source']})"
                )
            # Backfill file_hash/slim from whichever entry supplied them.
            if not merged[match_idx]["file_hash"] and entry["file_hash"]:
                merged[match_idx]["file_hash"] = entry["file_hash"]
            if merged[match_idx]["slim"] is None and entry["slim"] is not None:
                merged[match_idx]["slim"] = entry["slim"]
            # Prefer a 32-char (Laby) hash for display/rendering.
            if len(entry["skin_hash"]) == 32 and len(merged[match_idx]["skin_hash"]) != 32:
                merged[match_idx]["skin_hash"] = entry["skin_hash"]
            continue

        log.debug(f"  Adding skin {entry['skin_hash']} @ {entry['timestamp']} (from {entry['source']})")
        merged.append(dict(entry))

    merged.sort(
        key=lambda e: e["datetime"] or datetime.min.replace(tzinfo=timezone.utc)
    )
    result = [
        (e["skin_hash"], e["timestamp"], e["file_hash"], e["slim"]) for e in merged
    ]

    if result:
        current_key, _ = content_key(current_skin_hash, None)
        existing_keys = {content_key(h, fh)[0] for h, _, fh, _ in result}
        if current_key not in existing_keys:
            log.debug(f"  Adding current skin '{current_skin_hash}' (not in history)")
            result.append((current_skin_hash, None, None, None))
    else:
        log.debug(f"  Adding current skin '{current_skin_hash}' (empty history)")
        result.append((current_skin_hash, None, None, None))

    log.info(f"Final merged skin result: {len(result)} unique skin changes")

    return result


# --- DATABASE ---
Base = declarative_base()

ClientProfileSnapshot, ClientCosmeticSnapshot = client_profile_models(Base)
ChallengeToken = challenge_token_models(Base)
challenge_store = ChallengeTokenStore(default_ttl_minutes=CHALLENGE_TTL_MINUTES)
if LABY_CHALLENGE_TOKEN:
    challenge_store.set("labymod", LABY_CHALLENGE_TOKEN, source="config")


def sync_challenge_tokens() -> None:
    with tx() as session:
        load_tokens_from_db(session, ChallengeToken, challenge_store)


def invalidate_challenge_token(provider: str) -> None:
    challenge_store.clear(provider)
    with tx() as session:
        delete_token_from_db(session, ChallengeToken, provider)


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
    skin_history = relationship(
        "SkinHistory",
        back_populates="profile",
        cascade="all, delete-orphan",
    )
    client_profiles = relationship(
        "ClientProfileSnapshot",
        back_populates="profile",
        cascade="all, delete-orphan",
    )
    client_cosmetics = relationship(
        "ClientCosmeticSnapshot",
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


class SkinHistory(Base):
    __tablename__ = "skin_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(
        String(36),
        ForeignKey("profiles.uuid", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    skin_hash = Column(String(64), nullable=False, index=True)
    file_hash = Column(String(64), index=True)
    slim = Column(Integer)
    changed_at = Column(String(32), index=True)
    observed_at = Column(DateTime, nullable=False)
    profile = relationship("Profile", back_populates="skin_history")
    __table_args__ = (UniqueConstraint("uuid", "skin_hash", "changed_at"),)


class SourceUpdate(Base):
    __tablename__ = "source_updates"
    uuid = Column(
        String(36), ForeignKey("profiles.uuid", ondelete="CASCADE"), primary_key=True
    )
    source = Column(String(32), primary_key=True)
    last_updated_at = Column(DateTime, nullable=False, index=True)
    profile = relationship("Profile", back_populates="source_updates")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ip_address = Column(String(45), nullable=False, index=True)  # Support IPv6
    username = Column(String(32), nullable=True, index=True)
    first_seen_at = Column(DateTime, nullable=False)
    last_seen_at = Column(DateTime, nullable=False)
    requests = relationship("Request", back_populates="user", cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint("ip_address", "username"),)


class Request(Base):
    __tablename__ = "requests"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    requested_username = Column(String(32), nullable=False, index=True)
    source = Column(String(50), nullable=True)
    version = Column(String(20), nullable=True)
    mc_version = Column(String(20), nullable=True)
    endpoint = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    response_status = Column(Integer, nullable=False)
    user = relationship("User", back_populates="requests")


db_url = f"sqlite:///{DB_PATH}" if DB_TYPE == "sqlite" else DB_URL
engine = create_engine(db_url, echo=False, future=True)
SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False)
)


def init_db():
    """Initialize database with all tables, creating missing ones automatically."""
    # Create all tables that don't exist yet
    Base.metadata.create_all(bind=engine)

    # Check if we need to add the new logging tables to existing databases
    with engine.connect() as conn:
        # Check if the new tables exist
        try:
            result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            users_exists = result.fetchone() is not None
        except Exception:
            users_exists = False

        try:
            result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'")
            requests_exists = result.fetchone() is not None
        except Exception:
            requests_exists = False

        try:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='client_profile_snapshots'"
            )
            client_profiles_exists = result.fetchone() is not None
        except Exception:
            client_profiles_exists = False

    if users_exists and requests_exists:
        log.info("Database already contains logging tables (users, requests)")
    else:
        log.info("Database migration completed - added logging tables (users, requests)")

    if not client_profiles_exists:
        log.info("Database migration completed - added client profile tables")


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


def resolve_player(
    username: str, uuid_param: str
) -> Tuple[Optional[str], str, Optional[str]]:
    """Resolve UUID, query username, and current Mojang name."""
    uuid = dashed_uuid(uuid_param) if uuid_param else None
    if not uuid and username:
        if not re.fullmatch(r"[A-Za-z0-9_]{1,16}", username):
            abort(400, "username required")

        normalized = normalize_username(username)
        if normalized:
            _, uuid = normalized
        else:
            with tx() as s:
                uuid = get_uuid_from_history_by_name(s, username)

    if not uuid:
        return None, username, None

    current_name = get_profile_by_uuid_from_mojang(uuid)
    if not current_name and not username:
        return None, username, None

    return uuid, username or current_name or "", current_name


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

    # Sort chronologically: None (original name) first, then by parsed
    # timestamp. Parsing (rather than string comparison) keeps rows stored
    # in older formats (epoch millis, trailing "Z") in the right order.
    def sort_key(r):
        return parse_timestamp(r.changed_at) or datetime.min.replace(
            tzinfo=timezone.utc
        )

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


def delete_profile(session, uuid: str) -> bool:
    """Deletes a profile and its history by UUID."""
    profile = session.get(Profile, uuid)
    if profile:
        session.delete(profile)
        log.info(f"Deleted profile for UUID {uuid}")
        return True
    return False


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

        # Dedupe against existing rows fuzzily: the same change event can be
        # stored with a slightly different timestamp (source clock drift,
        # bad estimates, format changes across app versions), so exact string
        # matching accumulates near-duplicate rows over time. Two same-name
        # rows are the same event when their timestamps are close OR when no
        # other name's change falls between them — you can't change a name to
        # itself, so a genuine reuse always has an intermediate name.
        fuzzy_window = timedelta(days=FUZZY_WINDOW_DAYS)
        existing_rows = list(con.query(History).filter_by(uuid=uuid).all())

        timeline = [
            (r.name.lower(), parse_timestamp(r.changed_at)) for r in existing_rows
        ] + [(n.lower(), parse_timestamp(c)) for n, c in pairs]

        def has_intermediate_change(name: str, dt_a, dt_b) -> bool:
            lo, hi = min(dt_a, dt_b), max(dt_a, dt_b)
            return any(
                other_name != name.lower() and other_dt is not None and lo < other_dt < hi
                for other_name, other_dt in timeline
            )

        def find_same_event(name: str, new_dt):
            for r in existing_rows:
                if r.name.lower() != name.lower():
                    continue
                r_dt = parse_timestamp(r.changed_at)
                if new_dt is None:
                    # Incomplete/current-name marker: any same-name row
                    # already covers it.
                    return r
                if r_dt is None:
                    # Existing row is the original-name marker (no change
                    # date exists for it); a dated entry is a separate
                    # (reuse) event.
                    continue
                if abs(new_dt - r_dt) <= fuzzy_window:
                    return r
                if not has_intermediate_change(name, new_dt, r_dt):
                    return r
            return None

        inserted = 0
        updated = 0
        for name, changed_at in pairs:
            new_dt = parse_timestamp(changed_at)
            match = find_same_event(name, new_dt)
            if match is not None:
                # Same event already recorded — converge on the earlier
                # timestamp (approximations always trail the real event).
                match_dt = parse_timestamp(match.changed_at)
                if new_dt and match_dt and new_dt < match_dt:
                    match.changed_at = changed_at
                    updated += 1
                continue
            row = History(
                uuid=uuid,
                name=name,
                changed_at=changed_at,
                observed_at=now,
            )
            con.add(row)
            existing_rows.append(row)
            inserted += 1
        log.info(
            f"Inserted {inserted} new history rows, refined {updated} timestamps (preserved existing)."
        )

        update_source_timestamp(con, uuid, "scraper")
        con.flush()  # Ensure changes are visible to the query
        return query_history_public(con, uuid)


def query_skin_history_public(session, uuid: str) -> Dict[str, Any]:
    profile = session.get(Profile, uuid)
    if not profile:
        return {
            "query": None,
            "uuid": uuid,
            "last_seen_at": None,
            "history": [],
        }

    rows = session.query(SkinHistory).filter_by(uuid=uuid).all()

    def sort_key(r):
        return parse_timestamp(r.changed_at) or datetime.min.replace(
            tzinfo=timezone.utc
        )

    rows.sort(key=sort_key)

    history = []
    for i, r in enumerate(rows):
        entry = {
            "id": i + 1,
            "skin_hash": r.skin_hash,
            "file_hash": r.file_hash,
            "slim": bool(r.slim) if r.slim is not None else None,
            "changed_at": r.changed_at,
            "observed_at": r.observed_at.isoformat(),
        }
        entry.update(skin_public_urls(r.skin_hash))
        history.append(entry)
    return {
        "query": profile.query,
        "uuid": profile.uuid,
        "last_seen_at": profile.last_seen_at.isoformat(),
        "history": history,
    }


def _update_skin_profile_from_sources(
    uuid: str, current_name: str
) -> Dict[str, Any]:
    log.info(
        f"Updating skin history from sources: uuid={uuid}, current_name={current_name}"
    )

    rows = gather_remote_skin_rows_by_uuid(uuid, current_name)
    log.debug(f"Total fetched: {len(rows)} skin entries from all sources")

    current_skin = get_current_skin_from_laby(uuid)
    if not current_skin:
        dated_rows = [r for r in rows if r.get("changedAt")]
        if dated_rows:
            dated_rows.sort(
                key=lambda r: parse_timestamp(r.get("changedAt"))
                or datetime.min.replace(tzinfo=timezone.utc)
            )
            latest = dated_rows[-1]
            current_skin_hash = latest["skin_hash"]
        elif rows:
            current_skin_hash = rows[0]["skin_hash"]
        else:
            current_skin_hash = ""
    else:
        current_skin_hash, _ = current_skin

    if not current_skin_hash:
        log.warning(f"No current skin found for UUID {uuid}")
        with tx() as con:
            ensure_profile(con, uuid, current_name)
            update_source_timestamp(con, uuid, "skin_scraper")
            return query_skin_history_public(con, uuid)

    pairs = merge_remote_skin_sources(rows, current_skin_hash, resolve_skin_content_key)
    log.info(f"Merged into {len(pairs)} final skin history entries")

    for skin_hash, _, file_hash, _ in pairs:
        skin_store.prefetch(skin_hash, file_hash)

    with tx() as con:
        ensure_profile(con, uuid, current_name)
        now = datetime.now(timezone.utc)

        fuzzy_window = timedelta(days=FUZZY_WINDOW_DAYS)
        existing_rows = list(con.query(SkinHistory).filter_by(uuid=uuid).all())

        content_key_cache: Dict[Tuple[str, Optional[str]], Tuple[str, bool]] = {}

        def row_content_key(skin_hash: str, file_hash: Optional[str]) -> str:
            return row_content_key_verified(skin_hash, file_hash)[0]

        def row_content_key_verified(skin_hash: str, file_hash: Optional[str]) -> Tuple[str, bool]:
            cache_key = (skin_hash.lower(), (file_hash or "").lower())
            if cache_key not in content_key_cache:
                resolved = resolve_skin_content_key(skin_hash, file_hash)
                content_key_cache[cache_key] = (
                    (resolved, True) if resolved else (skin_hash.lower(), False)
                )
            return content_key_cache[cache_key]

        # One-time retroactive cleanup: collapse any existing rows that are
        # already duplicates under content-key identity (profiles refreshed
        # before this fix could accumulate one row per source per skin, since
        # Laby and NameMC use incompatible hash schemes for the same texture
        # and their crawl timestamps for it can differ by weeks).
        existing_by_key: Dict[str, List[Any]] = defaultdict(list)
        for r in existing_rows:
            existing_by_key[row_content_key(r.skin_hash, r.file_hash)].append(r)

        deduped_existing: List[Any] = []
        removed = 0
        for key, group in existing_by_key.items():
            if len(group) == 1:
                deduped_existing.append(group[0])
                continue
            dated = [r for r in group if r.changed_at]
            canonical = min(dated, key=lambda r: parse_timestamp(r.changed_at)) if dated else group[0]
            for r in group:
                if r is canonical:
                    continue
                if r.file_hash and not canonical.file_hash:
                    canonical.file_hash = r.file_hash
                if r.slim is not None and canonical.slim is None:
                    canonical.slim = r.slim
                log.info(
                    f"  Removing duplicate skin row {r.skin_hash} @ {r.changed_at} "
                    f"(same content as {canonical.skin_hash} @ {canonical.changed_at})"
                )
                con.delete(r)
                removed += 1
            deduped_existing.append(canonical)
        existing_rows = deduped_existing
        if removed:
            log.info(f"Collapsed {removed} pre-existing duplicate skin rows for uuid={uuid}")
            # Force the deletes to hit the DB now — otherwise SQLAlchemy can
            # batch a later INSERT of a row sharing the same (uuid, skin_hash,
            # changed_at) before the DELETE actually executes, violating the
            # unique constraint even though the row is logically gone.
            con.flush()

        timeline = [
            (row_content_key(r.skin_hash, r.file_hash), parse_timestamp(r.changed_at))
            for r in existing_rows
        ] + [(row_content_key(h, fh), parse_timestamp(c)) for h, c, fh, _ in pairs]

        def has_intermediate_change(key: str, dt_a, dt_b) -> bool:
            lo, hi = min(dt_a, dt_b), max(dt_a, dt_b)
            return any(
                other_key != key and other_dt is not None and lo < other_dt < hi
                for other_key, other_dt in timeline
            )

        def find_same_event(key: str, new_verified: bool, new_dt):
            for r in existing_rows:
                r_key, r_verified = row_content_key_verified(r.skin_hash, r.file_hash)
                if r_key != key:
                    continue
                if new_verified and r_verified:
                    # Proven pixel identity overrides any time-gap heuristic —
                    # crawl dates for the same texture can drift by weeks.
                    return r
                r_dt = parse_timestamp(r.changed_at)
                if new_dt is None:
                    return r
                if r_dt is None:
                    continue
                if abs(new_dt - r_dt) <= fuzzy_window:
                    return r
                if not has_intermediate_change(key, new_dt, r_dt):
                    return r
            return None

        inserted = 0
        updated = 0
        for skin_hash, changed_at, file_hash, slim in pairs:
            slim_value = None
            if slim is True:
                slim_value = 1
            elif slim is False:
                slim_value = 0

            key, verified = row_content_key_verified(skin_hash, file_hash)
            new_dt = parse_timestamp(changed_at)
            match = find_same_event(key, verified, new_dt)
            if match is not None:
                match_dt = parse_timestamp(match.changed_at)
                if new_dt and match_dt and new_dt < match_dt:
                    match.changed_at = changed_at
                    updated += 1
                if file_hash and not match.file_hash:
                    match.file_hash = file_hash
                if slim_value is not None and match.slim is None:
                    match.slim = slim_value
                continue

            row = SkinHistory(
                uuid=uuid,
                skin_hash=skin_hash,
                file_hash=file_hash,
                slim=slim_value,
                changed_at=changed_at,
                observed_at=now,
            )
            con.add(row)
            existing_rows.append(row)
            inserted += 1
        log.info(
            f"Inserted {inserted} new skin history rows, refined {updated} timestamps, "
            f"removed {removed} duplicates (preserved existing)."
        )

        update_source_timestamp(con, uuid, "skin_scraper")
        con.flush()
        return query_skin_history_public(con, uuid)


app = Flask(__name__)

register_skin_routes(app, skin_store)
register_cape_routes(app, cape_store)


@app.route("/api/namehistory", methods=["GET", "POST", "DELETE"])
def api_namehistory():
    if request.method == 'DELETE':
        return api_delete()
    
    username = request.args.get("username", "").strip()
    uuid_param = request.args.get("uuid", "").strip()
    source = request.args.get("source")
    version = request.args.get("version")
    req_name = request.args.get("req_name")
    mc_version = request.args.get("mc_version")

    if not username and not uuid_param:
        return api_docs_html_route()

    uuid = dashed_uuid(uuid_param) if uuid_param else None
    if not uuid and username:
        if not re.fullmatch(r"[A-Za-z0-9_]{1,16}", username):
            abort(400, "username required")

        normalized = normalize_username(username)
        if normalized:
            _, uuid = normalized
        else:
            with tx() as s:
                uuid = get_uuid_from_history_by_name(s, username)

    if not uuid:
        abort(404, "Profile not found")

    current_name = get_profile_by_uuid_from_mojang(uuid)
    if not current_name:
        abort(404, "Profile not found")

    response_data = None
    with tx() as s:
        if not is_source_stale(s, uuid, "scraper", SCRAPER_STALE_HOURS):
            response_data = query_history_public(s, uuid)
        else:
            response_data = _update_profile_from_sources(uuid, current_name)

    # Log the request
    ip_address = (
        request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        .split(",")[0]
        .strip()
    )
    requester_username = req_name.strip() if req_name and req_name.strip() else None
    requested_username = username or uuid

    try:
        with tx() as s:
            user = None
            if requester_username:
                user = s.query(User).filter_by(ip_address=ip_address, username=requester_username).first()
            if not user:
                user = s.query(User).filter_by(ip_address=ip_address).first()
            now = datetime.now(timezone.utc)
            if not user:
                user = User(
                    ip_address=ip_address,
                    username=requester_username,
                    first_seen_at=now,
                    last_seen_at=now
                )
                s.add(user)
                s.flush([user])
            else:
                user.last_seen_at = now
                if requester_username and not user.username:
                    user.username = requester_username

            request_log = Request(
                user_id=user.id,
                requested_username=requested_username,
                source=source,
                version=version,
                mc_version=mc_version,
                endpoint=request.endpoint or "/api/namehistory",
                timestamp=now,
                response_status=200
            )
            s.add(request_log)
    except Exception as e:
        log.error(f"Failed to log request: {e}")

    return jsonify(response_data)


@app.route("/api/skinhistory", methods=["GET", "POST", "DELETE"])
def api_skinhistory():
    if request.method == "DELETE":
        return api_skin_delete()

    username = request.args.get("username", "").strip()
    uuid_param = request.args.get("uuid", "").strip()
    source = request.args.get("source")
    version = request.args.get("version")
    req_name = request.args.get("req_name")
    mc_version = request.args.get("mc_version")

    if not username and not uuid_param:
        return api_skin_docs_html_route()

    uuid = dashed_uuid(uuid_param) if uuid_param else None
    if not uuid and username:
        if not re.fullmatch(r"[A-Za-z0-9_]{1,16}", username):
            abort(400, "username required")

        normalized = normalize_username(username)
        if normalized:
            _, uuid = normalized
        else:
            with tx() as s:
                uuid = get_uuid_from_history_by_name(s, username)

    if not uuid:
        abort(404, "Profile not found")

    current_name = get_profile_by_uuid_from_mojang(uuid)
    if not current_name:
        abort(404, "Profile not found")

    force_refresh = request.args.get("refresh", "").lower() in {"1", "true", "yes"}
    response_data = None
    with tx() as s:
        if (
            not force_refresh
            and not is_source_stale(s, uuid, "skin_scraper", SCRAPER_STALE_HOURS)
        ):
            response_data = query_skin_history_public(s, uuid)
        else:
            response_data = _update_skin_profile_from_sources(uuid, current_name)

    ip_address = (
        request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        .split(",")[0]
        .strip()
    )
    requester_username = req_name.strip() if req_name and req_name.strip() else None
    requested_username = username or uuid

    try:
        with tx() as s:
            log_request(
                session=s,
                ip_address=ip_address,
                requester_username=requester_username,
                requested_username=requested_username,
                source=source,
                version=version,
                endpoint=request.endpoint or "/api/skinhistory",
                response_status=200,
                mc_version=mc_version,
            )
    except Exception as e:
        log.error(f"Failed to log request: {e}")

    return jsonify(response_data)


@app.route("/api/skinhistory/delete", methods=["DELETE"])
@require_auth(AuthLevel.ADMIN)
def api_skin_delete():
    username = request.args.get("username", "").strip()
    uuid_param = request.args.get("uuid", "").strip()
    source = request.args.get("source")
    version = request.args.get("version")
    req_name = request.args.get("req_name")
    mc_version = request.args.get("mc_version")

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
        profile = s.get(Profile, uuid)
        if not profile:
            abort(404, "Profile not found in database")
        s.query(SkinHistory).filter_by(uuid=uuid).delete()
        s.query(SourceUpdate).filter_by(uuid=uuid, source="skin_scraper").delete()

    ip_address = (
        request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        .split(",")[0]
        .strip()
    )
    requester_username = req_name.strip() if req_name and req_name.strip() else (
        username or None
    )
    requested_username = username or uuid

    try:
        with tx() as s:
            log_request(
                session=s,
                ip_address=ip_address,
                requester_username=requester_username,
                requested_username=requested_username,
                source=source,
                version=version,
                endpoint=request.endpoint or "/api/skinhistory/delete",
                response_status=200,
                mc_version=mc_version,
            )
    except Exception as e:
        log.error(f"Failed to log request: {e}")

    return jsonify({"message": "Skin history deleted", "uuid": uuid})


@app.route("/api/skinhistory/update", methods=["POST"])
@require_auth(AuthLevel.API)
def api_skin_update():
    body = request.get_json(force=True, silent=False) or {}
    source = request.args.get("source")
    version = request.args.get("version")
    req_name = request.args.get("req_name")
    mc_version = request.args.get("mc_version")

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
                _update_skin_profile_from_sources(norm[1], norm[0])
            )
            bulk_sleep()

    for u in uuids:
        u = dashed_uuid(u)
        name = get_profile_by_uuid_from_mojang(u)
        if not name:
            results["errors"].append({"uuid": u, "error": "Not found"})
        else:
            results["updated"].append(_update_skin_profile_from_sources(u, name))
            bulk_sleep()

    ip_address = (
        request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        .split(",")[0]
        .strip()
    )
    requester_username = req_name.strip() if req_name and req_name.strip() else None
    requested_username = usernames[0] if usernames else (
        uuids[0] if uuids else "batch_skin_update"
    )

    try:
        with tx() as s:
            log_request(
                session=s,
                ip_address=ip_address,
                requester_username=requester_username,
                requested_username=requested_username,
                source=source,
                version=version,
                endpoint=request.endpoint or "/api/skinhistory/update",
                response_status=200,
                mc_version=mc_version,
            )
    except Exception as e:
        log.error(f"Failed to log request: {e}")

    return jsonify(results)


@app.route("/api/namehistory/delete", methods=["DELETE"])
@require_auth(AuthLevel.ADMIN)
def api_delete():
    username = request.args.get("username", "").strip()
    uuid_param = request.args.get("uuid", "").strip()
    source = request.args.get("source")
    version = request.args.get("version")
    req_name = request.args.get("req_name")
    mc_version = request.args.get("mc_version")

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

    response_data = None
    with tx() as s:
        if not delete_profile(s, uuid):
            abort(404, "Profile not found in database")
        response_data = {"message": "Profile deleted", "uuid": uuid}

    # Log the request
    ip_address = (
        request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        .split(",")[0]
        .strip()
    )

    requester_username = req_name.strip() if req_name and req_name.strip() else (username or None)
    requested_username = username or uuid

    try:
        with tx() as s:
            user = None
            if requester_username:
                user = s.query(User).filter_by(ip_address=ip_address, username=requester_username).first()
            if not user:
                user = s.query(User).filter_by(ip_address=ip_address).first()
            now = datetime.now(timezone.utc)
            if not user:
                user = User(
                    ip_address=ip_address,
                    username=requester_username,
                    first_seen_at=now,
                    last_seen_at=now
                )
                s.add(user)
                s.flush([user])
            else:
                user.last_seen_at = now
                if requester_username and not user.username:
                    user.username = requester_username
            
            request_log = Request(
                user_id=user.id,
                requested_username=requested_username,
                source=source,
                version=version,
                mc_version=mc_version,
                endpoint=request.endpoint or "/api/namehistory",
                timestamp=now,
                response_status=200
            )
            s.add(request_log)
    except Exception as e:
        log.error(f"Failed to log request: {e}")

    return jsonify(response_data)

@app.route("/api/namehistory/update", methods=["POST"])
@require_auth(AuthLevel.API)
def api_update():
    body = request.get_json(force=True, silent=False) or {}
    source = request.args.get("source")
    version = request.args.get("version")
    req_name = request.args.get("req_name")
    mc_version = request.args.get("mc_version")

    results = {"updated": [], "errors": []}

    usernames = body.get("usernames", [])
    if "username" in body:
        usernames.append(body["username"])
    uuids = body.get("uuids", [])
    if "uuid" in body:
        uuids.append(body["uuid"])

    # Process usernames
    for name in usernames:
        norm = normalize_username(name)
        if not norm:
            results["errors"].append({"username": name, "error": "Not found"})
        else:
            results["updated"].append(
                _update_profile_from_sources(norm[1], norm[0])
            )
            bulk_sleep()

    # Process UUIDs
    for u in uuids:
        u = dashed_uuid(u)
        name = get_profile_by_uuid_from_mojang(u)
        if not name:
            results["errors"].append({"uuid": u, "error": "Not found"})
        else:
            results["updated"].append(_update_profile_from_sources(u, name))
            bulk_sleep()

    # Log the request (we'll use the first requested username or a summary)
    ip_address = (
        request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        .split(",")[0]
        .strip()
    )

    requester_username = req_name.strip() if req_name and req_name.strip() else None
    requested_username = usernames[0] if usernames else (uuids[0] if uuids else "batch_update")

    try:
        with tx() as s:
            log_request(
                session=s,
                ip_address=ip_address,
                requester_username=requester_username,
                requested_username=requested_username,
                source=source,
                version=version,
                endpoint=request.endpoint or "/api/namehistory/update",
                response_status=200,
                mc_version=mc_version,
            )
    except Exception as e:
        log.error(f"Failed to log request: {e}")

    return jsonify(results)


def get_openapi_spec():
    """Generates the OpenAPI 3.0 specification for the API."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Player History API",
            "version": "1.0.0",
            "description": "An API to track Minecraft player name and skin history.",
        },
        "paths": {
            "/api/namehistory": {
                "get": {
                    "summary": "Get name history by username",
                    "parameters": [
                        {
                            "name": "username",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The Minecraft username.",
                        }
                    ],
                    "responses": {
                        "200": {"description": "Successful response"},
                        "400": {"description": "Invalid username"},
                        "404": {"description": "Username not found"},
                    },
                },
                "post": {
                    "summary": "Get name history by username",
                    "parameters": [
                        {
                            "name": "username",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The Minecraft username.",
                        }
                    ],
                    "responses": {
                        "200": {"description": "Successful response"},
                        "400": {"description": "Invalid username"},
                        "404": {"description": "Username not found"},
                    },
                },
                "delete": {
                    "summary": "Delete a profile",
                    "security": [{"AdminKey": []}],
                    "parameters": [
                        {
                            "name": "username",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "The Minecraft username to delete.",
                        },
                        {
                            "name": "uuid",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "The UUID of the profile to delete.",
                        },
                    ],
                    "responses": {
                        "200": {"description": "Profile deleted"},
                        "400": {"description": "Username or UUID required"},
                        "401": {"description": "Admin authorization required"},
                        "404": {"description": "Profile not found"},
                    },
                }
            },
            "/api/namehistory/update": {
                "post": {
                    "summary": "Force-update profiles",
                    "security": [{"ApiKey": []}],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "usernames": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "uuids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Update process completed"},
                        "401": {"description": "API authorization required"},
                    },
                }
            },
            "/api/skinhistory": {
                "get": {
                    "summary": "Get skin history by username or UUID",
                    "parameters": [
                        {
                            "name": "username",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "The Minecraft username.",
                        },
                        {
                            "name": "uuid",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "The player's UUID.",
                        },
                    ],
                    "responses": {
                        "200": {"description": "Successful response"},
                        "400": {"description": "Invalid username"},
                        "404": {"description": "Profile not found"},
                    },
                },
                "post": {
                    "summary": "Get skin history by username or UUID",
                    "parameters": [
                        {
                            "name": "username",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "The Minecraft username.",
                        },
                        {
                            "name": "uuid",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "The player's UUID.",
                        },
                    ],
                    "responses": {
                        "200": {"description": "Successful response"},
                        "400": {"description": "Invalid username"},
                        "404": {"description": "Profile not found"},
                    },
                },
                "delete": {
                    "summary": "Delete cached skin history",
                    "security": [{"AdminKey": []}],
                    "parameters": [
                        {
                            "name": "username",
                            "in": "query",
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "uuid",
                            "in": "query",
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {
                        "200": {"description": "Skin history deleted"},
                        "401": {"description": "Admin authorization required"},
                        "404": {"description": "Profile not found"},
                    },
                },
            },
            "/api/skinhistory/update": {
                "post": {
                    "summary": "Force-update skin history",
                    "security": [{"ApiKey": []}],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "usernames": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "uuids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Update process completed"},
                        "401": {"description": "API authorization required"},
                    },
                }
            },
            "/api/minecraft/player/profile": {
                "get": {
                    "summary": "Aggregated player profile across clients",
                    "parameters": [
                        {"name": "username", "in": "query", "schema": {"type": "string"}},
                        {"name": "uuid", "in": "query", "schema": {"type": "string"}},
                        {
                            "name": "refresh",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Set to 1/true to bypass cache",
                        },
                    ],
                    "responses": {"200": {"description": "Aggregated profile"}},
                }
            },
            "/api/minecraft/player/{action}": {
                "get": {
                    "summary": "Player actions (namehistory, skinhistory, profile/{source}, cosmetics/{source})",
                    "parameters": [
                        {
                            "name": "action",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {"name": "username", "in": "query", "schema": {"type": "string"}},
                        {"name": "uuid", "in": "query", "schema": {"type": "string"}},
                    ],
                    "responses": {"200": {"description": "Action response"}},
                }
            },
        },
        "components": {
            "securitySchemes": {
                "ApiKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "Authorization",
                    "description": "API key for write-like actions. Use 'API-Key <token>'.",
                },
                "AdminKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "Authorization",
                    "description": "Admin key for admin actions. Use 'Admin-Key <token>'.",
                },
            }
        },
    }
    return spec


def get_docs_html(base_url):
    """Returns a simple, professional HTML documentation page using Pico CSS."""
    api_key_required = bool(API_KEY)
    admin_key_required = bool(ADMIN_KEY)

    return f"""
    <!DOCTYPE html>
    <html lang="en" data-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Player History API Docs</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
        <style>
            .container {{
                max-width: 90%;
                margin: auto;
            }}
            .button-group {{
                display: flex;
                gap: 0.5rem;
            }}
            .try-it-output {{
                margin-top: 1rem;
                background-color: #1e1e1e;
                padding: 1rem;
                border-radius: 0.5rem;
                color: #d4d4d4;
                font-family: 'Courier New', Courier, monospace;
            }}
        </style>
    </head>
    <body>
        <main class="container">
            <header>
                <h1>Player History API Documentation</h1>
                <p>This API allows you to track the name and skin history of Minecraft players.</p>
                <p>Legacy endpoints remain at <code>/api/namehistory</code> and <code>/api/skinhistory</code>.</p>
                <p>The new unified API lives under <a href="/api/minecraft/player/">/api/minecraft/player/</a>.</p>
                <p>The OpenAPI specification is available at <a href="/api/namehistory/docs.json">/api/namehistory/docs.json</a> and <a href="/api/skinhistory/docs.json">/api/skinhistory/docs.json</a>.</p>
            </header>

            <article>
                <h2><kbd>GET</kbd> /api/minecraft/player/profile</h2>
                <p>Aggregated profile from LabyMod, Badlion, Hypixel, NameMC, minecraftuuid.com, and more.</p>
                <p>Returns <code>est_playtime</code>, <code>monthly_views</code>, <code>est_last_online</code>, <code>last_server</code>, and per-source breakdowns.</p>
                <div class="button-group">
                    <button onclick="tryIt('mc-profile')">Try It</button>
                    <button onclick="copyCurl('mc-profile', 'Liforra')">Copy Curl Command</button>
                </div>
                <pre id="mc-profile-output" class="try-it-output" style="display:none;"></pre>
            </article>

            <article>
                <h2><kbd>GET</kbd> /api/minecraft/player/&lt;action&gt;</h2>
                <p>Actions: <code>namehistory</code>, <code>skinhistory</code>, <code>profile/labymod</code>, <code>profile/hypixel</code>, <code>cosmetics/labymod</code>, etc.</p>
                <p>See <a href="/api/minecraft/player/">/api/minecraft/player/</a> for the full list.</p>
            </article>

            <article>
                <h2><kbd>GET</kbd> /api/namehistory</h2>
                <p>Retrieves the name history for a given Minecraft username or UUID.</p>
                <h4>Parameters</h4>
                <ul>
                    <li><code>username</code> (query): The Minecraft username.</li>
                    <li><code>uuid</code> (query): The player's UUID.</li>
                </ul>
                <div class="button-group">
                    <button onclick="tryIt('get')">Try It with username</button>
                    <button onclick="copyCurl('get', 'Notch')">Copy Curl Command (username)</button>
                </div>
                <div class="button-group" style="margin-top: 1rem;">
                    <button onclick="tryIt('uuid')">Try It with UUID</button>
                    <button onclick="copyCurl('uuid', '069a79f4-44e9-4726-a5be-fca90e38aaf5')">Copy Curl Command (UUID)</button>
                </div>
                <pre id="get-output" class="try-it-output" style="display:none;"></pre>
            </article>

            <article>
                <h2><kbd>POST</kbd> /api/namehistory/update {'<small>Requires API Key</small>' if api_key_required else ''}</h2>
                <p>Forces an immediate update for one or more player profiles from external sources.</p>
                <h4>Request Body (JSON)</h4>
                <blockquote><pre><code>{{
    "usernames": ["Notch", "jeb_"],
    "uuids": ["069a79f4-44e9-4726-a5be-fca90e38aaf5"]
}}</code></pre></blockquote>
                <div class="button-group">
                    <button onclick="copyCurl('update')">Copy Curl Command</button>
                </div>
            </article>

            <article>
                <h2><kbd>DELETE</kbd> /api/namehistory/delete</h2>
                <p>Deletes a player's entire profile and name history from the database.</p>
                <h4>Parameters</h4>
                <ul>
                    <li><code>username</code> (query) or <code>uuid</code> (query): The identifier for the profile to delete.</li>
                </ul>
                <div class="button-group">
                    <button onclick="copyCurl('delete')">Copy Curl Command</button>
                </div>
            </article>

            <article>
                <h2><kbd>GET</kbd> /api/skinhistory</h2>
                <p>Retrieves the skin history for a given Minecraft username or UUID.</p>
                <h4>Parameters</h4>
                <ul>
                    <li><code>username</code> (query): The Minecraft username.</li>
                    <li><code>uuid</code> (query): The player's UUID.</li>
                </ul>
                <div class="button-group">
                    <button onclick="tryIt('skin-get')">Try It with username</button>
                    <button onclick="copyCurl('skin-get', 'Dream')">Copy Curl Command (username)</button>
                </div>
                <div class="button-group" style="margin-top: 1rem;">
                    <button onclick="tryIt('skin-uuid')">Try It with UUID</button>
                    <button onclick="copyCurl('skin-uuid', 'ec70bcaf-702f-4bb8-b48d-276fa52a780c')">Copy Curl Command (UUID)</button>
                </div>
                <pre id="skin-get-output" class="try-it-output" style="display:none;"></pre>
            </article>

            <article>
                <h2><kbd>POST</kbd> /api/skinhistory/update {'<small>Requires API Key</small>' if api_key_required else ''}</h2>
                <p>Forces an immediate skin history update for one or more player profiles.</p>
                <div class="button-group">
                    <button onclick="copyCurl('skin-update')">Copy Curl Command</button>
                </div>
            </article>

            <article>
                <h2><kbd>DELETE</kbd> /api/skinhistory/delete</h2>
                <p>Deletes cached skin history for a player without removing their name history.</p>
                <div class="button-group">
                    <button onclick="copyCurl('skin-delete')">Copy Curl Command</button>
                </div>
            </article>
        </main>
        <script>
            function getBaseUrl() {{
                return window.location.origin + '/';
            }}

            function tryIt(type) {{
                const baseUrl = getBaseUrl();
                let url = '';
                let param = '';
                if (type === 'get') {{
                    param = prompt("Enter a username", "Notch");
                    if (!param) return;
                    url = `${{baseUrl}}api/namehistory?username=${{param}}`;
                }} else if (type === 'uuid') {{
                    param = prompt("Enter a UUID", "069a79f4-44e9-4726-a5be-fca90e38aaf5");
                    if (!param) return;
                    url = `${{baseUrl}}api/namehistory?uuid=${{param}}`;
                }} else if (type === 'skin-get') {{
                    param = prompt("Enter a username", "Dream");
                    if (!param) return;
                    url = `${{baseUrl}}api/skinhistory?username=${{param}}`;
                }} else if (type === 'skin-uuid') {{
                    param = prompt("Enter a UUID", "ec70bcaf-702f-4bb8-b48d-276fa52a780c");
                    if (!param) return;
                    url = `${{baseUrl}}api/skinhistory?uuid=${{param}}`;
                }} else if (type === 'mc-profile') {{
                    param = prompt("Enter a username", "Liforra");
                    if (!param) return;
                    url = `${{baseUrl}}api/minecraft/player/profile?username=${{param}}`;
                }}

                const outputId = type === 'mc-profile' ? 'mc-profile-output' : (type.startsWith('skin') ? 'skin-get-output' : 'get-output');
                const output = document.getElementById(outputId);
                output.style.display = 'block';
                output.textContent = `Fetching: ${{url}}\n\n`;

                fetch(url)
                    .then(response => response.json())
                    .then(data => {{
                        output.textContent += JSON.stringify(data, null, 2);
                    }})
                    .catch(error => {{
                        output.textContent += `Error: ${{error}}`;
                    }});
            }}

            function copyCurl(type, param) {{
                const baseUrl = getBaseUrl();
                let command = '';
                if (type === 'get') {{
                    command = `curl "${{baseUrl}}api/namehistory?username=${{param}}"`;
                }} else if (type === 'uuid') {{
                    command = `curl "${{baseUrl}}api/namehistory?uuid=${{param}}"`;
                }} else if (type === 'update') {{
                    command = `curl -X POST -H "Content-Type: application/json" {'-H "Authorization: API-Key YOUR_API_KEY" ' if api_key_required else ''}-d '{{"usernames": ["Notch"]}}' "${{baseUrl}}api/namehistory/update"`;
                }} else if (type === 'delete') {{
                    command = `curl -X DELETE {'-H "Authorization: Admin-Key YOUR_ADMIN_KEY" ' if admin_key_required else ''}"${{baseUrl}}api/namehistory/delete?username=Notch"`;
                }} else if (type === 'skin-get') {{
                    command = `curl "${{baseUrl}}api/skinhistory?username=${{param}}"`;
                }} else if (type === 'skin-uuid') {{
                    command = `curl "${{baseUrl}}api/skinhistory?uuid=${{param}}"`;
                }} else if (type === 'skin-update') {{
                    command = `curl -X POST -H "Content-Type: application/json" {'-H "Authorization: API-Key YOUR_API_KEY" ' if api_key_required else ''}-d '{{"usernames": ["Dream"]}}' "${{baseUrl}}api/skinhistory/update"`;
                }} else if (type === 'skin-delete') {{
                    command = `curl -X DELETE {'-H "Authorization: Admin-Key YOUR_ADMIN_KEY" ' if admin_key_required else ''}"${{baseUrl}}api/skinhistory/delete?username=Dream"`;
                }} else if (type === 'mc-profile') {{
                    command = `curl "${{baseUrl}}api/minecraft/player/profile?username=${{param}}"`;
                }}

                navigator.clipboard.writeText(command).then(() => {{
                    alert('Curl command copied to clipboard!');
                }});
            }}
        </script>
    </body>
    </html>
    """

@app.route("/api/skinhistory/docs.json", methods=["GET"])
def api_skin_docs_json():
    return jsonify(get_openapi_spec())


@app.route("/api/skinhistory/", methods=["GET"])
def api_skin_docs_html_route():
    base_url = request.host_url
    return get_docs_html(base_url)


@app.route("/api/namehistory/docs.json", methods=["GET"])
def api_docs_json():
    return jsonify(get_openapi_spec())

@app.route("/api/namehistory/", methods=["GET"])
def api_docs_html_route():
    base_url = request.host_url
    return get_docs_html(base_url)


RATE_BUCKET: Dict[str, List[float]] = {}
RATE_LOCK = threading.Lock()


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


def log_request(session, ip_address, requester_username, requested_username, source, version, endpoint, response_status, mc_version):
    try:
        user = None
        if requester_username:
            user = session.query(User).filter_by(ip_address=ip_address, username=requester_username).first()
        if not user:
            user = session.query(User).filter_by(ip_address=ip_address).first()
        now = datetime.now(timezone.utc)
        if not user:
            user = User(
                ip_address=ip_address,
                username=requester_username,
                first_seen_at=now,
                last_seen_at=now
            )
            session.add(user)
            session.flush([user])
        else:
            user.last_seen_at = now
            if requester_username and not user.username:
                user.username = requester_username

        request_log = Request(
            user_id=user.id,
            requested_username=requested_username,
            source=source,
            version=version,
            mc_version=mc_version,
            endpoint=endpoint,
            timestamp=now,
            response_status=response_status
        )
        session.add(request_log)
    except Exception as e:
        log.error(f"Failed to log request: {e}")


try:
    Base.metadata.create_all(bind=engine)
    with tx() as session:
        load_tokens_from_db(session, ChallengeToken, challenge_store)
except Exception as exc:
    log.warning("Challenge token DB load skipped: %s", exc)

register_minecraft_routes(
    app,
    {
        "ClientProfileSnapshot": ClientProfileSnapshot,
        "ClientCosmeticSnapshot": ClientCosmeticSnapshot,
        "tx": tx,
        "ensure_profile": ensure_profile,
        "is_source_stale": is_source_stale,
        "update_source_timestamp": update_source_timestamp,
        "resolve_player": resolve_player,
        "log_request": log_request,
        "query_history_public": query_history_public,
        "query_skin_history_public": query_skin_history_public,
        "_update_profile_from_sources": _update_profile_from_sources,
        "_update_skin_profile_from_sources": _update_skin_profile_from_sources,
        "get_profile_by_uuid_from_mojang": get_profile_by_uuid_from_mojang,
        "SCRAPER_STALE_HOURS": SCRAPER_STALE_HOURS,
        "jitter_sleep": jitter_sleep,
        "log": log,
        "check_api_auth": check_api_auth,
        "provider_ctx": {
            "scraper_session": scraper_session,
            "mojang_session": mojang_session,
            "fetch_profile_html": fetch_profile_html,
            "hypixel_api_key": HYPIXEL_API_KEY,
            "challenge_store": challenge_store,
            "laby_challenge_header": LABY_CHALLENGE_HEADER,
            "laby_challenge_token": LABY_CHALLENGE_TOKEN,
            "sync_challenge_tokens": sync_challenge_tokens,
            "invalidate_challenge_token": invalidate_challenge_token,
        },
    },
)

register_captcha_routes(
    app,
    {
        "challenge_store": challenge_store,
        "tx": tx,
        "ChallengeToken": ChallengeToken,
        "persist_challenge_token": lambda session, provider, source: persist_token_to_db(
            session, ChallengeToken, challenge_store, provider, source
        ),
        "load_challenge_tokens": lambda session: load_tokens_from_db(
            session, ChallengeToken, challenge_store
        ),
        "ADMIN_KEY": ADMIN_KEY,
        "LABY_TURNSTILE_SITE_KEY": LABY_TURNSTILE_SITE_KEY,
        "CHALLENGE_TTL_MINUTES": CHALLENGE_TTL_MINUTES,
        "scraper_session": scraper_session,
        "log": log,
    },
)


register_votes_routes(
    app,
    {
        "resolve_player": resolve_player,
        "log_request": log_request,
        "tx": tx,
        "votes_config": VOTES_CONFIG,
        "log": log,
    },
)


# --- MAIN ---
if __name__ == "__main__":
    init_db()
    log.info("DB initialized")
    if AUTO_UPDATE_ENABLED:
        # Auto-update worker can be started in a separate thread here if desired
        pass
    log.info(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT)
