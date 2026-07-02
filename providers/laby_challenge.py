from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from providers.http_client import fetch_with_fallback

log = logging.getLogger("namehistory")

# Embedded in Laby's public frontend (/_next/static/chunks/1q1n1emmm0u2_.js)
LABY_TURNSTILE_SITE_KEY = "0x4AAAAAAAAtPksJub4J6z5p"
LABY_TURNSTILE_ACTION = "get_challenge_token"
LABY_CHALLENGE_EXCHANGE_URL = "https://laby.net/api/v3/challenge/token"

_SITE_KEY_CACHE: Dict[str, Any] = {"key": None, "expires_at": 0.0}
_SITE_KEY_LOCK = threading.Lock()
_SITE_KEY_CACHE_TTL = 6 * 3600


def _sync_challenge_tokens(ctx: dict) -> None:
    sync = ctx.get("sync_challenge_tokens")
    if sync:
        sync()


def _invalidate_challenge(ctx: dict, provider: str = "labymod") -> None:
    store = ctx.get("challenge_store")
    if store:
        store.clear(provider)
        store.mark_needed(provider)
    invalidate = ctx.get("invalidate_challenge_token")
    if invalidate:
        invalidate(provider)


def build_challenge_headers(
    ctx: dict, provider: str = "labymod"
) -> Dict[str, str]:
    headers: Dict[str, str] = {"Accept": "application/json"}
    store = ctx.get("challenge_store")
    header_name = str(ctx.get("laby_challenge_header") or "X-Challenge-Token")
    token = None
    if store:
        token = store.get(provider)
    if not token:
        token = str(ctx.get("laby_challenge_token") or "").strip() or None
    if token:
        headers[header_name] = token
    return headers


def _extract_site_key_from_text(text: str) -> Optional[str]:
    patterns = (
        r'"(0x4[A-Za-z0-9_-]{10,})"',
        r"'(0x4[A-Za-z0-9_-]{10,})'",
        r'data-sitekey="(0x4[^"]+)"',
        r"sitekey['\"]?\s*[:=]\s*['\"](0x4[^'\"]+)['\"]",
        r"turnstileSiteKey['\"]?\s*[:=]\s*['\"](0x4[^'\"]+)['\"]",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1)
    return None


def discover_turnstile_site_key(scraper_session: Any) -> str:
    """Return Laby Turnstile site key (cached). Falls back to known public key."""
    now = time.time()
    with _SITE_KEY_LOCK:
        cached = _SITE_KEY_CACHE.get("key")
        if cached and _SITE_KEY_CACHE["expires_at"] > now:
            return cached

    discovered: Optional[str] = None
    try:
        response = fetch_with_fallback(
            "https://laby.net/@Liforra",
            session=scraper_session,
            allow=[200],
        )
        if response and response.status_code == 200:
            discovered = _extract_site_key_from_text(response.text)
            if not discovered:
                scripts = re.findall(r'src="(/_next/static/[^"]+\.js)"', response.text)
                # Turnstile bundle is usually late in the chunk list — scan all of them.
                for path in scripts:
                    js = fetch_with_fallback(
                        f"https://laby.net{path}",
                        session=scraper_session,
                        allow=[200],
                    )
                    if not js or js.status_code != 200:
                        continue
                    text = js.text
                    if (
                        "turnstile" not in text.lower()
                        and "sitekey" not in text.lower()
                        and LABY_TURNSTILE_SITE_KEY[:8] not in text
                    ):
                        continue
                    discovered = _extract_site_key_from_text(text)
                    if discovered:
                        break
    except Exception as exc:
        log.debug("Turnstile site key discovery failed: %s", exc)

    site_key = discovered or LABY_TURNSTILE_SITE_KEY
    with _SITE_KEY_LOCK:
        _SITE_KEY_CACHE["key"] = site_key
        _SITE_KEY_CACHE["expires_at"] = now + _SITE_KEY_CACHE_TTL
    return site_key


def exchange_turnstile_for_challenge(
    turnstile_token: str,
    scraper_session: Any,
    *,
    fingerprint: Optional[str] = None,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Exchange a Cloudflare Turnstile token for a Laby X-Challenge-Token.
    Returns (challenge_token, expires_in_seconds, error_message).
    """
    turnstile_token = turnstile_token.strip()
    if not turnstile_token:
        return None, None, "Empty Turnstile token"

    body: Dict[str, Any] = {"turnstile": turnstile_token}
    if fingerprint:
        body["fingerprint"] = fingerprint

    try:
        fetch_with_fallback(
            "https://laby.net/@Liforra",
            session=scraper_session,
            allow=[200],
        )
        response = scraper_session.post(
            LABY_CHALLENGE_EXCHANGE_URL,
            json=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Origin": "https://laby.net",
                "Referer": "https://laby.net/",
            },
            timeout=20,
        )
    except Exception as exc:
        return None, None, f"Challenge exchange failed: {exc}"

    if response.status_code != 200:
        try:
            payload = response.json()
            message = payload.get("message") or payload.get("error") or response.text[:200]
        except Exception:
            message = response.text[:200] or f"HTTP {response.status_code}"
        return None, None, str(message)

    try:
        payload = response.json()
    except Exception:
        return None, None, "Invalid JSON from Laby challenge exchange"

    token = payload.get("token")
    expires_in = payload.get("expiresIn") or payload.get("expires_in")
    if not token:
        return None, None, "Laby challenge exchange returned no token"

    try:
        expires_in = int(expires_in) if expires_in is not None else None
    except (TypeError, ValueError):
        expires_in = None

    return str(token), expires_in, None


def laby_v3_get(
    uuid: str,
    endpoint: str,
    scraper_session: Any,
    ctx: dict,
) -> Tuple[Optional[dict], int, Optional[str]]:
    _sync_challenge_tokens(ctx)
    url = f"https://laby.net/api/v3/user/{uuid}/{endpoint}"
    headers = build_challenge_headers(ctx)
    header_name = str(ctx.get("laby_challenge_header") or "X-Challenge-Token")
    had_token = bool(headers.get(header_name))
    response = fetch_with_fallback(
        url,
        session=scraper_session,
        headers=headers,
        allow=[200, 204, 403, 404, 428],
    )
    if response is None:
        return None, 0, "No response from Laby API"

    if response.status_code == 428:
        store = ctx.get("challenge_store")
        # Flag store when views needs a token, or when we had no token at all.
        if store and (endpoint == "views" or not had_token):
            store.mark_needed("labymod")
        # Only drop the stored token when views rejects it — not when other v3
        # endpoints 428 while views already succeeded with the same token.
        if had_token and endpoint == "views":
            _invalidate_challenge(ctx)
        return None, 428, "Challenge token required — solve at /api/minecraft/captcha"

    if response.status_code == 403 and had_token and endpoint == "views":
        _invalidate_challenge(ctx)
        try:
            payload = response.json()
        except Exception:
            payload = {}
        return payload, 403, payload.get("message", "Forbidden")

    if response.status_code == 204:
        return None, 204, None

    if response.status_code == 403:
        try:
            payload = response.json()
        except Exception:
            payload = {}
        return payload, 403, payload.get("message", "Forbidden")

    if response.status_code != 200:
        return None, response.status_code, response.text[:200]

    try:
        return response.json(), 200, None
    except Exception:
        return None, response.status_code, "Invalid JSON from Laby API"
