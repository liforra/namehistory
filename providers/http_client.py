from __future__ import annotations

import logging
from typing import Any, List, Optional

from curl_cffi.requests import Session

log = logging.getLogger("namehistory")

DEFAULT_IMPERSONATE = "chrome124"
FALLBACK_IMPERSONATES: List[str] = [
    "chrome124",
    "chrome120",
    "safari17_0",
    "chrome110",
]


def create_scraper_session(impersonate: str = DEFAULT_IMPERSONATE) -> Session:
    return Session(impersonate=impersonate)


def fetch_with_fallback(
    url: str,
    *,
    session: Optional[Session] = None,
    timeout: float = 15.0,
    headers: Optional[dict] = None,
    allow: Optional[List[int]] = None,
    **kwargs: Any,
) -> Any:
    """
    GET with browser impersonation fallbacks. Returns the first response that is not
    blocked (403/428) when possible.
    """
    allow = allow or [200, 204, 404]
    impersonates = FALLBACK_IMPERSONATES if session is None else [None]

    last_response = None
    for imp in impersonates:
        try:
            client = session if session is not None else create_scraper_session(imp)
            response = client.get(
                url,
                headers=headers,
                timeout=timeout,
                **kwargs,
            )
            last_response = response
            if response.status_code in allow:
                return response
            if response.status_code not in (403, 428, 429, 503):
                return response
        except Exception as exc:
            log.debug("fetch_with_fallback %s (%s): %s", url, imp, exc)
            continue

    return last_response
