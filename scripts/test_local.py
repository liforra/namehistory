#!/usr/bin/env python3
"""Run a full local smoke test against the API (loads secrets from .env)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main


def _ok(label: str, status: int, expected: int = 200) -> bool:
    ok = status == expected
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {label}: HTTP {status}")
    return ok


def main_run() -> int:
    main.init_db()
    client = main.app.test_client()
    user = sys.argv[1] if len(sys.argv) > 1 else "Liforra"
    passed = 0
    total = 0

    cases = [
        ("legacy namehistory", f"/api/namehistory?username={user}"),
        ("legacy skinhistory", f"/api/skinhistory?username={user}&refresh=1"),
        ("minecraft docs", "/api/minecraft/player/"),
        ("minecraft profile", f"/api/minecraft/player/profile?username={user}&refresh=1"),
        ("minecraft namehistory", f"/api/minecraft/player/namehistory?username={user}"),
        ("minecraft skinhistory", f"/api/minecraft/player/skinhistory?username={user}"),
        ("profile labymod", f"/api/minecraft/player/profile/labymod?username={user}"),
        ("profile hypixel", f"/api/minecraft/player/profile/hypixel?username={user}"),
        ("profile minecraftuuid", f"/api/minecraft/player/profile/minecraftuuid?username={user}"),
        ("cosmetics labymod", f"/api/minecraft/player/cosmetics/labymod?username={user}"),
        ("cosmetics essentials", f"/api/minecraft/player/cosmetics/essentials?username={user}"),
        ("cosmetics lunar", f"/api/minecraft/player/cosmetics/lunar?username={user}"),
    ]

    for label, path in cases:
        total += 1
        resp = client.get(path)
        if _ok(label, resp.status_code):
            passed += 1
            if label == "minecraft profile":
                data = resp.get_json() or {}
                print(
                    "       est_playtime=",
                    data.get("est_playtime"),
                    "monthly_views=",
                    data.get("monthly_views"),
                    "est_last_online=",
                    data.get("est_last_online"),
                )
                hyp = (data.get("sources") or {}).get("hypixel", {})
                print(
                    "       hypixel available=",
                    hyp.get("available"),
                    "error=",
                    hyp.get("error"),
                )

    total += 1
    resp = client.post(f"/api/minecraft/player/update?username={user}")
    if _ok("minecraft update", resp.status_code):
        passed += 1
        print("       ", json.dumps(resp.get_json() or {}, indent=2)[:400])

    print(f"\n{passed}/{total} checks passed")
    has_hypixel_key = bool(main.HYPIXEL_API_KEY)
    print(f"Hypixel API key loaded from env/config: {'yes' if has_hypixel_key else 'no'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main_run())
