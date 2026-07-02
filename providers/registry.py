from __future__ import annotations

from typing import Any, Callable, Dict, List

from providers.essentials import fetch_essentials_cosmetics
from providers.lunar import fetch_lunar_cosmetics
from providers.mineplex import fetch_mineplex_profile
from providers.badlion import fetch_badlion_cosmetics, fetch_badlion_profile
from providers.hypixel import fetch_hypixel_profile
from providers.labymod import fetch_labymod_cosmetics, fetch_labymod_profile
from providers.minecraftuuid import fetch_minecraftuuid_profile
from providers.namemc_stats import fetch_namemc_profile_stats
from providers.base import CosmeticsSnapshot, SourceSnapshot

ProfileFetcher = Callable[[str, str, Any], SourceSnapshot]
CosmeticFetcher = Callable[[str, str, Any], CosmeticsSnapshot]

PROFILE_PROVIDERS: Dict[str, ProfileFetcher] = {
    "labymod": fetch_labymod_profile,
    "badlion": fetch_badlion_profile,
    "hypixel": fetch_hypixel_profile,
    "minecraftuuid": fetch_minecraftuuid_profile,
    "namemc": fetch_namemc_profile_stats,
    "mineplex": fetch_mineplex_profile,
}

COSMETIC_PROVIDERS: Dict[str, CosmeticFetcher] = {
    "labymod": fetch_labymod_cosmetics,
    "badlion": fetch_badlion_cosmetics,
    "essentials": fetch_essentials_cosmetics,
    "lunar": fetch_lunar_cosmetics,
}

PROFILE_ALIASES = {
    "laby": "labymod",
    "labynet": "labymod",
    "labymod": "labymod",
    "badlion": "badlion",
    "hypixel": "hypixel",
    "minecraftuuid": "minecraftuuid",
    "mcuuid": "minecraftuuid",
    "namemc": "namemc",
    "mineplex": "mineplex",
}


def _ctx_value(ctx: dict, key: str, default: Any = None) -> Any:
    return ctx.get(key, default)


def fetch_profile_source(
    source: str, uuid: str, username: str, ctx: dict
) -> SourceSnapshot:
    canonical = PROFILE_ALIASES.get(source.lower(), source.lower())
    fetcher = PROFILE_PROVIDERS.get(canonical)
    if not fetcher:
        return SourceSnapshot(
            source=source,
            available=False,
            error=f"Unknown profile source: {source}",
        )

    if canonical == "labymod":
        sync = ctx.get("sync_challenge_tokens")
        if sync:
            sync()
        return fetch_labymod_profile(
            uuid, username, _ctx_value(ctx, "scraper_session"), ctx
        )
    if canonical == "hypixel":
        return fetch_hypixel_profile(
            uuid,
            username,
            _ctx_value(ctx, "hypixel_api_key", ""),
            _ctx_value(ctx, "mojang_session"),
        )
    if canonical == "namemc":
        return fetch_namemc_profile_stats(
            uuid, username, _ctx_value(ctx, "fetch_profile_html")
        )
    return fetcher(uuid, username, _ctx_value(ctx, "scraper_session"))


def fetch_cosmetics_source(
    source: str, uuid: str, username: str, ctx: dict
) -> CosmeticsSnapshot:
    canonical = PROFILE_ALIASES.get(source.lower(), source.lower())
    fetcher = COSMETIC_PROVIDERS.get(canonical)
    if not fetcher:
        return CosmeticsSnapshot(
            source=source,
            available=False,
            error=f"Unknown cosmetics source: {source}",
        )
    if canonical == "labymod":
        return fetch_labymod_cosmetics(
            uuid, username, _ctx_value(ctx, "scraper_session"), ctx
        )
    return fetcher(uuid, username, _ctx_value(ctx, "scraper_session"))


def fetch_all_profiles(uuid: str, username: str, ctx: dict) -> List[SourceSnapshot]:
    sync = ctx.get("sync_challenge_tokens")
    if sync:
        sync()
    snapshots: List[SourceSnapshot] = []
    for name in PROFILE_PROVIDERS:
        snapshots.append(fetch_profile_source(name, uuid, username, ctx))
    return snapshots
