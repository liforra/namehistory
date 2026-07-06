from __future__ import annotations

import os
from typing import Any, Dict, Optional

import tomllib


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 8000},
    "rate_limit": {"rpm": 15, "window_seconds": 60},
    "auth": {"api_key": "", "admin_key": ""},
    "fetch": {
        "default_sleep": 0.5,
        "bulk_min_delay": 0.8,
        "bulk_max_delay": 1.5,
        "fuzzy_window_days": 1,
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
    "providers": {
        "hypixel_api_key": "",
        "laby_turnstile_site_key": "",
        "laby_challenge_header": "X-Challenge-Token",
        "laby_challenge_token": "",
        "challenge_ttl_minutes": 45,
    },
    "skins": {
        "cache_dir": "skin_cache",
    },
    "votes": {
        "sites": {
            "namemc": True,
            "minecraft_mp": True,
            "minecraftpocket_servers": True,
            "trustmyserver": True,
            "topgservers": True,
        },
        "servers": [],
    },
}

# Maps environment variable -> (section, key)
ENV_OVERRIDES = {
    "NAMEHISTORY_HOST": ("server", "host"),
    "NAMEHISTORY_PORT": ("server", "port"),
    "NAMEHISTORY_DB_PATH": ("database", "path"),
    "NAMEHISTORY_API_KEY": ("auth", "api_key"),
    "NAMEHISTORY_ADMIN_KEY": ("auth", "admin_key"),
    "HYPIXEL_API_KEY": ("providers", "hypixel_api_key"),
    "NAMEHISTORY_HYPIXEL_API_KEY": ("providers", "hypixel_api_key"),
    "LABY_TURNSTILE_SITE_KEY": ("providers", "laby_turnstile_site_key"),
    "LABY_CHALLENGE_TOKEN": ("providers", "laby_challenge_token"),
    "LABY_CHALLENGE_HEADER": ("providers", "laby_challenge_header"),
    "CHALLENGE_TTL_MINUTES": ("providers", "challenge_ttl_minutes"),
}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _parse_dotenv_file(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip('"').strip("'")
            values[key] = value
    return values


def load_dotenv(path: Optional[str] = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (without overwriting)."""
    candidates = [path] if path else [
        os.path.join(PROJECT_ROOT, ".env"),
        ".env",
    ]
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        for key, value in _parse_dotenv_file(candidate).items():
            os.environ.setdefault(key, value)
        return


def _apply_dotenv_to_config(config: Dict[str, Any], path: str) -> Dict[str, Any]:
    out = config
    for env_key, (section, cfg_key) in ENV_OVERRIDES.items():
        value = _parse_dotenv_file(path).get(env_key)
        if value is None or value == "":
            continue
        if section not in out:
            out[section] = {}
        if cfg_key == "port":
            out[section][cfg_key] = int(value)
        elif cfg_key == "challenge_ttl_minutes":
            out[section][cfg_key] = int(value)
        else:
            out[section][cfg_key] = value
    return out


def _config_path(filename: str) -> str:
    return os.path.join(PROJECT_ROOT, filename)


def _apply_toml_file(config: Dict[str, Any], path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return config
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    return _merge_dict(config, data)


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    out = config
    for env_key, (section, cfg_key) in ENV_OVERRIDES.items():
        value = os.environ.get(env_key)
        if value is None or value == "":
            continue
        if section not in out:
            out[section] = {}
        if cfg_key == "port":
            out[section][cfg_key] = int(value)
        elif cfg_key == "challenge_ttl_minutes":
            out[section][cfg_key] = int(value)
        else:
            out[section][cfg_key] = value
    return out


def load_config() -> Dict[str, Any]:
    """
    Load configuration in priority order (lowest -> highest):
      defaults -> config.toml -> config.local.toml -> .env -> environment variables
    """
    dotenv_path = os.path.join(PROJECT_ROOT, ".env")
    load_dotenv(dotenv_path)
    config = dict(DEFAULT_CONFIG)
    config = _apply_toml_file(config, _config_path("config.toml"))
    config = _apply_toml_file(config, _config_path("config.local.toml"))
    if os.path.exists(dotenv_path):
        config = _apply_dotenv_to_config(config, dotenv_path)
    config = _apply_env_overrides(config)
    return config
