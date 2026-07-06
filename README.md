# ⚠️ WARNING: USE AT YOUR OWN CAUTION

**Using this project may violate the Terms of Service (ToS) of NameMC and LabyMod. You are solely responsible for any consequences. Proceed at your own risk.**

---

# PlayerHistoryAPI

PlayerHistoryAPI is a Python-based API server for retrieving, storing, and managing Minecraft player name history. It aggregates data from Mojang, NameMC, and Laby.net, and provides a local cache and REST API for efficient and privacy-friendly lookups.

Main Repository on [Codeberg](https://codeberg.org/liforra/namehistory), read-only Mirror on [Github](https://github.com/liforra/namehistory)

## Features

- **Fetches Minecraft player name history** from Mojang, NameMC, and Laby.net
- **REST API** for querying by username or UUID
- **Bulk update and refresh endpoints**
- **Automatic background updates** for stale profiles
- **Multi-database support:** SQLite, MariaDB, MySQL, and PostgreSQL (via SQLAlchemy)
- **Rate limiting** to prevent abuse
- **Configurable via `config.toml`**
- **Logging** with optional JSON output and file rotation
- **Database cleaning** to remove duplicates
- **Debug mode** for raw data inspection

## How It Works

1. **API Endpoints**: The Flask server exposes endpoints to query name history by username or UUID, trigger updates, and refresh all cached profiles.
2. **Data Aggregation**: When a query is made, the server fetches data from Mojang (official), NameMC (scraped), and Laby.net (API), merges and deduplicates the results, and stores them locally.
3. **Caching**: Results are cached in a local database (SQLite, MariaDB, MySQL, or PostgreSQL). Stale data is automatically refreshed in the background.
4. **Rate Limiting**: Requests are rate-limited per IP to prevent abuse.
5. **Configuration**: All major settings (server, rate limits, fetch delays, logging, etc.) can be customized in `config.toml`.

## API Endpoints

### Legacy (unchanged)

- `GET /api/namehistory?username=<name>`: Get name history by username
- `GET /api/skinhistory?username=<name>`: Get skin history by username
- `POST /api/namehistory/update`: Update one or more profiles (by username or UUID)
- `POST /api/skinhistory/update`: Update skin history for one or more profiles

### Unified Minecraft API

Base: `/api/minecraft/player/<action>?username=<name>` or `?uuid=<uuid>`

- `GET /api/minecraft/player/profile` — aggregated profile (`est_playtime`, `monthly_views`, `est_last_online`, `last_server`, per-source breakdown)
- `GET /api/minecraft/player/namehistory` — name history
- `GET /api/minecraft/player/skinhistory` — skin history
- `POST /api/minecraft/player/update` — force refresh profile, name, and skin data
- `GET /api/minecraft/player/profile/<source>` — labymod, badlion, hypixel, minecraftuuid, namemc, mineplex
- `GET /api/minecraft/player/cosmetics/<source>` — labymod, badlion, essentials, lunar
- `GET /api/minecraft/player/votes` — check configured servers for votes/likes across listing sites
- `GET /api/minecraft/player/votes/<site>` — namemc, minecraft_mp, topgservers, etc.

Discovery: `GET /api/minecraft/player/`

**Provider challenge (LabyMod views/month):** Some LabyMod v3 endpoints need a Cloudflare Turnstile token. Prefer solving once via the web UI rather than hardcoding tokens:

- `GET /api/minecraft/captcha` — solve Turnstile or paste a token (stored for all requests until expiry)
- `GET /api/minecraft/captcha/status` — JSON status
- `POST /api/minecraft/captcha` — `{"provider":"labymod","token":"..."}`
- `GET /api/minecraft/admin-login` — admin form (requires `NAMEHISTORY_ADMIN_KEY` when set)
- `POST /api/minecraft/admin-login` — same with `Authorization: Admin-Key ...`

Optional config/env: `LABY_CHALLENGE_TOKEN`, `LABY_TURNSTILE_SITE_KEY`, `CHALLENGE_TTL_MINUTES` (default 45).

**Vote sites:** Listing sites do not publish a full per-player vote history. Configure servers under `[votes.servers]` in `config.toml` — NameMC likes work without keys; Minecraft-MP, TopGServers, etc. need each server's API key from that site's dashboard.

Add `&refresh=1` to bypass cache on profile endpoints.

### Legacy detail endpoints

- **GET /api/namehistory?username=USERNAME**
  - Returns the name history for a given username. If the username is not current, but exists in the cache, it will attempt to resolve the new name and return the updated history.
- **GET /api/namehistory/uuid/UUID**
  - Returns the name history for a given UUID. If the cache is stale, it will fetch fresh data.
- **POST /api/namehistory/update**
  - Accepts JSON with `username`, `uuid`, `usernames`, or `uuids` to update one or more profiles. Returns updated data or errors.
- **POST /api/namehistory/refresh-all**
  - Schedules all cached profiles for refresh over 24 hours.

## Configuration

Create a `config.toml` file in the project root to override defaults. Example for all supported databases:

```toml
[server]
host = "0.0.0.0"
port = 8080

[rate_limit]
rpm = 10
window_seconds = 60

# --- Database config ---
# Supported types: sqlite, mysql, mariadb, postgresql
[database]
# For SQLite:
type = "sqlite"
path = "namehistory.db"
# For MySQL/MariaDB (requires PyMySQL):
# type = "mysql"
# url = "mysql+pymysql://user:password@localhost/dbname"
# For PostgreSQL (requires psycopg2):
# type = "postgresql"
# url = "postgresql+psycopg2://user:password@localhost/dbname"

[logging]
level = "DEBUG"
json = true
file = "namehistory.log"
max_bytes = 10485760
backup_count = 5

[auto_update]
enabled = true
check_interval_minutes = 60
mojang_stale_hours = 6
scraper_stale_hours = 24
batch_size = 10
```

## Usage

1. **Install dependencies:**
   ```bash
   uv sync
   ```
   This reads `pyproject.toml`/`uv.lock` and creates a `.venv` automatically. Prefix subsequent commands with `uv run` (e.g. `uv run python main.py`), or activate the venv yourself.
2. **Configure secrets for local development:**
   ```bash
   cp .env.example .env
   # Edit .env and set HYPIXEL_API_KEY=your-key-here
   ```
   Configuration priority: defaults → `config.toml` → `config.local.toml` → `.env` → environment variables.

   Supported env vars include `HYPIXEL_API_KEY`, `LABY_CHALLENGE_TOKEN`, `LABY_TURNSTILE_SITE_KEY`, `NAMEHISTORY_API_KEY`, `NAMEHISTORY_ADMIN_KEY`, `NAMEHISTORY_PORT`, and `NAMEHISTORY_DB_PATH`.
3. **(Optional) Create and edit `config.toml`**
4. **Set up your database backend:**
   - For SQLite (default): no extra setup needed
   - For MySQL/MariaDB: ensure the database exists and the user has permissions
   - For PostgreSQL: ensure the database exists and the user has permissions
5. **Initialize the database and start the server:**
   ```bash
   uv run python main.py
   ```
6. **(Optional) Clean the database:**
   ```bash
   uv run python main.py clean
   ```
7. **Run local smoke test:**
   ```bash
   uv run python scripts/test_local.py Liforra
   ```
8. **(Optional) Migrate existing database after upgrades:**
   ```bash
   uv run python migrate.py
   ```

## Database

- Uses SQLite (`namehistory.db` by default), or MariaDB, MySQL, or PostgreSQL if configured
- All database access is via SQLAlchemy ORM
- Stores profiles, name history, provider details, and update timestamps
- Cleans up duplicate entries with the `clean` command

## Logging

- Console and optional file logging
- Supports JSON log format
- Log rotation and backup

## Dependencies

- Python 3.14+ (pinned in `.python-version`; [uv](https://docs.astral.sh/uv/) installs it automatically)
- Flask
- requests
- curl_cffi (browser-impersonated scraping)
- beautifulsoup4
- Pillow (skin rendering)
- python-dateutil
- SQLAlchemy (multi-database support)
- PyMySQL (MySQL/MariaDB)
- psycopg2-binary (PostgreSQL)

Exact versions are locked in `uv.lock`; `uv sync` installs everything.

## Legal Notice

- This project is not affiliated with, endorsed by, or in any way officially connected with NameMC, LabyMod, Mojang, or Microsoft.
- Use of this tool may violate the Terms of Service of NameMC and LabyMod. You are solely responsible for any actions taken with this tool.
- The author assumes no liability for misuse or damages resulting from the use of this project.

## License

MIT License. See the `LICENSE` file for details.
