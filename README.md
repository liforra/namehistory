# ⚠️ WARNING: USE AT YOUR OWN CAUTION

**Using this project may violate the Terms of Service (ToS) of NameMC and LabyMod. You are solely responsible for any consequences. Proceed at your own risk.**

---

# PlayerHistoryAPI

PlayerHistoryAPI is a Python-based API server for retrieving, storing, and managing Minecraft player name history. It aggregates data from Mojang, NameMC, and Laby.net, and provides a local cache and REST API for efficient and privacy-friendly lookups.

## Features

- **Fetches Minecraft player name history** from Mojang, NameMC, and Laby.net
- **REST API** for querying by username or UUID
- **Bulk update and refresh endpoints**
- **Automatic background updates** for stale profiles
- **SQLite database** for local caching and deduplication
- **Rate limiting** to prevent abuse
- **Configurable via `config.toml`**
- **Logging** with optional JSON output and file rotation
- **Database cleaning** to remove duplicates
- **Debug mode** for raw data inspection

## How It Works

1. **API Endpoints**: The Flask server exposes endpoints to query name history by username or UUID, trigger updates, and refresh all cached profiles.
2. **Data Aggregation**: When a query is made, the server fetches data from Mojang (official), NameMC (scraped), and Laby.net (API), merges and deduplicates the results, and stores them locally.
3. **Caching**: Results are cached in a local SQLite database. Stale data is automatically refreshed in the background.
4. **Rate Limiting**: Requests are rate-limited per IP to prevent abuse.
5. **Configuration**: All major settings (server, rate limits, fetch delays, logging, etc.) can be customized in `config.toml`.

## API Endpoints

- `GET /api/namehistory?username=<name>`: Get name history by username
- `GET /api/namehistory/uuid/<uuid>`: Get name history by UUID
- `POST /api/namehistory/update`: Update one or more profiles (by username or UUID)
- `POST /api/namehistory/refresh-all`: Schedule a refresh for all cached profiles

### Endpoint Details

- **GET /api/namehistory?username=USERNAME**
  - Returns the name history for a given username. If the username is not current, but exists in the cache, it will attempt to resolve the new name and return the updated history.
- **GET /api/namehistory/uuid/UUID**
  - Returns the name history for a given UUID. If the cache is stale, it will fetch fresh data.
- **POST /api/namehistory/update**
  - Accepts JSON with `username`, `uuid`, `usernames`, or `uuids` to update one or more profiles. Returns updated data or errors.
- **POST /api/namehistory/refresh-all**
  - Schedules all cached profiles for refresh over 24 hours.

## Configuration

Create a `config.toml` file in the project root to override defaults. Example:

```toml
[server]
host = "0.0.0.0"
port = 8080

[rate_limit]
rpm = 10
window_seconds = 60

[database]
path = "namehistory.db"

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
   pip install -r requirements.txt
   ```
2. **(Optional) Create and edit `config.toml`**
3. **Initialize the database and start the server:**
   ```bash
   python main.py
   ```
4. **(Optional) Clean the database:**
   ```bash
   python main.py clean
   ```
5. **(Optional) Debug mode (download raw HTML/API data):**
   ```bash
   python main.py --dump-html <username>
   ```

## Database

- Uses SQLite (`namehistory.db` by default)
- Stores profiles, name history, provider details, and update timestamps
- Cleans up duplicate entries with the `clean` command

## Logging

- Console and optional file logging
- Supports JSON log format
- Log rotation and backup

## Dependencies

- Python 3.7+
- Flask
- requests
- curl_cffi
- beautifulsoup4
- tomli (for TOML config)

## Legal Notice

- This project is not affiliated with, endorsed by, or in any way officially connected with NameMC, LabyMod, Mojang, or Microsoft.
- Use of this tool may violate the Terms of Service of NameMC and LabyMod. You are solely responsible for any actions taken with this tool.
- The author assumes no liability for misuse or damages resulting from the use of this project.

## License

MIT License. See the `LICENSE` file for details.
