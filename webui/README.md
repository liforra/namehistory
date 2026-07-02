# Player History Web UI

Standalone public website for the [Player History API](../README.md). This folder is fully self-contained and can be copied or deployed anywhere.

## Features

- **Player lookup** — username or UUID search with profile overview, name history, skin history, per-source data, and cosmetics
- **API documentation** — human-readable reference with copyable curl examples
- **Admin panel** — sign in with the server's `Admin-Key` (`NAMEHISTORY_ADMIN_KEY`) to refresh data, delete profiles, and manage provider challenge tokens

## Quick start

```bash
cd webui
cp .env.example .env
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api/*` to `http://127.0.0.1:8000` by default (change `VITE_API_PROXY_TARGET` in `.env`).

Start the API server separately:

```bash
cd ..
python main.py
```

## Production build

```bash
npm run build
```

Static files are written to `dist/`. Serve them with any static host (nginx, Caddy, GitHub Pages, etc.).

```bash
npm run preview   # local preview of the production build
```

## Configuration

| Variable | Purpose |
|----------|---------|
| `VITE_API_BASE_URL` | Absolute API origin for production builds, e.g. `https://api.example.com`. Leave empty when the UI and API share the same origin behind a reverse proxy. |
| `VITE_API_PROXY_TARGET` | Backend URL for `npm run dev` / `npm run preview` proxy only. |

Example `.env` for a split deployment:

```env
VITE_API_BASE_URL=https://history.example.com
```

## Deployment patterns

### Same origin (recommended)

Put the built UI at `/` and proxy API routes to the Flask app:

```nginx
location / {
    root /var/www/namehistory-webui/dist;
    try_files $uri $uri/ /index.html;
}

location /api/ {
    proxy_pass http://127.0.0.1:8000;
}
```

Leave `VITE_API_BASE_URL` empty so the browser calls `/api/...` on the same host.

### Separate origins

1. Build with `VITE_API_BASE_URL` pointing at the API.
2. Enable CORS on the API for the UI origin (not included in the API by default).

## Admin authentication

The admin panel stores your token in **sessionStorage** only (cleared when the tab closes). It is sent as:

```http
Authorization: Admin-Key <token>
```

Login verifies the token against a protected admin endpoint on the API. If the server has no admin key configured, admin routes are open and any non-empty token will sign you in.

## Independence

This package has no Python or Node runtime requirement in production — only the built static assets in `dist/`. Node is needed only for development and building.
