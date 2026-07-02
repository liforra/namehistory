from __future__ import annotations

from typing import Any, Dict, Optional

from flask import Response, abort, jsonify, request

from providers.laby_challenge import (
    LABY_TURNSTILE_ACTION,
    discover_turnstile_site_key,
    exchange_turnstile_for_challenge,
)


def register_captcha_routes(app, deps: Dict[str, Any]) -> None:
    challenge_store = deps["challenge_store"]
    tx = deps["tx"]
    ChallengeToken = deps["ChallengeToken"]
    persist_token = deps["persist_challenge_token"]
    load_tokens = deps["load_challenge_tokens"]
    ADMIN_KEY = deps["ADMIN_KEY"]
    LABY_TURNSTILE_SITE_KEY = deps.get("LABY_TURNSTILE_SITE_KEY", "")
    CHALLENGE_TTL_MINUTES = deps.get("CHALLENGE_TTL_MINUTES", 45)
    scraper_session = deps["scraper_session"]
    log = deps["log"]

    def _site_key() -> str:
        if LABY_TURNSTILE_SITE_KEY:
            return LABY_TURNSTILE_SITE_KEY
        return discover_turnstile_site_key(scraper_session)

    def _store_challenge_token(
        provider: str,
        raw_token: str,
        *,
        source: str,
        ttl_minutes: Optional[int] = None,
        exchange: bool = True,
    ):
        token = raw_token.strip()
        ttl = ttl_minutes if ttl_minutes is not None else CHALLENGE_TTL_MINUTES
        stored_source = source

        if exchange and provider == "labymod":
            challenge_token, expires_in, err = exchange_turnstile_for_challenge(
                token, scraper_session
            )
            if challenge_token:
                token = challenge_token
                stored_source = f"{source}+exchange"
                if expires_in and expires_in > 0:
                    ttl = max(1, int(expires_in // 60))
            else:
                abort(
                    400,
                    f"Could not exchange Turnstile token: {err or 'unknown error'}",
                )

        challenge_store.set(provider, token, ttl_minutes=ttl, source=stored_source)
        with tx() as session:
            persist_token(session, provider, stored_source)
        return challenge_store.status(provider)

    @app.route("/api/minecraft/captcha/status", methods=["GET"])
    def captcha_status():
        with tx() as session:
            load_tokens(session)
        return jsonify(
            {
                "providers": challenge_store.all_status(),
                "captcha_url": "/api/minecraft/captcha",
                "submit_url": "/api/minecraft/captcha",
            }
        )

    @app.route("/api/minecraft/captcha", methods=["GET", "POST"])
    def captcha_page():
        if request.method == "POST":
            body = request.get_json(force=True, silent=True) or {}
            provider = str(body.get("provider") or "labymod").lower()
            token = str(body.get("token") or body.get("challenge_token") or "").strip()
            if not token:
                abort(400, "token required")

            ttl = body.get("ttl_minutes", CHALLENGE_TTL_MINUTES)
            try:
                ttl = int(ttl)
            except (TypeError, ValueError):
                ttl = CHALLENGE_TTL_MINUTES

            exchange = body.get("exchange", True)
            if isinstance(exchange, str):
                exchange = exchange.lower() not in {"0", "false", "no"}

            status = _store_challenge_token(
                provider,
                token,
                source="captcha-ui",
                ttl_minutes=ttl,
                exchange=exchange,
            )

            log.info("Challenge token stored for provider=%s", provider)
            return jsonify(
                {
                    "message": "Challenge token saved",
                    "provider": provider,
                    "status": status,
                }
            )

        site_key = _site_key()
        status = challenge_store.all_status()
        labymod_status = status.get("labymod", challenge_store.status("labymod"))

        turnstile_block = f"""
            <div class="cf-turnstile" data-sitekey="{site_key}" data-action="{LABY_TURNSTILE_ACTION}" data-callback="onSolved"></div>
            <script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Provider Captcha</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }}
    .card {{ border: 1px solid #ccc; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem; }}
    input, textarea, button {{ width: 100%; box-sizing: border-box; margin: 0.4rem 0; padding: 0.6rem; }}
    .ok {{ color: #0a7a2f; }} .warn {{ color: #9a6700; }}
    code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Provider challenge token</h1>
  <p>Some sources (currently <strong>LabyMod views/month</strong>) require a Cloudflare Turnstile token.
     Solve the captcha here or paste a token manually. The server reuses it for all profile lookups until it expires.</p>

  <div class="card">
    <h2>Status</h2>
    <p>LabyMod token valid: <strong class="{'ok' if labymod_status.get('valid') else 'warn'}">
      {'yes' if labymod_status.get('valid') else 'no'}</strong></p>
    <p>Challenge needed: <strong>{'yes' if labymod_status.get('needed') else 'no'}</strong></p>
    <p>Expires: <code>{labymod_status.get('expires_at') or 'n/a'}</code></p>
  </div>

  <div class="card">
    <h2>Solve captcha</h2>
    {turnstile_block}
    <p id="msg"></p>
  </div>

  <div class="card">
    <h2>Paste token manually</h2>
    <p>Paste a Turnstile token (from the widget above) or an already-exchanged <code>X-Challenge-Token</code>.
       Turnstile tokens are exchanged automatically via Laby's API.</p>
    <label><input type="checkbox" id="skip-exchange" /> Already an X-Challenge-Token (skip exchange)</label>
    <textarea id="manual-token" rows="4" placeholder="Paste challenge / Turnstile token"></textarea>
    <button type="button" onclick="submitManual()">Save token</button>
  </div>

  <script>
    async function submitToken(token, exchange=true) {{
      const res = await fetch('/api/minecraft/captcha', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ provider: 'labymod', token, exchange }})
      }});
      const data = await res.json();
      document.getElementById('msg').textContent = res.ok
        ? 'Saved. You can refresh profile requests now.'
        : (data.description || data.message || 'Failed to save token');
      if (res.ok) setTimeout(() => location.reload(), 800);
    }}
    function onSolved(token) {{ submitToken(token, true); }}
    function submitManual() {{
      const token = document.getElementById('manual-token').value.trim();
      if (!token) return alert('Paste a token first');
      const skip = document.getElementById('skip-exchange').checked;
      submitToken(token, !skip);
    }}
  </script>
</body>
</html>"""
        return Response(html, mimetype="text/html")

    @app.route("/api/minecraft/admin-login", methods=["GET", "POST"])
    def admin_login():
        if request.method == "GET":
            return Response(
                """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Admin challenge login</title></head>
<body style="font-family:system-ui;max-width:560px;margin:2rem auto">
<h1>Admin challenge login</h1>
<p>Store a provider challenge token using your <code>Admin-Key</code>.</p>
<form method="post">
  <label>Admin key<br><input name="admin_key" type="password" style="width:100%"></label><br><br>
  <label>Provider<br><input name="provider" value="labymod" style="width:100%"></label><br><br>
  <label>Challenge token<br><textarea name="token" rows="4" style="width:100%"></textarea></label><br><br>
  <label>TTL minutes<br><input name="ttl_minutes" value="45" style="width:100%"></label><br><br>
  <button type="submit">Save</button>
</form>
<p>Or use API: <code>POST /api/minecraft/admin-login</code> with header <code>Authorization: Admin-Key ...</code></p>
</body></html>""",
                mimetype="text/html",
            )

        if ADMIN_KEY:
            hdr = request.headers.get("Authorization", "")
            token_hdr = ""
            if hdr.startswith("Admin-Key "):
                token_hdr = hdr.split(None, 1)[1].strip()
            elif request.form.get("admin_key"):
                token_hdr = request.form.get("admin_key", "").strip()
            if token_hdr != ADMIN_KEY:
                abort(401, "Admin authorization required")

        body = request.get_json(force=True, silent=True) or {}
        provider = str(
            body.get("provider") or request.form.get("provider") or "labymod"
        ).lower()
        token = str(
            body.get("token")
            or body.get("challenge_token")
            or request.form.get("token")
            or ""
        ).strip()
        if not token:
            abort(400, "token required")

        ttl_raw = body.get("ttl_minutes") or request.form.get("ttl_minutes") or CHALLENGE_TTL_MINUTES
        try:
            ttl = int(ttl_raw)
        except (TypeError, ValueError):
            ttl = CHALLENGE_TTL_MINUTES

        exchange = body.get("exchange", True)
        if isinstance(exchange, str):
            exchange = exchange.lower() not in {"0", "false", "no"}

        status = _store_challenge_token(
            provider,
            token,
            source="admin-login",
            ttl_minutes=ttl,
            exchange=exchange,
        )

        accept = request.headers.get("Accept", "")
        if "text/html" in accept and not body:
            return Response(
                "<p>Saved. <a href='/api/minecraft/captcha'>Back to captcha status</a></p>",
                mimetype="text/html",
            )
        return jsonify(
            {
                "message": "Challenge token saved",
                "provider": provider,
                "status": status,
            }
        )
