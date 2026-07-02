from __future__ import annotations

from typing import Any, Dict

from flask import abort, jsonify, request

from providers.votes.registry import VOTE_SITES, fetch_player_votes, fetch_votes_for_site


def register_votes_routes(app, deps: Dict[str, Any]) -> None:
    resolve_player = deps["resolve_player"]
    log_request = deps["log_request"]
    tx = deps["tx"]
    votes_config = deps.get("votes_config") or {}

    def _log_hit(requested_username: str, endpoint: str, status: int = 200) -> None:
        ip_address = (
            request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
            .split(",")[0]
            .strip()
        )
        req_name = request.args.get("req_name")
        requester_username = req_name.strip() if req_name and req_name.strip() else None
        try:
            with tx() as s:
                log_request(
                    session=s,
                    ip_address=ip_address,
                    requester_username=requester_username,
                    requested_username=requested_username,
                    source=None,
                    version=request.args.get("version"),
                    endpoint=endpoint,
                    response_status=status,
                    mc_version=request.args.get("mc_version"),
                )
        except Exception as exc:
            deps["log"].error(f"Failed to log request: {exc}")

    @app.route("/api/minecraft/player/votes", methods=["GET"])
    @app.route("/api/minecraft/player/votes/", methods=["GET"])
    def player_votes_docs():
        if request.args.get("username") or request.args.get("uuid"):
            username = request.args.get("username", "").strip()
            uuid_param = request.args.get("uuid", "").strip()
            uuid, username, display_name = resolve_player(username, uuid_param)
            if not uuid:
                abort(404, "Profile not found")
            snapshot = fetch_player_votes(uuid, display_name, votes_config)
            _log_hit(display_name, "votes")
            return jsonify(snapshot.to_dict())

        return jsonify(
            {
                "action": "votes",
                "description": (
                    "Check configured servers for player votes/likes. Vote listing "
                    "sites do not expose a global per-player history — configure "
                    "which servers to check in config.toml under [votes]."
                ),
                "sites": VOTE_SITES,
                "example": "/api/minecraft/player/votes?username=Notch",
                "per_site": "/api/minecraft/player/votes/<site>?username=Notch",
            }
        )

    @app.route("/api/minecraft/player/votes/<site>", methods=["GET"])
    def player_votes_site(site: str):
        username = request.args.get("username", "").strip()
        uuid_param = request.args.get("uuid", "").strip()
        if not username and not uuid_param:
            abort(400, "username or uuid required")

        canonical = site.lower()
        if canonical not in VOTE_SITES and canonical not in {
            "minecraft-mp",
            "mcmp",
            "mps",
            "tms",
            "topg",
        }:
            abort(404, f"Unknown vote site: {site}")

        uuid, username, display_name = resolve_player(username, uuid_param)
        if not uuid:
            abort(404, "Profile not found")

        snapshot = fetch_votes_for_site(site, uuid, display_name, votes_config)
        _log_hit(display_name, f"votes/{site}")
        return jsonify(snapshot.to_dict())
