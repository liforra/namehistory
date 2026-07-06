from __future__ import annotations

import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from flask import Response, abort, request, send_file
from PIL import Image

log = logging.getLogger("namehistory")

SKIN_HASH_RE = re.compile(r"^[a-f0-9]{8,64}$", re.I)
HEAD_SIZES = {16, 32, 48, 64, 128}


def classify_skin_hash(skin_hash: str) -> str:
    h = skin_hash.strip().lower()
    if re.fullmatch(r"[a-f0-9]{16}", h):
        return "namemc"
    if re.fullmatch(r"[a-f0-9]{32}", h):
        return "laby"
    return "unknown"


def skin_texture_urls(skin_hash: str, file_hash: Optional[str] = None) -> List[str]:
    """Remote sources used only to populate the local cache (tried in order)."""
    h = skin_hash.strip().lower()
    kind = classify_skin_hash(h)
    urls: List[str] = []

    if kind == "namemc":
        urls.append(f"https://s.namemc.com/i/{h}.png")
        urls.append(f"https://mineskin.eu/skin/{h}")
    elif kind == "laby":
        urls.append(f"https://laby.net/api/v3/texture/{h}/skin.png")
        if file_hash:
            urls.append(f"https://laby.net/api/v3/texture/{file_hash.lower()}/skin.png")
        urls.append(f"https://mineskin.eu/skin/{h}")
    else:
        urls.append(f"https://mineskin.eu/skin/{h}")
        urls.append(f"https://laby.net/api/v3/texture/{h}/skin.png")
        urls.append(f"https://s.namemc.com/i/{h}.png")

    seen = set()
    out: List[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def skin_asset_paths(cache_dir: Path, skin_hash: str) -> Tuple[Path, Path]:
    h = skin_hash.strip().lower()
    shard = h[:2]
    skin_path = cache_dir / "skins" / shard / f"{h}.png"
    meta_path = cache_dir / "skins" / shard / f"{h}.json"
    return skin_path, meta_path


def head_asset_path(cache_dir: Path, skin_hash: str, size: int) -> Path:
    h = skin_hash.strip().lower()
    return cache_dir / "heads" / h[:2] / f"{h}_{size}.png"


def skin_public_urls(skin_hash: str, size: int = 32) -> dict[str, str]:
    h = skin_hash.strip().lower()
    return {
        "skin_url": f"/api/minecraft/skin/{h}/texture.png",
        "head_url": f"/api/minecraft/skin/{h}/head.png?size={size}",
    }


def cape_public_urls(cape_hash: str, kind: str = "cape") -> dict[str, str]:
    h = cape_hash.strip().lower()
    return {
        "front_url": f"/api/minecraft/{kind}/{h}/front.png",
        "flat_url": f"/api/minecraft/{kind}/{h}/flat.png",
    }


def _is_image_response(content_type: str, body: bytes) -> bool:
    ct = (content_type or "").lower()
    if "image" in ct:
        return True
    return body[:8].startswith(b"\x89PNG\r\n\x1a\n")


def _normalize_skin_image(data: bytes) -> bytes:
    image = Image.open(io.BytesIO(data))
    image = image.convert("RGBA")
    width, height = image.size
    if width != 64 or height not in (32, 64):
        raise ValueError(f"unexpected skin dimensions {width}x{height}")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def render_head_png(skin_png: bytes, size: int = 32) -> bytes:
    image = Image.open(io.BytesIO(skin_png)).convert("RGBA")
    width, height = image.size
    if width != 64 or height not in (32, 64):
        raise ValueError(f"unexpected skin dimensions {width}x{height}")

    head = image.crop((8, 8, 16, 16))
    if height == 64:
        overlay = image.crop((40, 8, 48, 16))
        head.paste(overlay, (0, 0), overlay)

    if size != 8:
        head = head.resize((size, size), Image.NEAREST)

    out = io.BytesIO()
    head.save(out, format="PNG")
    return out.getvalue()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class SkinAssetStore:
    def __init__(self, cache_dir: Path, fetcher: Callable[..., Any]):
        self.cache_dir = cache_dir
        self.fetcher = fetcher
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_skin(self, skin_hash: str, file_hash: Optional[str] = None) -> bytes:
        last_error: Optional[Exception] = None
        for url in skin_texture_urls(skin_hash, file_hash):
            try:
                response = self.fetcher(url, timeout=15, allow_redirects=True)
                if response is None or response.status_code != 200:
                    continue
                body = response.content
                if not body or not _is_image_response(
                    response.headers.get("content-type", ""), body
                ):
                    continue
                return _normalize_skin_image(body)
            except Exception as exc:
                last_error = exc
                log.debug("skin fetch failed %s: %s", url, exc)
        raise RuntimeError(
            f"could not download skin {skin_hash}"
            + (f": {last_error}" if last_error else "")
        )

    def ensure_skin(
        self, skin_hash: str, file_hash: Optional[str] = None
    ) -> Path:
        h = skin_hash.strip().lower()
        if not SKIN_HASH_RE.fullmatch(h):
            raise ValueError("invalid skin hash")

        skin_path, _meta_path = skin_asset_paths(self.cache_dir, h)
        if skin_path.is_file():
            return skin_path

        png = self._download_skin(h, file_hash)
        _atomic_write(skin_path, png)
        log.info("cached skin texture %s (%d bytes)", h, len(png))
        return skin_path

    def ensure_head(
        self, skin_hash: str, size: int = 32, file_hash: Optional[str] = None
    ) -> Path:
        h = skin_hash.strip().lower()
        if size not in HEAD_SIZES:
            abort(400, "unsupported head size")

        head_path = head_asset_path(self.cache_dir, h, size)
        if head_path.is_file():
            return head_path

        skin_path = self.ensure_skin(h, file_hash)
        head_png = render_head_png(skin_path.read_bytes(), size=size)
        _atomic_write(head_path, head_png)
        return head_path

    def prefetch(self, skin_hash: str, file_hash: Optional[str] = None) -> bool:
        try:
            self.ensure_skin(skin_hash, file_hash)
            self.ensure_head(skin_hash, 32, file_hash)
            return True
        except Exception as exc:
            log.warning("skin prefetch failed for %s: %s", skin_hash, exc)
            return False


CAPE_KINDS = {"cape": "capeFront.png", "cloak": "cloakFront.png"}
CAPE_VARIANTS = {"front": None, "flat": "skin.png"}  # front URL depends on kind


class CapeAssetStore:
    """
    Capes/cloaks have a completely different texture layout than player skins
    (no head/body regions to crop), so they need their own renderer rather
    than reusing SkinAssetStore's head-crop pipeline. Laby's texture API
    pre-renders a "front" view (how the garment actually looks worn) which is
    what a UI thumbnail wants; "flat" is the raw unfolded texture sheet. Only
    Laby (MD5-style hash) garments are supported — this project has no other
    cape/cloak source.
    """

    def __init__(self, cache_dir: Path, fetcher: Callable[..., Any]):
        self.cache_dir = cache_dir
        self.fetcher = fetcher
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _asset_path(self, cape_hash: str, kind: str, variant: str) -> Path:
        h = cape_hash.strip().lower()
        return self.cache_dir / "capes" / h[:2] / f"{h}_{kind}_{variant}.png"

    def ensure(self, cape_hash: str, kind: str = "cape", variant: str = "front") -> Path:
        h = cape_hash.strip().lower()
        if not SKIN_HASH_RE.fullmatch(h):
            raise ValueError("invalid cape hash")
        if kind not in CAPE_KINDS or variant not in CAPE_VARIANTS:
            raise ValueError("invalid cape kind/variant")

        path = self._asset_path(h, kind, variant)
        if path.is_file():
            return path

        remote_name = CAPE_VARIANTS[variant] or CAPE_KINDS[kind]
        url = f"https://laby.net/api/v3/texture/{h}/{remote_name}"
        try:
            response = self.fetcher(url, timeout=15, allow_redirects=True)
            if response is None or response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code if response else 'no response'}")
            body = response.content
            if not body or not _is_image_response(
                response.headers.get("content-type", ""), body
            ):
                raise RuntimeError("non-image response")
            image = Image.open(io.BytesIO(body)).convert("RGBA")
            out = io.BytesIO()
            image.save(out, format="PNG")
            _atomic_write(path, out.getvalue())
        except Exception as exc:
            raise RuntimeError(f"could not download {kind} {h} ({variant}): {exc}") from exc

        log.info("cached %s texture %s/%s (%d bytes)", kind, h, variant, path.stat().st_size)
        return path


def register_cape_routes(app, store: CapeAssetStore) -> None:
    @app.route("/api/minecraft/<kind>/<cape_hash>/<variant>.png", methods=["GET"])
    def cape_asset(kind: str, cape_hash: str, variant: str):
        if kind not in CAPE_KINDS:
            abort(404, "unknown garment kind")
        h = cape_hash.strip().lower()
        if not SKIN_HASH_RE.fullmatch(h):
            abort(400, "invalid cape hash")
        if variant not in CAPE_VARIANTS:
            abort(404, "unknown cape variant")
        try:
            path = store.ensure(h, kind, variant)
        except Exception:
            abort(404, "cape texture not available")
        response = send_file(path, mimetype="image/png", conditional=True)
        response.cache_control.max_age = 31536000
        response.cache_control.public = True
        return response


def register_skin_routes(app, store: SkinAssetStore) -> None:
    @app.route("/api/minecraft/skin/<skin_hash>/texture.png", methods=["GET"])
    def skin_texture_asset(skin_hash: str):
        h = skin_hash.strip().lower()
        if not SKIN_HASH_RE.fullmatch(h):
            abort(400, "invalid skin hash")
        file_hash = request.args.get("file_hash", "").strip() or None
        try:
            path = store.ensure_skin(h, file_hash)
        except Exception:
            abort(404, "skin texture not available")
        response = send_file(path, mimetype="image/png", conditional=True)
        response.cache_control.max_age = 31536000
        response.cache_control.public = True
        return response

    @app.route("/api/minecraft/skin/<skin_hash>/head.png", methods=["GET"])
    def skin_head_asset(skin_hash: str):
        h = skin_hash.strip().lower()
        if not SKIN_HASH_RE.fullmatch(h):
            abort(400, "invalid skin hash")
        try:
            size = int(request.args.get("size", "32"))
        except ValueError:
            abort(400, "invalid size")
        file_hash = request.args.get("file_hash", "").strip() or None
        try:
            path = store.ensure_head(h, size=size, file_hash=file_hash)
        except Exception:
            abort(404, "skin head not available")
        response = send_file(path, mimetype="image/png", conditional=True)
        response.cache_control.max_age = 31536000
        response.cache_control.public = True
        return response
