export function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "—";
  const total = Math.max(0, Number(seconds));
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const mins = Math.floor((total % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

export function formatNumber(value) {
  if (value == null) return "—";
  return new Intl.NumberFormat().format(value);
}

export function formatDate(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatShortDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatHeldDuration(ms) {
  if (ms == null || ms < 0) return "—";
  const days = ms / 86400000;
  if (days >= 365) return `~${(days / 365).toFixed(1)}y`;
  if (days >= 30) return `~${(days / 30).toFixed(1)}mo`;
  if (days >= 1) return `~${days.toFixed(1)}d`;
  const hours = ms / 3600000;
  if (hours >= 1) return `~${hours.toFixed(1)}h`;
  return `~${Math.max(1, Math.round(ms / 60000))}m`;
}

export function avatarUrl(uuid, size = 128) {
  const id = (uuid || "").replace(/-/g, "");
  return `https://mc-heads.net/avatar/${id}/${size}`;
}

export function bodyUrl(uuid, size = 256) {
  const id = (uuid || "").replace(/-/g, "");
  return `https://mc-heads.net/body/${id}/${size}`;
}

export function fullRenderUrl(uuid) {
  const id = (uuid || "").replace(/-/g, "");
  return `https://visage.surgeplay.com/full/512/${id}`;
}

export function skinTextureUrl(skinHash, fileHash = null) {
  if (!skinHash) return "";
  const hash = String(skinHash).trim().toLowerCase();
  const params = new URLSearchParams();
  if (fileHash) params.set("file_hash", String(fileHash).trim().toLowerCase());
  const qs = params.toString();
  return `/api/minecraft/skin/${hash}/texture.png${qs ? `?${qs}` : ""}`;
}

export function skinHeadUrl(skinHash, size = 64, slim = null, fileHash = null) {
  if (!skinHash) return "";
  const hash = String(skinHash).trim().toLowerCase();
  const params = new URLSearchParams({ size: String(size) });
  if (fileHash) params.set("file_hash", String(fileHash).trim().toLowerCase());
  return `/api/minecraft/skin/${hash}/head.png?${params}`;
}

/** Legacy external fallbacks if the local cache has not been populated yet. */
export function skinPreviewUrls(skinHash, slim = null) {
  if (!skinHash) return [];
  const hash = String(skinHash).trim();
  const isNameMcId = /^[a-f0-9]{16}$/i.test(hash);
  const isLabyHash = /^[a-f0-9]{32}$/i.test(hash);
  const model = slim === true ? "slim" : "classic";

  if (isNameMcId) {
    return [
      `https://s.namemc.com/2d/skin/face.png?id=${hash}&scale=4`,
      `https://s.namemc.com/3d/skin/body.png?id=${hash}&model=${model}&width=128&height=128`,
    ];
  }

  if (isLabyHash) {
    return [
      `https://skin.laby.net/api/render/skin/${hash}.png?shadow=true`,
      `https://laby.net/api/v3/texture/${hash}/skin.png`,
    ];
  }

  return [
    `https://skin.laby.net/api/render/skin/${hash}.png?shadow=true`,
    `https://s.namemc.com/2d/skin/face.png?id=${hash}&scale=4`,
  ];
}

export function attachSkinPreview(img, skinHash, slim = null, fileHash = null, preferredUrl = null) {
  const urls = [];
  if (preferredUrl) urls.push(preferredUrl);
  const local = skinHeadUrl(skinHash, 64, slim, fileHash);
  if (local && !urls.includes(local)) urls.push(local);
  for (const url of skinPreviewUrls(skinHash, slim)) {
    if (!urls.includes(url)) urls.push(url);
  }
  let index = 0;

  const tryNext = () => {
    if (index >= urls.length) {
      const placeholder = el("div", { className: "skin-thumb-fallback", text: "?" });
      if (img.parentNode) img.replaceWith(placeholder);
      return;
    }
    img.src = urls[index++];
  };

  img.addEventListener("error", tryNext);
  tryNext();
  return img;
}

/**
 * Capes/cloaks have a completely different texture layout than player
 * skins — there is no head/body region to crop — so they need their own
 * preview chain rather than reusing attachSkinPreview's skin-cropping URLs.
 * `frontUrl` is the API-provided local render (preferred); falls back to
 * Laby's direct render for the same hash, then a placeholder.
 */
export function attachCapePreview(img, frontUrl, hash, kind = "cape") {
  const urls = [];
  if (frontUrl) urls.push(frontUrl);
  if (hash) {
    const variant = kind === "cloak" ? "cloakFront" : "capeFront";
    const direct = `https://laby.net/api/v3/texture/${String(hash).trim().toLowerCase()}/${variant}.png`;
    if (!urls.includes(direct)) urls.push(direct);
  }
  let index = 0;

  const tryNext = () => {
    if (index >= urls.length) {
      const placeholder = el("div", { className: "skin-thumb-fallback", text: kind === "cloak" ? "L" : "C" });
      if (img.parentNode) img.replaceWith(placeholder);
      return;
    }
    img.src = urls[index++];
  };

  img.addEventListener("error", tryNext);
  tryNext();
  return img;
}

export function copyButton(label, value) {
  const btn = el("button", {
    className: "nm-copy",
    type: "button",
    text: label,
    onClick: async () => {
      try {
        await navigator.clipboard.writeText(value);
        btn.textContent = "Copied";
        setTimeout(() => {
          btn.textContent = label;
        }, 1200);
      } catch {
        btn.textContent = "Failed";
      }
    },
  });
  return btn;
}

export function enrichNameHistory(rows) {
  const items = (rows || []).map((r) => ({
    ...r,
    ts: r.changed_at ? new Date(r.changed_at).getTime() : null,
  }));

  items.sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0));

  const enriched = items.map((row, i) => {
    const next = items[i + 1];
    let heldMs = null;
    if (next?.ts != null && row.ts != null) heldMs = next.ts - row.ts;
    else if (!next && row.ts != null) heldMs = Date.now() - row.ts;
    return { ...row, heldMs };
  });

  return enriched.reverse();
}

export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, val] of Object.entries(attrs)) {
    if (key === "className") node.className = val;
    else if (key === "text") node.textContent = val;
    else if (key === "html") node.innerHTML = val;
    else if (key.startsWith("on") && typeof val === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), val);
    } else if (val !== false && val != null) {
      node.setAttribute(key, val);
    }
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}
