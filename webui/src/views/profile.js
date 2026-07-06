import { api } from "../lib/api.js";
import { COSMETIC_SOURCES, PROFILE_SOURCES } from "../config.js";
import { profileNeedsCaptcha } from "../lib/challenge_gate.js";
import {
  attachCapePreview,
  attachSkinPreview,
  avatarUrl,
  copyButton,
  el,
  enrichNameHistory,
  formatDate,
  formatDuration,
  formatHeldDuration,
  formatNumber,
  formatShortDate,
  fullRenderUrl,
} from "../lib/format.js";

function nmPanel(title, content, extra = null) {
  const head = el("div", { className: "nm-panel-head" }, [
    el("h2", { className: "nm-panel-title", text: title }),
    extra,
  ]);
  return el("section", { className: "nm-panel" }, [head, el("div", { className: "nm-panel-body" }, [content])]);
}

function renderSkinViewer(uuid, username) {
  const viewer = el("div", { className: "nm-skin-viewer" });
  const img = el("img", {
    className: "nm-skin-render",
    src: fullRenderUrl(uuid),
    alt: `${username} skin`,
    loading: "eager",
  });
  img.addEventListener("error", () => {
    img.src = `https://mc-heads.net/body/${(uuid || "").replace(/-/g, "")}/256`;
  });
  viewer.append(img);
  return viewer;
}

function renderSkinThumbs(skinRows, onSelect) {
  const unique = [];
  const seen = new Set();
  for (const row of [...(skinRows || [])].reverse()) {
    const h = (row.skin_hash || "").toLowerCase();
    if (!h || seen.has(h)) continue;
    seen.add(h);
    unique.push(row);
  }

  if (!unique.length) {
    return el("p", { className: "nm-empty", text: "No skins recorded." });
  }

  const grid = el("div", { className: "nm-thumb-grid" });
  const buttons = [];
  for (const row of unique) {
    const thumb = el("img", {
      className: "nm-skin-thumb",
      alt: "Skin",
      loading: "lazy",
    });
    attachSkinPreview(thumb, row.skin_hash, row.slim, row.file_hash, row.head_url);
    const btn = el(
      "button",
      {
        className: "nm-thumb-wrap",
        type: "button",
        title: formatShortDate(row.changed_at),
      },
      [thumb]
    );
    btn.addEventListener("click", () => {
      for (const b of buttons) b.classList.remove("selected");
      btn.classList.add("selected");
      onSelect?.("skin", row.skin_hash);
    });
    buttons.push(btn);
    grid.append(btn);
  }
  return grid;
}

function collectCapes(cosmeticsBySource) {
  const capes = [];
  const seen = new Set();
  for (const entry of cosmeticsBySource || []) {
    const snap = entry?.data?.cosmetics;
    if (!snap?.available) continue;
    for (const item of snap.cosmetics || []) {
      if (item.type !== "cape" && item.type !== "cloak") continue;
      const key = (item.hash || item.file_hash || "").toLowerCase();
      if (!key || seen.has(key)) continue;
      seen.add(key);
      capes.push({ ...item, source: entry.source });
    }
  }
  return capes;
}

const TEXTURE_TYPE_LABELS = { skin: "Skin", cape: "Cape", cloak: "Cloak" };

/** Click-to-inspect panel: name, description, wear count, and tags for a skin or cape. */
function renderTextureInfoPanel() {
  const body = el("div", { className: "nm-texture-info" }, [
    el("p", {
      className: "nm-empty",
      text: "Click a skin or cape to see its name, tags, and how many players have worn it.",
    }),
  ]);

  let requestId = 0;

  async function show(type, hash) {
    const id = ++requestId;
    body.innerHTML = "";
    body.append(el("p", { className: "nm-muted", text: "Loading…" }));

    const [metaRes, tagsRes, usersRes] = await Promise.allSettled([
      api.textureMeta(type, hash),
      api.textureTags(type, hash),
      api.textureUsers(type, hash),
    ]);
    if (id !== requestId) return;

    const meta = metaRes.status === "fulfilled" ? metaRes.value : null;
    const tags = tagsRes.status === "fulfilled" ? tagsRes.value?.tags || [] : [];
    const users = usersRes.status === "fulfilled" ? usersRes.value : null;

    body.innerHTML = "";
    const label = TEXTURE_TYPE_LABELS[type] || type;

    if (meta?.name) {
      body.append(el("h3", { className: "nm-texture-info-name", text: `${meta.name} (${label})` }));
    } else {
      body.append(el("h3", { className: "nm-texture-info-name", text: label }));
    }
    if (meta?.description) {
      body.append(el("p", { className: "nm-texture-info-desc", text: meta.description }));
    }
    if (users?.count != null) {
      body.append(
        el("p", {
          className: "nm-texture-info-worn",
          text: `Worn by ${formatNumber(users.count)} player${users.count === 1 ? "" : "s"}`,
        })
      );
    }
    if (tags.length) {
      const chipRow = el("div", { className: "chip-row" });
      for (const tag of tags.slice(0, 12)) {
        chipRow.append(el("span", { className: "pill", text: tag.emoji ? `${tag.emoji} ${tag.name}` : tag.name }));
      }
      body.append(chipRow);
    }
    if (!meta?.name && users?.count == null && !tags.length) {
      body.append(el("p", { className: "nm-empty", text: "No additional info available for this texture." }));
    }
  }

  return { el: nmPanel("Texture info", body), show };
}

/** Lazily loads and shows this player's cape-count leaderboard position; hides itself if unavailable. */
function renderCapeRankPanel(username) {
  const body = el("div", {}, [el("p", { className: "nm-muted", text: "Checking cape rank…" })]);
  const panel = nmPanel("Cape rank", body);

  api.capeRank(username).then(
    (data) => {
      if (!data?.available) {
        panel.remove();
        return;
      }
      body.innerHTML = "";
      const fields = [];
      if (data.position != null) fields.push(fieldRow("Rank", `#${formatNumber(data.position)}`));
      if (data.capeCount != null) fields.push(fieldRow("Capes owned", formatNumber(data.capeCount)));
      if (data.earliestDiscoveredAt) {
        fields.push(fieldRow("Collecting since", formatShortDate(data.earliestDiscoveredAt)));
      }
      if (!fields.length) {
        panel.remove();
        return;
      }
      body.append(renderFieldsList(fields));
    },
    () => panel.remove()
  );

  return panel;
}

function renderCapeThumbs(capes, onSelect) {
  if (!capes.length) {
    return el("p", { className: "nm-empty", text: "No capes recorded." });
  }
  const grid = el("div", { className: "nm-cape-cards" });
  const cards = [];
  for (const cape of capes) {
    const kind = cape.type === "cloak" ? "cloak" : "cape";
    const isButton = Boolean(cape.hash);
    const wrap = el(isButton ? "button" : "div", {
      className: "nm-cape-card",
      type: isButton ? "button" : undefined,
    });
    const preview = el("div", { className: "nm-cape-preview" });
    const nameEl = el("span", { className: "nm-cape-name", text: kind === "cloak" ? "Cloak" : "Cape" });

    if (cape.hash) {
      const img = el("img", { className: "nm-cape-thumb", alt: kind, loading: "lazy" });
      attachCapePreview(img, cape.front_url, cape.hash, kind);
      preview.append(img);

      api.textureMeta(kind, cape.hash).then(
        (meta) => {
          if (meta?.name) nameEl.textContent = meta.name;
        },
        () => {}
      );

      wrap.addEventListener("click", () => {
        for (const b of cards) b.classList.remove("selected");
        wrap.classList.add("selected");
        onSelect?.(kind, cape.hash);
      });
      cards.push(wrap);
    } else {
      preview.append(el("div", { className: "nm-thumb-fallback", text: kind === "cloak" ? "L" : "C" }));
    }
    wrap.append(preview, nameEl);
    grid.append(wrap);
  }
  return grid;
}

const PROVIDER_LABELS = {
  labymod: "LabyMod",
  badlion: "Badlion",
  hypixel: "Hypixel",
  minecraftuuid: "MinecraftUUID",
  namemc: "NameMC",
  mineplex: "Mineplex",
  essentials: "Essentials",
  lunar: "Lunar",
};

function providerLabel(key) {
  return PROVIDER_LABELS[key] || key;
}

function fieldRow(label, value) {
  return el("div", { className: "nm-field" }, [
    el("span", { className: "nm-field-label", text: label }),
    el("span", { className: "nm-field-value", text: value || "—" }),
  ]);
}

function fieldLink(label, href, text) {
  return el("div", { className: "nm-field" }, [
    el("span", { className: "nm-field-label", text: label }),
    el("span", { className: "nm-field-value" }, [
      el("a", { href, target: "_blank", rel: "noreferrer", text: text || href }),
    ]),
  ]);
}

function renderFieldsList(fields) {
  if (!fields.length) {
    return el("p", { className: "nm-empty", text: "No information available." });
  }
  return el("div", { className: "nm-fields" }, fields);
}

function buildGeneralFields(profile, nameData) {
  const fields = [];

  if (profile?.last_server) {
    fields.push(
      fieldRow(
        "Last server",
        `${profile.last_server}${profile.last_server_source ? ` (${providerLabel(profile.last_server_source)})` : ""}`
      )
    );
  }

  if (nameData?.last_seen_at) {
    fields.push(fieldRow("Cache updated", formatDate(nameData.last_seen_at)));
  }

  if (profile?.sources_ok?.length) {
    fields.push(
      fieldRow("Available sources", profile.sources_ok.map(providerLabel).join(", "))
    );
  }

  if (profile?.sources_failed?.length) {
    fields.push(
      fieldRow("Unavailable sources", profile.sources_failed.map(providerLabel).join(", "))
    );
  }

  return fields;
}

function buildProviderFields(sourceKey, source) {
  if (!source) {
    return [el("p", { className: "nm-empty", text: "No data for this provider." })];
  }

  if (!source.available) {
    return [el("p", { className: "nm-unavailable", text: "This provider has no data for this player." })];
  }

  const fields = [];
  const raw = source.raw || {};

  if (source.username) fields.push(fieldRow("Username", source.username));
  if (source.playtime_seconds != null) {
    fields.push(fieldRow("Playtime", formatDuration(source.playtime_seconds)));
  }
  if (source.join_date) {
    fields.push(fieldRow("First join", formatDate(source.join_date)));
  }
  if (source.last_online) {
    fields.push(fieldRow("Last online", formatDate(source.last_online)));
  }
  if (source.monthly_views != null) {
    fields.push(fieldRow("Views / month", formatNumber(source.monthly_views)));
  }
  if (source.last_server) {
    fields.push(fieldRow("Last server", source.last_server));
  }

  if (sourceKey === "hypixel") {
    if (raw.rank) fields.push(fieldRow("Rank", raw.rank));
    if (raw.networkLevel != null) fields.push(fieldRow("Network level", String(raw.networkLevel)));
    if (raw.achievementPoints != null) {
      fields.push(fieldRow("Achievement points", formatNumber(raw.achievementPoints)));
    }
    if (raw.profile_url) fields.push(fieldLink("Profile", raw.profile_url, "View on Plancke"));
  }

  if (sourceKey === "labymod") {
    if (raw.likes != null) fields.push(fieldRow("Likes", formatNumber(raw.likes)));
    if (raw.profile_url) fields.push(fieldLink("Profile", raw.profile_url, "View on LabyMod"));
  }

  if (sourceKey === "badlion") {
    if (raw.rank) fields.push(fieldRow("Rank", raw.rank));
    if (raw.views_all_time != null) {
      fields.push(fieldRow("Views (all time)", formatNumber(raw.views_all_time)));
    }
    if (raw.profile_url) fields.push(fieldLink("Profile", raw.profile_url, "View on Badlion"));
  }

  if (sourceKey === "namemc" && raw.profile_url) {
    fields.push(fieldLink("Profile", raw.profile_url, "View on NameMC"));
  }

  if (sourceKey === "mineplex" && raw.profile_url) {
    fields.push(fieldLink("Profile", raw.profile_url, "View on Mineplex"));
  }

  if (sourceKey === "minecraftuuid" && raw.profile_url) {
    fields.push(fieldLink("Profile", raw.profile_url, "View on MinecraftUUID"));
  }

  return fields;
}

function renderAdditionalInfoCard(profile, nameData) {
  const sources = profile?.sources || {};
  const providerKeys = PROFILE_SOURCES.filter((k) => sources[k] != null);

  const content = el("div", { className: "nm-info-content" });
  const tabs = el("div", { className: "nm-info-tabs", role: "tablist" });

  const entries = [
    { value: "general", label: "General", unavailable: false },
    ...providerKeys.map((key) => ({
      value: key,
      label: providerLabel(key),
      unavailable: !sources[key]?.available,
    })),
  ];

  const buttons = new Map();

  const renderSelection = (value) => {
    content.innerHTML = "";
    for (const [v, btn] of buttons) {
      btn.classList.toggle("active", v === value);
      btn.setAttribute("aria-selected", v === value ? "true" : "false");
    }
    if (value === "general") {
      content.append(renderFieldsList(buildGeneralFields(profile, nameData)));
      return;
    }
    content.append(renderFieldsList(buildProviderFields(value, sources[value])));
  };

  for (const entry of entries) {
    const btn = el("button", {
      className: `nm-info-tab${entry.unavailable ? " unavailable" : ""}`,
      type: "button",
      role: "tab",
      text: entry.label,
      title: entry.unavailable ? "No data from this provider" : undefined,
      onClick: () => renderSelection(entry.value),
    });
    buttons.set(entry.value, btn);
    tabs.append(btn);
  }

  renderSelection("general");

  return el("div", { className: "nm-additional-info" }, [tabs, content]);
}

function renderNameHistoryTable(nameData) {
  const rows = enrichNameHistory(nameData?.history || []);
  if (!rows.length) {
    return el("p", { className: "nm-empty", text: "No name history recorded." });
  }

  const maxHeld = Math.max(...rows.map((r) => r.heldMs ?? 0), 1);

  const table = el("table", { className: "nm-table" });
  const thead = el("thead", {}, [
    el("tr", {}, [
      el("th", { text: "" }),
      el("th", { text: "Name" }),
      el("th", { text: "Changed" }),
      el("th", { text: "Held" }),
      el("th", { text: "" }),
    ]),
  ]);

  const tbody = el("tbody");
  rows.forEach((row, i) => {
    const censored = row.censored || row.name === "-";
    const num = rows.length - i;
    const isCurrent = i === 0;
    const isOriginal = num === 1 && !row.changed_at;

    const nameCell = el("div", { className: "nm-name-cell" }, [
      censored
        ? el("span", { className: "nm-censored", text: "Hidden" })
        : el("span", { text: row.name }),
      isCurrent ? el("span", { className: "nm-badge nm-badge-current", text: "Current" }) : null,
      isOriginal ? el("span", { className: "nm-badge nm-badge-original", text: "Original" }) : null,
    ]);

    const heldCell = el("td", { className: "nm-held nm-held-cell" }, [
      el("span", { text: formatHeldDuration(row.heldMs) }),
    ]);
    if (row.heldMs != null && row.heldMs > 0) {
      const pct = Math.max(3, Math.round((row.heldMs / maxHeld) * 100));
      heldCell.append(
        el("span", { className: "nm-held-meter", "aria-hidden": "true" }, [
          el("span", { className: "nm-held-fill", style: `width:${pct}%` }),
        ])
      );
    }

    tbody.append(
      el("tr", {}, [
        el("td", { className: "nm-num", text: String(num) }),
        el("td", { className: "nm-name" }, [nameCell]),
        el("td", { className: "nm-date", text: row.changed_at ? formatShortDate(row.changed_at) : "—" }),
        heldCell,
        el("td", {}, [censored ? null : copyButton("Copy", row.name)]),
      ])
    );
  });

  table.append(thead, tbody);
  return el("div", { className: "nm-table-wrap" }, [table]);
}

function renderSkinHistoryTable(skinData) {
  const rows = [...(skinData?.history || [])].reverse();
  if (!rows.length) {
    return el("p", { className: "nm-empty", text: "No skin history recorded." });
  }

  const table = el("table", { className: "nm-table" });
  const tbody = el("tbody");
  rows.forEach((row, i) => {
    const thumb = el("img", { className: "nm-table-skin", alt: "", loading: "lazy" });
    attachSkinPreview(thumb, row.skin_hash, row.slim, row.file_hash, row.head_url);
    tbody.append(
      el("tr", {}, [
        el("td", { className: "nm-num", text: String(rows.length - i) }),
        el("td", {}, [el("div", { className: "nm-table-skin-wrap" }, [thumb])]),
        el("td", { className: "nm-date", text: row.changed_at ? formatShortDate(row.changed_at) : "—" }),
        el("td", {
          text: row.slim == null ? "—" : row.slim ? "Slim" : "Classic",
        }),
        el("td", { className: "nm-date", text: formatShortDate(row.observed_at) }),
      ])
    );
  });

  table.append(
    el("thead", {}, [
      el("tr", {}, [
        el("th", { text: "" }),
        el("th", { text: "Skin" }),
        el("th", { text: "Changed" }),
        el("th", { text: "Model" }),
        el("th", { text: "Recorded" }),
      ]),
    ]),
    tbody
  );
  return el("div", { className: "nm-table-wrap" }, [table]);
}

const COSMETIC_ICON_TYPES = new Set(["skin", "cape", "cloak", "bandana"]);

function cosmeticIcon(item) {
  const wrap = el("div", { className: "nm-cosmetic-icon" });
  const type = String(item.type || "").toLowerCase();
  if (!item.hash || !COSMETIC_ICON_TYPES.has(type)) return null;

  const img = el("img", { alt: type, loading: "lazy" });
  if (type === "skin") {
    attachSkinPreview(img, item.hash, item.slim, item.file_hash);
  } else if (type === "cape" || type === "cloak") {
    attachCapePreview(img, item.front_url, item.hash, type);
  } else {
    return null;
  }
  wrap.append(img);
  return wrap;
}

function renderAllCosmetics(cosmeticsBySource) {
  const sections = [];
  for (const entry of cosmeticsBySource || []) {
    const snap = entry?.data?.cosmetics;
    if (!snap) continue;
    const items = snap.cosmetics || [];

    const list = el("div", { className: "nm-cosmetic-items" });
    if (!snap.available) {
      list.append(
        el("p", {
          className: "nm-unavailable",
          text: snap.error || "Not available for this provider.",
        })
      );
    } else if (!items.length) {
      list.append(el("p", { className: "nm-muted", text: "No cosmetics found." }));
    } else {
      for (const item of items) {
        const type = String(item.type || "item");
        const icon = cosmeticIcon(item);
        const nameText = item.name || type.charAt(0).toUpperCase() + type.slice(1);
        list.append(
          el("div", { className: "nm-cosmetic-row" }, [
            icon,
            el("span", {
              className: "nm-cosmetic-type",
              text: nameText,
              style: item.color ? `color:${item.color}` : undefined,
            }),
            item.active != null
              ? el("span", {
                  className: `nm-provider-badge ${item.active ? "ok" : ""}`,
                  text: item.active ? "Equipped" : "Stored",
                })
              : null,
            el("span", {
              className: "nm-muted",
              text: item.first_seen_at
                ? formatShortDate(item.first_seen_at)
                : item.last_seen_at
                  ? formatShortDate(item.last_seen_at)
                  : "",
            }),
          ])
        );
      }
    }

    sections.push(
      el("div", { className: "nm-cosmetic-source" }, [
        el("h3", { className: "nm-cosmetic-heading", text: providerLabel(entry.source) }),
        list,
      ])
    );
  }

  if (!sections.length) {
    return el("p", { className: "nm-empty", text: "No cosmetics data from any provider." });
  }
  return el("div", { className: "nm-cosmetic-sections" }, sections);
}

function statTile(label, value, sub = null) {
  return el("div", { className: "nm-stat" }, [
    el("span", { className: "nm-stat-label", text: label }),
    el("span", { className: "nm-stat-value", text: value }),
    sub ? el("span", { className: "nm-stat-sub", text: sub }) : null,
  ]);
}

function renderStats({ profile, nameData, skinCount }) {
  const tiles = [];

  const nameCount = (nameData?.history || []).length;
  if (nameCount) tiles.push(statTile("Names", String(nameCount)));
  if (skinCount) tiles.push(statTile("Skins", String(skinCount)));

  if (profile?.est_playtime_seconds != null) {
    tiles.push(statTile("Est. playtime", formatDuration(profile.est_playtime_seconds)));
  }

  if (profile?.monthly_views != null) {
    tiles.push(
      statTile(
        "Views / month",
        formatNumber(profile.monthly_views),
        profile.monthly_views_source ? `via ${providerLabel(profile.monthly_views_source)}` : null
      )
    );
  }

  if (profile?.est_last_online) {
    tiles.push(
      statTile(
        "Last online",
        formatShortDate(profile.est_last_online),
        profile.est_last_online_source ? `via ${providerLabel(profile.est_last_online_source)}` : null
      )
    );
  }

  if (!tiles.length) return null;
  return el("div", { className: "nm-stats" }, tiles);
}

function renderProfileHeader(username, uuid) {
  const avatar = el("img", {
    className: "nm-avatar",
    src: avatarUrl(uuid, 144),
    alt: "",
    loading: "eager",
  });
  avatar.addEventListener("error", () => avatar.remove());

  return el("header", { className: "nm-profile-header" }, [
    avatar,
    el("div", { className: "nm-profile-id" }, [
      el("h1", { className: "nm-username", text: username }),
      uuid
        ? el("div", { className: "nm-uuid-row" }, [
            el("code", { className: "nm-uuid", text: uuid }),
            copyButton("Copy", uuid),
          ])
        : null,
    ]),
  ]);
}

export function renderProfilePage({ username, uuid, profile, nameData, skinData, cosmeticsBySource }) {
  const capes = collectCapes(cosmeticsBySource);
  const skinCount = new Set((skinData?.history || []).map((r) => r.skin_hash?.toLowerCase()).filter(Boolean)).size;

  const textureInfo = renderTextureInfoPanel();
  const onSelectTexture = (type, hash) => textureInfo.show(type, hash);

  const page = el("article", { className: "nm-profile" }, [
    renderProfileHeader(username, uuid),

    profileNeedsCaptcha(profile)
      ? el("div", { className: "nm-alert" }, [
          el("strong", { text: "Verification needed" }),
          el("span", { text: " — complete the captcha to unlock full provider data." }),
        ])
      : null,

    renderStats({ profile, nameData, skinCount }),

    el("div", { className: "nm-layout" }, [
      el("aside", { className: "nm-sidebar" }, [
        nmPanel("Skin", renderSkinViewer(uuid, username)),
        nmPanel(`Skins · ${skinCount || skinData?.history?.length || 0}`, renderSkinThumbs(skinData?.history, onSelectTexture)),
        nmPanel(`Capes · ${capes.length}`, renderCapeThumbs(capes, onSelectTexture)),
        textureInfo.el,
        renderCapeRankPanel(username),
      ]),

      el("div", { className: "nm-main" }, [
        nmPanel("Name history", renderNameHistoryTable(nameData)),
        nmPanel("Details", renderAdditionalInfoCard(profile, nameData)),
        nmPanel("Skin history", renderSkinHistoryTable(skinData)),
        nmPanel("Cosmetics", renderAllCosmetics(cosmeticsBySource)),
      ]),
    ]),
  ]);

  return page;
}
