import { COSMETIC_SOURCES, PROFILE_SOURCES } from "../config.js";
import { profileNeedsCaptcha } from "../lib/challenge_gate.js";
import {
  attachSkinPreview,
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

function renderSkinViewer(uuid, username, skinRows) {
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

function renderSkinThumbs(skinRows) {
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
  for (const row of unique) {
    const thumb = el("img", {
      className: "nm-skin-thumb",
      alt: "Skin",
      loading: "lazy",
    });
    attachSkinPreview(thumb, row.skin_hash, row.slim, row.file_hash, row.head_url);
    grid.append(
      el("div", { className: "nm-thumb-wrap", title: formatShortDate(row.changed_at) }, [thumb])
    );
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

function renderCapeThumbs(capes) {
  if (!capes.length) {
    return el("p", { className: "nm-empty", text: "No capes recorded." });
  }
  const grid = el("div", { className: "nm-thumb-grid nm-cape-grid" });
  for (const cape of capes) {
    const wrap = el("div", { className: "nm-thumb-wrap" });
    if (cape.hash) {
      const img = el("img", { className: "nm-cape-thumb", alt: "Cape", loading: "lazy" });
      attachSkinPreview(img, cape.hash);
      wrap.append(img);
    } else {
      wrap.append(el("div", { className: "nm-thumb-fallback", text: "C" }));
    }
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
  const uuid = profile?.uuid || nameData?.uuid || "";
  const fields = [];

  if (uuid) {
    fields.push(
      el("div", { className: "nm-field" }, [
        el("span", { className: "nm-field-label", text: "UUID" }),
        el("div", { className: "nm-field-value nm-uuid-row" }, [
          el("code", { className: "nm-uuid", text: uuid }),
          copyButton("Copy", uuid),
        ]),
      ])
    );
  }

  if (profile?.monthly_views != null) {
    const src = profile.monthly_views_source;
    fields.push(
      fieldRow(
        "Views",
        `${formatNumber(profile.monthly_views)} / month${src ? ` (${providerLabel(src)})` : ""}`
      )
    );
  }

  if (profile?.est_playtime_seconds != null) {
    fields.push(fieldRow("Est. playtime", formatDuration(profile.est_playtime_seconds)));
  }

  if (profile?.est_last_online) {
    const src = profile.est_last_online_source;
    fields.push(
      fieldRow(
        "Last online",
        `${formatDate(profile.est_last_online)}${src ? ` (${providerLabel(src)})` : ""}`
      )
    );
  }

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

  const select = el("select", { className: "nm-info-select" });
  select.append(el("option", { value: "general", text: "General" }));
  for (const key of providerKeys) {
    const label = providerLabel(key);
    const status = sources[key]?.available ? "" : " (unavailable)";
    select.append(el("option", { value: key, text: `${label}${status}` }));
  }

  const content = el("div", { className: "nm-info-content" });

  const renderSelection = (value) => {
    content.innerHTML = "";
    if (value === "general") {
      content.append(renderFieldsList(buildGeneralFields(profile, nameData)));
      return;
    }
    content.append(renderFieldsList(buildProviderFields(value, sources[value])));
  };

  select.addEventListener("change", () => renderSelection(select.value));
  renderSelection("general");

  return el("div", { className: "nm-additional-info" }, [select, content]);
}

function renderNameHistoryTable(nameData) {
  const rows = enrichNameHistory(nameData?.history || []);
  if (!rows.length) {
    return el("p", { className: "nm-empty", text: "No name history recorded." });
  }

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
    tbody.append(
      el("tr", {}, [
        el("td", { className: "nm-num", text: String(num) }),
        el("td", { className: "nm-name" }, [
          censored
            ? el("span", { className: "nm-censored", text: "Hidden" })
            : el("span", { text: row.name }),
        ]),
        el("td", { className: "nm-date", text: row.changed_at ? formatShortDate(row.changed_at) : "—" }),
        el("td", { className: "nm-held", text: formatHeldDuration(row.heldMs) }),
        el("td", {}, [
          censored ? null : copyButton("Copy", row.name),
        ]),
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
        el("td", { text: row.changed_at ? formatShortDate(row.changed_at) : "—" }),
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

function renderAllCosmetics(cosmeticsBySource) {
  const sections = [];
  for (const entry of cosmeticsBySource || []) {
    const snap = entry?.data?.cosmetics;
    if (!snap) continue;
    const items = snap.cosmetics || [];
    if (!snap.available && !items.length) continue;

    const list = el("div", { className: "nm-cosmetic-items" });
    if (!items.length) {
      list.append(el("p", { className: "nm-muted", text: "No items." }));
    } else {
      for (const item of items) {
        const type = String(item.type || "item");
        list.append(
          el("div", { className: "nm-cosmetic-row" }, [
            el("span", { className: "nm-cosmetic-type", text: type }),
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
        el("h3", { className: "nm-cosmetic-heading", text: entry.source }),
        list,
      ])
    );
  }

  if (!sections.length) {
    return el("p", { className: "nm-empty", text: "No cosmetics data from any provider." });
  }
  return el("div", { className: "nm-cosmetic-sections" }, sections);
}

export function renderProfilePage({ username, uuid, profile, nameData, skinData, cosmeticsBySource }) {
  const capes = collectCapes(cosmeticsBySource);
  const skinCount = new Set((skinData?.history || []).map((r) => r.skin_hash?.toLowerCase()).filter(Boolean)).size;

  const page = el("article", { className: "nm-profile" }, [
    el("header", { className: "nm-profile-header" }, [
      el("h1", { className: "nm-username", text: username }),
    ]),

    profileNeedsCaptcha(profile)
      ? el("div", { className: "nm-alert" }, [
          el("strong", { text: "Verification needed" }),
          el("span", { text: " — complete the captcha to unlock full provider data." }),
        ])
      : null,

    el("div", { className: "nm-layout" }, [
      el("aside", { className: "nm-sidebar" }, [
        nmPanel("Skin", renderSkinViewer(uuid, username, skinData?.history)),
        nmPanel(`Skins (${skinCount || skinData?.history?.length || 0})`, renderSkinThumbs(skinData?.history)),
        nmPanel(`Capes (${capes.length})`, renderCapeThumbs(capes)),
      ]),

      el("div", { className: "nm-main" }, [
        nmPanel("Additional information", renderAdditionalInfoCard(profile, nameData)),
        nmPanel("Name History", renderNameHistoryTable(nameData)),
        nmPanel("Skin History", renderSkinHistoryTable(skinData)),
        nmPanel("Cosmetics", renderAllCosmetics(cosmeticsBySource)),
      ]),
    ]),
  ]);

  return page;
}
