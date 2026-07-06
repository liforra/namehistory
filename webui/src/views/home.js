import { api, ApiError } from "../lib/api.js";
import { pageShell } from "../components/layout.js";
import { ensureChallengeSatisfied, profileNeedsCaptcha } from "../lib/challenge_gate.js";
import { el } from "../lib/format.js";
import { navigate, queryParam } from "../lib/router.js";
import { COSMETIC_SOURCES } from "../config.js";
import { renderProfilePage } from "./profile.js";

const EXAMPLE_NAMES = ["Notch", "jeb_", "Dinnerbone", "Technoblade"];

const FEATURES = [
  {
    icon: "📜",
    title: "Name history",
    text: "Every rename with dates and how long each name was held, merged from multiple sources.",
  },
  {
    icon: "🎨",
    title: "Skin timeline",
    text: "Past skins with previews, model type, and when each was first seen.",
  },
  {
    icon: "🧩",
    title: "Client profiles",
    text: "LabyMod, Badlion, Lunar, and Essentials cosmetics and capes in one place.",
  },
  {
    icon: "📊",
    title: "Server stats",
    text: "Playtime, views, last seen, and per-provider data from Hypixel and more.",
  },
];

function parseQuery(input) {
  const value = (input || "").trim();
  if (!value) return { username: "", uuid: "" };
  const uuidRe = /^[0-9a-fA-F-]{32,36}$/;
  if (uuidRe.test(value.replace(/-/g, ""))) {
    return { username: "", uuid: value };
  }
  return { username: value, uuid: "" };
}

function renderSearchBar(initial = "", compact = false) {
  const form = el("form", { className: compact ? "nm-search nm-search-compact" : "nm-search" });
  const input = el("input", {
    className: "nm-search-input",
    name: "q",
    type: "search",
    placeholder: "Username or UUID…",
    value: initial,
    autocomplete: "off",
    spellcheck: "false",
    required: true,
  });
  const submit = el("button", { className: "nm-search-btn", type: "submit", text: "Search" });

  form.append(el("div", { className: "nm-search-row" }, [input, submit]));
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    navigate(`/?q=${encodeURIComponent(q)}`);
  });
  return form;
}

function renderLanding() {
  return el("div", {}, [
    el("section", { className: "nm-landing" }, [
      el("h1", { text: "Minecraft player history" }),
      el("p", {
        text: "Name changes, skin timelines, capes, and client profiles — aggregated from every major source.",
      }),
      renderSearchBar(),
      el("div", { className: "nm-examples" }, [
        el("span", { text: "Try:" }),
        ...EXAMPLE_NAMES.map((name) =>
          el("button", {
            className: "nm-example",
            type: "button",
            text: name,
            onClick: () => navigate(`/?q=${encodeURIComponent(name)}`),
          })
        ),
      ]),
    ]),
    el("div", { className: "nm-features" }, FEATURES.map((f) =>
      el("div", { className: "nm-feature" }, [
        el("div", { className: "nm-feature-icon", text: f.icon }),
        el("h3", { text: f.title }),
        el("p", { text: f.text }),
      ])
    )),
  ]);
}

function renderError(message, query) {
  const section = el("section", { className: "nm-alert nm-alert-error" }, [
    el("strong", { text: "Lookup failed" }),
    el("p", { text: message }),
  ]);
  section.append(
    el("div", { className: "nm-error-actions" }, [
      el("button", {
        className: "btn btn-ghost btn-sm",
        type: "button",
        text: "Retry",
        onClick: () => navigate(`/?q=${encodeURIComponent(query)}&r=${Date.now()}`),
      }),
    ])
  );
  return section;
}

/** Skeleton mirroring the profile layout so content doesn't jump on load. */
function renderSkeleton() {
  const sk = (cls) => el("div", { className: `sk ${cls}` });
  return el("div", { className: "nm-skeleton", "aria-busy": "true", "aria-label": "Loading player data" }, [
    el("div", { className: "nm-skeleton-header" }, [
      sk("sk-avatar"),
      el("div", {}, [sk("sk-line sk-line-lg"), sk("sk-line sk-line-sm")]),
    ]),
    el("div", { className: "nm-stats" }, Array.from({ length: 4 }, () => sk("sk-stat"))),
    el("div", { className: "nm-layout" }, [
      el("div", { className: "nm-sidebar" }, [sk("sk-panel sk-panel-tall"), sk("sk-panel")]),
      el("div", { className: "nm-main" }, [sk("sk-panel"), sk("sk-panel sk-panel-tall")]),
    ]),
  ]);
}

async function loadPlayer(username, uuid) {
  await ensureChallengeSatisfied();

  let profile = await api.playerProfile(username, uuid).catch(() => null);
  if (profileNeedsCaptcha(profile)) {
    await ensureChallengeSatisfied(profile);
    profile = await api.playerProfile(username, uuid, true).catch(() => profile);
  }

  const [nameData, skinData, ...cosmeticResults] = await Promise.all([
    api.playerNameHistory(username, uuid),
    api.playerSkinHistory(username, uuid),
    ...COSMETIC_SOURCES.map((source) =>
      api.playerCosmeticsSource(source, username, uuid).catch(() => null)
    ),
  ]);

  const cosmeticsBySource = COSMETIC_SOURCES.map((source, i) => ({
    source,
    data: cosmeticResults[i],
  }));

  return { profile, nameData, skinData, cosmeticsBySource };
}

export async function renderHome(root) {
  const q = queryParam("q") || "";
  const content = el("div", { className: "nm-page" });

  if (!q) {
    content.append(renderLanding());
    root.append(pageShell("/", content));
    return;
  }

  content.append(renderSearchBar(q, true));
  content.append(renderSkeleton());
  root.append(pageShell("/", content));

  const { username, uuid } = parseQuery(q);
  try {
    const data = await loadPlayer(username, uuid);
    content.removeChild(content.lastChild);

    const resolvedUuid = data.nameData?.uuid || data.profile?.uuid || uuid;
    const resolvedName = data.nameData?.query || data.profile?.query || username;

    content.append(
      renderProfilePage({
        username: resolvedName,
        uuid: resolvedUuid,
        profile: data.profile,
        nameData: data.nameData,
        skinData: data.skinData,
        cosmeticsBySource: data.cosmeticsBySource,
      })
    );
  } catch (err) {
    content.removeChild(content.lastChild);
    content.append(renderError(err instanceof ApiError ? err.message : "Unexpected error", q));
  }
}
