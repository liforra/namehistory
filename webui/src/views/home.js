import { api, ApiError } from "../lib/api.js";
import { pageShell } from "../components/layout.js";
import { ensureChallengeSatisfied, profileNeedsCaptcha } from "../lib/challenge_gate.js";
import { el } from "../lib/format.js";
import { navigate, queryParam } from "../lib/router.js";
import { COSMETIC_SOURCES } from "../config.js";
import { renderProfilePage } from "./profile.js";

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
    placeholder: "Enter a Minecraft username or UUID",
    value: initial,
    autocomplete: "off",
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
  return el("section", { className: "nm-landing" }, [
    el("h1", { text: "Player lookup" }),
    el("p", { text: "Search any Minecraft username or UUID for name history, skins, capes, and client profiles." }),
    renderSearchBar(),
  ]);
}

function renderError(message) {
  return el("section", { className: "nm-alert nm-alert-error" }, [
    el("strong", { text: "Lookup failed" }),
    el("p", { text: message }),
  ]);
}

function renderLoading() {
  return el("section", { className: "nm-loading" }, [
    el("div", { className: "spinner" }),
    el("p", { text: "Loading player data…" }),
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
  content.append(renderLoading());
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
    content.append(renderError(err instanceof ApiError ? err.message : "Unexpected error"));
  }
}
