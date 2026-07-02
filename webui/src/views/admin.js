import { pageShell } from "../components/layout.js";
import { api, ApiError } from "../lib/api.js";
import {
  clearAdminToken,
  isAdminLoggedIn,
  setAdminToken,
} from "../lib/auth.js";
import { el, formatDate } from "../lib/format.js";
import { navigate } from "../lib/router.js";

function renderLogin(onSuccess) {
  const errorBox = el("p", { className: "alert alert-error", hidden: true });
  const input = el("input", {
    className: "input",
    type: "password",
    name: "token",
    placeholder: "Admin API token",
    autocomplete: "current-password",
    required: true,
  });

  const form = el("form", { className: "card stack" }, [
    el("h2", { text: "Admin sign in" }),
    el("p", {
      className: "muted",
      text: "Enter the server's admin key (NAMEHISTORY_ADMIN_KEY). It is stored only in this browser session.",
    }),
    el("label", { text: "Admin token" }),
    input,
    errorBox,
    el("button", { className: "btn btn-primary", type: "submit", text: "Sign in" }),
  ]);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorBox.hidden = true;
    const token = input.value.trim();
    if (!token) return;

    try {
      await api.verifyAdminToken(token);
      setAdminToken(token);
      onSuccess();
    } catch (err) {
      errorBox.hidden = false;
      errorBox.textContent =
        err instanceof ApiError
          ? err.message
          : "Could not verify token. Check the key and API URL.";
    }
  });

  return form;
}

function renderChallengeStatus(data) {
  const providers = data?.providers || {};
  const rows = Object.entries(providers).map(([name, info]) => [
    name,
    info?.valid ? "Valid" : info?.needed ? "Required" : "Not set",
    info?.expires_at ? formatDate(info.expires_at) : "—",
  ]);
  if (!rows.length) {
    return el("p", { className: "muted", text: "No provider challenges configured." });
  }
  const table = el("table", { className: "kv-table" });
  const tbody = el("tbody");
  for (const [provider, state, expires] of rows) {
    tbody.append(
      el("tr", {}, [
        el("th", { text: provider }),
        el("td", { text: state }),
        el("td", { text: expires }),
      ])
    );
  }
  table.append(tbody);
  return table;
}

function formatActionResult(result) {
  if (typeof result === "string") return result;
  if (result?.message) return result.message;
  if (Array.isArray(result?.updated)) {
    return `Updated ${result.updated.length} profile(s).`;
  }
  if (result?.uuid) return `Done for ${result.uuid}.`;
  return "Action completed.";
}

function actionCard(title, description, fields, onSubmit) {
  const message = el("p", { className: "form-message muted", hidden: true });
  const form = el("form", { className: "card stack admin-action" }, [
    el("h3", { text: title }),
    el("p", { className: "muted", text: description }),
    ...fields,
    message,
    el("button", { className: "btn btn-secondary", type: "submit", text: "Run" }),
  ]);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    message.hidden = false;
    message.textContent = "Working…";
    try {
      const result = await onSubmit(new FormData(form));
      message.textContent = formatActionResult(result);
      message.className = "form-message ok";
    } catch (err) {
      message.textContent = err instanceof ApiError ? err.message : String(err);
      message.className = "form-message error";
    }
  });

  return form;
}

function field(name, label, opts = {}) {
  const control = opts.type === "textarea"
    ? el("textarea", { className: "input", name, rows: opts.rows || 3, placeholder: opts.placeholder || "" })
    : el("input", {
        className: "input",
        name,
        type: opts.type || "text",
        placeholder: opts.placeholder || "",
        value: opts.value ?? "",
      });
  return el("label", { className: "stack-sm" }, [
    el("span", { text: label }),
    control,
  ]);
}

function renderDashboard(container) {
  container.innerHTML = "";

  const header = el("section", { className: "admin-header" }, [
    el("div", {}, [
      el("h1", { text: "Admin panel" }),
      el("p", {
        className: "muted",
        text: "Authenticated with Admin-Key. Actions call the live API with your session token.",
      }),
    ]),
    el("button", {
      className: "btn btn-ghost",
      type: "button",
      text: "Sign out",
      onClick: () => {
        clearAdminToken();
        navigate("/admin");
        window.location.reload();
      },
    }),
  ]);

  const statusMount = el("div", { className: "stack" });
  statusMount.append(el("p", { className: "muted", text: "Loading challenge status…" }));
  api.captchaStatus()
    .then((data) => {
      statusMount.innerHTML = "";
      statusMount.append(renderChallengeStatus(data));
    })
    .catch((err) => {
      statusMount.innerHTML = "";
      statusMount.append(el("p", { className: "muted", text: err.message || String(err) }));
    });

  const grid = el("div", { className: "admin-grid" }, [
    el("section", { className: "card stack" }, [
      el("h2", { text: "Challenge status" }),
      statusMount,
      el(
        "button",
        {
          className: "btn btn-ghost btn-sm",
          type: "button",
          text: "Refresh status",
          onClick: async () => {
            statusMount.innerHTML = "";
            statusMount.append(el("p", { className: "muted", text: "Loading…" }));
            try {
              statusMount.innerHTML = "";
              statusMount.append(renderChallengeStatus(await api.captchaStatus()));
            } catch (err) {
              statusMount.innerHTML = "";
              statusMount.append(el("p", { className: "muted", text: err.message }));
            }
          },
        }
      ),
    ]),

    actionCard(
      "Refresh player data",
      "Force-update profile, name history, and skin history for one player.",
      [
        field("target", "Username or UUID", { placeholder: "Notch" }),
      ],
      async (fd) => {
        const target = fd.get("target").trim();
        const isUuid = /^[0-9a-fA-F-]{32,36}$/.test(target.replace(/-/g, ""));
        return api.refreshPlayer(isUuid ? "" : target, isUuid ? target : "");
      }
    ),

    actionCard(
      "Batch name history update",
      "POST /api/namehistory/update with usernames or UUIDs.",
      [
        field("usernames", "Usernames (comma-separated)", { placeholder: "Notch, jeb_" }),
        field("uuids", "UUIDs (comma-separated)", { placeholder: "" }),
      ],
      async (fd) => {
        const usernames = fd.get("usernames").split(",").map((s) => s.trim()).filter(Boolean);
        const uuids = fd.get("uuids").split(",").map((s) => s.trim()).filter(Boolean);
        const body = {};
        if (usernames.length) body.usernames = usernames;
        if (uuids.length) body.uuids = uuids;
        return api.updateNameHistory(body);
      }
    ),

    actionCard(
      "Batch skin history update",
      "POST /api/skinhistory/update with usernames or UUIDs.",
      [
        field("usernames", "Usernames (comma-separated)"),
        field("uuids", "UUIDs (comma-separated)"),
      ],
      async (fd) => {
        const usernames = fd.get("usernames").split(",").map((s) => s.trim()).filter(Boolean);
        const uuids = fd.get("uuids").split(",").map((s) => s.trim()).filter(Boolean);
        const body = {};
        if (usernames.length) body.usernames = usernames;
        if (uuids.length) body.uuids = uuids;
        return api.updateSkinHistory(body);
      }
    ),

    actionCard(
      "Delete profile",
      "Removes the profile and all cached name history.",
      [field("target", "Username or UUID")],
      async (fd) => {
        const target = fd.get("target").trim();
        const isUuid = /^[0-9a-fA-F-]{32,36}$/.test(target.replace(/-/g, ""));
        return api.deleteProfile(isUuid ? "" : target, isUuid ? target : "");
      }
    ),

    actionCard(
      "Delete skin history",
      "Clears skin history without deleting the profile.",
      [field("target", "Username or UUID")],
      async (fd) => {
        const target = fd.get("target").trim();
        const isUuid = /^[0-9a-fA-F-]{32,36}$/.test(target.replace(/-/g, ""));
        return api.deleteSkinHistory(isUuid ? "" : target, isUuid ? target : "");
      }
    ),

    actionCard(
      "Submit challenge token",
      "Store a provider challenge token via the admin-login route.",
      [
        field("provider", "Provider", { value: "labymod" }),
        field("token", "Challenge / Turnstile token", { type: "textarea", rows: 4 }),
        field("ttl_minutes", "TTL minutes", { value: "45" }),
      ],
      async (fd) => {
        return api.adminSubmitChallenge({
          provider: fd.get("provider").trim() || "labymod",
          token: fd.get("token").trim(),
          ttl_minutes: Number(fd.get("ttl_minutes")) || 45,
          exchange: true,
        });
      }
    ),
  ]);

  container.append(header, grid);
}

export function renderAdmin(root) {
  const shellContent = el("div", { className: "stack-lg admin-page" });
  const mount = el("div", { className: "stack-lg" });

  if (isAdminLoggedIn()) {
    renderDashboard(mount);
  } else {
    mount.append(
      el("section", { className: "hero compact" }, [
        el("h1", { text: "Administration" }),
        el("p", {
          className: "lead",
          text: "Sign in with your admin API token to refresh data, manage challenge tokens, and delete cached profiles.",
        }),
      ]),
      renderLogin(() => {
        renderDashboard(mount);
      })
    );
  }

  shellContent.append(mount);
  root.append(pageShell("/admin", shellContent));
}
