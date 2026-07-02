import { pageShell } from "../components/layout.js";
import { COSMETIC_SOURCES, PROFILE_SOURCES } from "../config.js";
import { getApiBase } from "../config.js";
import { el } from "../lib/format.js";

function endpointCard(method, path, description, example) {
  const base = getApiBase() || window.location.origin;
  const full = `${base.replace(/\/$/, "")}${path}`;

  return el("article", { className: "endpoint card" }, [
    el("div", { className: "endpoint-head" }, [
      el("span", { className: `method method-${method.toLowerCase()}`, text: method }),
      el("code", { className: "endpoint-path", text: path }),
    ]),
    el("p", { text: description }),
    el("pre", { className: "code-block", text: example || `curl "${full}"` }),
    el(
      "button",
      {
        className: "btn btn-ghost btn-sm",
        type: "button",
        text: "Copy curl",
        onClick: () => navigator.clipboard.writeText(example || `curl "${full}"`),
      }
    ),
  ]);
}

export function renderDocs(root) {
  const base = getApiBase() || window.location.origin;
  const origin = base.replace(/\/$/, "");

  const content = el("div", { className: "stack-lg docs" }, [
    el("section", { className: "hero compact" }, [
      el("h1", { text: "API documentation" }),
      el("p", {
        className: "lead",
        text: "Reference for the Player History REST API. Public reads are open unless the server configures API keys.",
      }),
    ]),

    el("section", { className: "card" }, [
      el("h2", { text: "Authentication" }),
      el("p", {
        text: "Write and admin routes accept Authorization headers. Admin-Key satisfies both admin and API requirements when configured on the server.",
      }),
      el("pre", {
        className: "code-block",
        text: `Authorization: Admin-Key <your-admin-token>\nAuthorization: API-Key <your-api-token>`,
      }),
    ]),

    el("section", { className: "stack" }, [
      el("h2", { text: "Unified Minecraft API" }),
      endpointCard(
        "GET",
        "/api/minecraft/player/",
        "Discovery document listing actions and provider sources.",
        `curl "${origin}/api/minecraft/player/"`
      ),
      endpointCard(
        "GET",
        "/api/minecraft/player/profile?username=Notch",
        "Aggregated profile with playtime, views, last online, and per-source breakdown.",
        `curl "${origin}/api/minecraft/player/profile?username=Notch"`
      ),
      endpointCard(
        "GET",
        "/api/minecraft/player/namehistory?username=Notch",
        "Name history via the unified route.",
        `curl "${origin}/api/minecraft/player/namehistory?username=Notch"`
      ),
      endpointCard(
        "GET",
        "/api/minecraft/player/skinhistory?username=Notch",
        "Skin history via the unified route.",
        `curl "${origin}/api/minecraft/player/skinhistory?username=Notch"`
      ),
      endpointCard(
        "POST",
        "/api/minecraft/player/update?username=Notch",
        "Force refresh profile, name, and skin data. Requires API or Admin key.",
        `curl -X POST -H "Authorization: Admin-Key YOUR_KEY" "${origin}/api/minecraft/player/update?username=Notch"`
      ),
      el("div", { className: "card" }, [
        el("h3", { text: "Per-source profile endpoints" }),
        el("p", { className: "muted", text: PROFILE_SOURCES.join(", ") }),
        el("pre", {
          className: "code-block",
          text: `GET /api/minecraft/player/profile/<source>?username=<name>`,
        }),
      ]),
      el("div", { className: "card" }, [
        el("h3", { text: "Cosmetics endpoints" }),
        el("p", { className: "muted", text: COSMETIC_SOURCES.join(", ") }),
        el("pre", {
          className: "code-block",
          text: `GET /api/minecraft/player/cosmetics/<source>?username=<name>`,
        }),
      ]),
    ]),

    el("section", { className: "stack" }, [
      el("h2", { text: "Legacy endpoints" }),
      endpointCard(
        "GET",
        "/api/namehistory?username=Notch",
        "Legacy name history lookup.",
        `curl "${origin}/api/namehistory?username=Notch"`
      ),
      endpointCard(
        "POST",
        "/api/namehistory/update",
        "Batch refresh name history. JSON body with username, uuid, usernames, or uuids.",
        `curl -X POST -H "Authorization: Admin-Key YOUR_KEY" -H "Content-Type: application/json" -d '{"usernames":["Notch"]}' "${origin}/api/namehistory/update"`
      ),
      endpointCard(
        "DELETE",
        "/api/namehistory/delete?username=Notch",
        "Delete a cached profile and all name history. Admin key required when configured.",
        `curl -X DELETE -H "Authorization: Admin-Key YOUR_KEY" "${origin}/api/namehistory/delete?username=Notch"`
      ),
      endpointCard(
        "GET",
        "/api/skinhistory?username=Notch",
        "Legacy skin history lookup.",
        `curl "${origin}/api/skinhistory?username=Notch"`
      ),
      endpointCard(
        "POST",
        "/api/skinhistory/update",
        "Batch refresh skin history.",
        `curl -X POST -H "Authorization: Admin-Key YOUR_KEY" -H "Content-Type: application/json" -d '{"usernames":["Notch"]}' "${origin}/api/skinhistory/update"`
      ),
      endpointCard(
        "DELETE",
        "/api/skinhistory/delete?username=Notch",
        "Delete cached skin history without removing the profile.",
        `curl -X DELETE -H "Authorization: Admin-Key YOUR_KEY" "${origin}/api/skinhistory/delete?username=Notch"`
      ),
    ]),

    el("section", { className: "stack" }, [
      el("h2", { text: "Provider challenges" }),
      endpointCard(
        "GET",
        "/api/minecraft/captcha/status",
        "Challenge token status for LabyMod and other providers.",
        `curl "${origin}/api/minecraft/captcha/status"`
      ),
      endpointCard(
        "POST",
        "/api/minecraft/captcha",
        "Submit a Turnstile or challenge token (JSON: provider, token, ttl_minutes).",
        `curl -X POST -H "Content-Type: application/json" -d '{"provider":"labymod","token":"..."}' "${origin}/api/minecraft/captcha"`
      ),
      endpointCard(
        "POST",
        "/api/minecraft/admin-login",
        "Admin-only challenge token submission. Requires Admin-Key when configured.",
        `curl -X POST -H "Authorization: Admin-Key YOUR_KEY" -H "Content-Type: application/json" -d '{"provider":"labymod","token":"..."}' "${origin}/api/minecraft/admin-login"`
      ),
    ]),

    el("section", { className: "card" }, [
      el("h2", { text: "OpenAPI" }),
      el("p", { text: "Machine-readable specs are served by the API:" }),
      el("ul", {}, [
        el("li", {}, [el("a", { href: `${origin}/api/namehistory/docs.json`, text: "/api/namehistory/docs.json" })]),
        el("li", {}, [el("a", { href: `${origin}/api/skinhistory/docs.json`, text: "/api/skinhistory/docs.json" })]),
      ]),
    ]),
  ]);

  root.append(pageShell("/docs", content));
}
