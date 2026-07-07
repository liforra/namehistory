import { getApiBase } from "../config.js";
import { getAdminToken } from "./auth.js";

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

function buildUrl(path, params = {}) {
  const base = getApiBase();
  const url = new URL(path, base || window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

async function request(path, options = {}) {
  const {
    method = "GET",
    params,
    body,
    admin = false,
    apiKey = false,
    headers = {},
  } = options;

  const finalHeaders = { Accept: "application/json", ...headers };

  if (body !== undefined) {
    finalHeaders["Content-Type"] = "application/json";
  }

  const token = getAdminToken();
  if (admin && token) {
    finalHeaders.Authorization = `Admin-Key ${token}`;
  } else if (apiKey && token) {
    finalHeaders.Authorization = `API-Key ${token}`;
  } else if (admin || apiKey) {
    throw new ApiError("Authorization required", 401, null);
  }

  const response = await fetch(buildUrl(path, params), {
    method,
    headers: finalHeaders,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const message =
      (data && (data.description || data.message || data.error)) ||
      response.statusText ||
      "Request failed";
    throw new ApiError(message, response.status, data);
  }

  return data;
}

export const api = {
  discovery() {
    return request("/api/minecraft/player/");
  },

  playerProfile(username, uuid, refresh = false) {
    return request("/api/minecraft/player/profile", {
      params: { username, uuid, refresh: refresh ? "1" : undefined },
    });
  },

  playerNameHistory(username, uuid) {
    return request("/api/minecraft/player/namehistory", {
      params: { username, uuid },
    });
  },

  playerSkinHistory(username, uuid) {
    return request("/api/minecraft/player/skinhistory", {
      params: { username, uuid },
    });
  },

  playerProfileSource(source, username, uuid) {
    return request(`/api/minecraft/player/profile/${source}`, {
      params: { username, uuid },
    });
  },

  playerCosmeticsSource(source, username, uuid) {
    return request(`/api/minecraft/player/cosmetics/${source}`, {
      params: { username, uuid },
    });
  },

  capeRank(username, uuid) {
    return request("/api/minecraft/player/cape-rank", {
      params: { username, uuid },
    });
  },

  textureMeta(type, hash) {
    return request(`/api/minecraft/texture/${type}/${hash}/meta`);
  },

  textureUsers(type, hash) {
    return request(`/api/minecraft/texture/${type}/${hash}/users`);
  },

  textureTags(type, hash) {
    return request(`/api/minecraft/texture/${type}/${hash}/tags`);
  },

  refreshPlayer(username, uuid) {
    return request("/api/minecraft/player/update", {
      method: "POST",
      params: { username, uuid },
      admin: true,
    });
  },

  legacyNameHistory(username, uuid) {
    return request("/api/namehistory", { params: { username, uuid } });
  },

  legacySkinHistory(username, uuid) {
    return request("/api/skinhistory", { params: { username, uuid } });
  },

  updateNameHistory(payload) {
    return request("/api/namehistory/update", {
      method: "POST",
      body: payload,
      admin: true,
    });
  },

  updateSkinHistory(payload) {
    return request("/api/skinhistory/update", {
      method: "POST",
      body: payload,
      admin: true,
    });
  },

  deleteProfile(username, uuid) {
    return request("/api/namehistory/delete", {
      method: "DELETE",
      params: { username, uuid },
      admin: true,
    });
  },

  deleteSkinHistory(username, uuid) {
    return request("/api/skinhistory/delete", {
      method: "DELETE",
      params: { username, uuid },
      admin: true,
    });
  },

  captchaStatus() {
    return request("/api/minecraft/captcha/status");
  },

  submitChallengeToken(payload) {
    return request("/api/minecraft/captcha", {
      method: "POST",
      body: payload,
    });
  },

  adminSubmitChallenge(payload) {
    return request("/api/minecraft/admin-login", {
      method: "POST",
      body: payload,
      admin: true,
    });
  },

  async verifyAdminToken(token) {
    const key = "namehistory_admin_token";
    const prev = sessionStorage.getItem(key);
    sessionStorage.setItem(key, token);

    try {
      const response = await fetch(
        buildUrl("/api/namehistory/delete", { username: "__auth_probe__" }),
        {
          method: "DELETE",
          headers: {
            Authorization: `Admin-Key ${token}`,
            Accept: "application/json",
          },
        }
      );

      if (response.status === 401) {
        const body = await response.json().catch(() => ({}));
        throw new ApiError(
          body.description || "Invalid admin token",
          401,
          body
        );
      }

      return true;
    } catch (err) {
      if (prev) sessionStorage.setItem(key, prev);
      else sessionStorage.removeItem(key);
      throw err;
    }
  },
};
