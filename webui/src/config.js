/**
 * Runtime API configuration.
 * Set VITE_API_BASE_URL at build time, or leave empty to call the same origin.
 */
export function getApiBase() {
  const configured = (import.meta.env.VITE_API_BASE_URL || "").trim();
  if (configured) {
    return configured.replace(/\/$/, "");
  }
  return "";
}

export const PROFILE_SOURCES = [
  "labymod",
  "badlion",
  "hypixel",
  "minecraftuuid",
  "namemc",
  "mineplex",
];

export const COSMETIC_SOURCES = ["labymod", "badlion", "essentials", "lunar"];

export const APP_NAME = "Player History";
