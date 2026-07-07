const STORAGE_KEY = "namehistory_admin_token";

export function getAdminToken() {
  try {
    return sessionStorage.getItem(STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export function setAdminToken(token) {
  try {
    if (token) {
      sessionStorage.setItem(STORAGE_KEY, token);
    } else {
      sessionStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    /* ignore */
  }
}

export function clearAdminToken() {
  setAdminToken("");
}

export function isAdminLoggedIn() {
  return Boolean(getAdminToken());
}
