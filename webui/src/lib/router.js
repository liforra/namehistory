const routes = new Map();
let currentCleanup = null;

export function route(path, handler) {
  routes.set(path, handler);
}

function getPath() {
  const hash = window.location.hash.replace(/^#/, "") || "/";
  const [path] = hash.split("?");
  return path || "/";
}

export function navigate(path) {
  if (!path.startsWith("#")) {
    window.location.hash = path.startsWith("/") ? path : `/${path}`;
    return;
  }
  window.location.hash = path;
}

export async function startRouter(fallback) {
  const render = async () => {
    if (currentCleanup) {
      currentCleanup();
      currentCleanup = null;
    }

    const path = getPath();
    const handler = routes.get(path) || fallback;
    const root = document.getElementById("app");
    root.innerHTML = "";
    const result = await handler(root, path);
    if (typeof result === "function") currentCleanup = result;
  };

  window.addEventListener("hashchange", render);
  await render();
}

export function queryParam(name) {
  const hash = window.location.hash.replace(/^#/, "");
  const q = hash.includes("?") ? hash.split("?")[1] : window.location.search.slice(1);
  return new URLSearchParams(q).get(name);
}
