import { APP_NAME } from "../config.js";
import { isAdminLoggedIn } from "../lib/auth.js";
import { el } from "../lib/format.js";
import { navigate } from "../lib/router.js";

// 4×4 pixel creeper-face mark: 1 = solid, 2 = dim, 0 = off
const MARK_PIXELS = [
  1, 0, 0, 1,
  1, 0, 0, 1,
  2, 1, 1, 2,
  0, 1, 1, 0,
];

function brandMark() {
  return el(
    "span",
    { className: "brand-mark", "aria-hidden": "true" },
    MARK_PIXELS.map((p) =>
      el("i", { className: p === 1 ? "on" : p === 2 ? "dim" : "" })
    )
  );
}

export function renderNavbar(active = "/") {
  const links = [
    { href: "#/", label: "Lookup", path: "/" },
    { href: "#/docs", label: "API Docs", path: "/docs" },
    { href: "#/admin", label: isAdminLoggedIn() ? "Admin" : "Admin Login", path: "/admin" },
  ];

  return el("header", { className: "site-header" }, [
    el("div", { className: "container header-inner" }, [
      el("a", { className: "brand", href: "#/", onClick: (e) => { e.preventDefault(); navigate("/"); } }, [
        brandMark(),
        el("span", { className: "brand-text", text: APP_NAME }),
      ]),
      el("nav", { className: "nav" }, links.map((link) =>
        el("a", {
          className: `nav-link${active === link.path ? " active" : ""}`,
          href: link.href,
        }, link.label)
      )),
    ]),
  ]);
}

export function renderFooter() {
  return el("footer", { className: "site-footer" }, [
    el("div", { className: "container footer-inner" }, [
      el("p", { text: "Not affiliated with Mojang or Microsoft." }),
      el("nav", { className: "footer-links" }, [
        el("a", { href: "#/docs", text: "API" }),
        el("a", {
          href: "https://codeberg.org/liforra/namehistory",
          target: "_blank",
          rel: "noreferrer",
          text: "Codeberg",
        }),
        el("a", {
          href: "https://github.com/liforra/namehistory",
          target: "_blank",
          rel: "noreferrer",
          text: "GitHub",
        }),
      ]),
    ]),
  ]);
}

export function pageShell(active, content) {
  return el("div", { className: "page" }, [
    renderNavbar(active),
    el("main", { className: "container main" }, [content]),
    renderFooter(),
  ]);
}
