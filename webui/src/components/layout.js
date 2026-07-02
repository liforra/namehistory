import { APP_NAME } from "../config.js";
import { isAdminLoggedIn } from "../lib/auth.js";
import { el } from "../lib/format.js";
import { navigate } from "../lib/router.js";

export function renderNavbar(active = "/") {
  const links = [
    { href: "#/", label: "Lookup", path: "/" },
    { href: "#/docs", label: "API Docs", path: "/docs" },
    { href: "#/admin", label: isAdminLoggedIn() ? "Admin" : "Admin Login", path: "/admin" },
  ];

  return el("header", { className: "site-header" }, [
    el("div", { className: "container header-inner" }, [
      el("a", { className: "brand", href: "#/", onClick: (e) => { e.preventDefault(); navigate("/"); } }, [
        el("span", { className: "brand-mark", text: "PH" }),
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
      el("p", { text: "Public frontend for the Player History API. Not affiliated with Mojang or Microsoft." }),
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
