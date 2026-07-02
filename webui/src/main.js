import "./style.css";
import { renderNavbar } from "./components/layout.js";
import { route, startRouter } from "./lib/router.js";
import { ensureChallengeSatisfied } from "./lib/challenge_gate.js";
import { renderHome } from "./views/home.js";
import { renderDocs } from "./views/docs.js";
import { renderAdmin } from "./views/admin.js";

route("/", renderHome);
route("/docs", renderDocs);
route("/admin", renderAdmin);

startRouter((root) => {
  root.append(renderNavbar("/"));
  const main = document.createElement("main");
  main.className = "container main";
  main.innerHTML = `<section class="card"><h1>Page not found</h1><p><a href="#/">Return home</a></p></section>`;
  root.append(main);
});

// If the backend requires a provider captcha token, show it immediately
// so the user can solve before doing lookups.
ensureChallengeSatisfied().catch(() => {
  // ignore; lookup screens will re-check
});
