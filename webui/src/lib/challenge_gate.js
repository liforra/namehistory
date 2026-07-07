import { api } from "./api.js";
import { el } from "./format.js";

let overlayEl = null;
let pollTimer = null;
let inFlight = null;

function removeOverlay() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (overlayEl) {
    overlayEl.remove();
    overlayEl = null;
  }
}

function ensureOverlay() {
  if (overlayEl) return overlayEl;

  const status = el("span", { className: "muted", text: "Waiting for solve…" });
  const iframe = el("iframe", {
    className: "challenge-iframe",
    src: "/api/minecraft/captcha",
    loading: "eager",
    title: "Provider captcha",
  });

  const modal = el("div", { className: "challenge-modal" }, [
    el("div", { className: "challenge-modal-head" }, [
      el("div", {}, [
        el("h2", { text: "Verification required" }),
        el("p", {
          className: "muted",
          text: "Complete the check below to unlock full profile data.",
        }),
      ]),
      el("div", { className: "chip-row" }, [status]),
    ]),
    el("div", { className: "challenge-modal-body" }, [iframe]),
  ]);

  overlayEl = el("div", { className: "challenge-overlay" }, [modal]);
  overlayEl._statusEl = status;
  document.body.appendChild(overlayEl);
  return overlayEl;
}

export function profileNeedsCaptcha(profile) {
  const laby = profile?.provider_challenges?.labymod;
  if (!laby) return false;
  if (laby.valid) return false;
  return Boolean(laby.needed);
}

function statusNeedsCaptcha(statusPayload) {
  const providers = statusPayload?.providers || {};
  const laby = providers?.labymod || {};
  return Boolean(laby.needed) && !Boolean(laby.valid);
}

async function captchaStillRequired(profileHint) {
  if (profileHint && profileNeedsCaptcha(profileHint)) return true;
  const status = await api.captchaStatus().catch(() => null);
  return statusNeedsCaptcha(status);
}

/**
 * Shows captcha UI when required and waits until the API reports a valid token.
 */
export async function ensureChallengeSatisfied(profileHint = null) {
  if (!(await captchaStillRequired(profileHint))) {
    removeOverlay();
    return { gated: false };
  }

  if (inFlight) return inFlight;

  inFlight = (async () => {
    const overlay = ensureOverlay();
    overlay._statusEl.textContent = "Captcha required — solve to continue";

    await new Promise((resolve, reject) => {
      const started = Date.now();
      const tick = async () => {
        try {
          const status = await api.captchaStatus();
          if (!statusNeedsCaptcha(status)) {
            removeOverlay();
            resolve();
            return;
          }
          overlay._statusEl.textContent = "Waiting for verification…";
        } catch {
          overlay._statusEl.textContent = "Waiting for API…";
        }

        if (Date.now() - started > 10 * 60 * 1000) {
          removeOverlay();
          reject(new Error("Verification was not completed in time"));
        }
      };

      tick();
      pollTimer = setInterval(tick, 2000);
    });

    return { gated: true };
  })();

  try {
    return await inFlight;
  } finally {
    inFlight = null;
  }
}
