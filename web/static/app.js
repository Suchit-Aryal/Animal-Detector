// Shared helpers: theme, lightbox, notifications.
(function () {
  // Theme: the saved choice, or the system setting until the user picks one.
  const root = document.documentElement;
  const btn = document.getElementById("theme");
  function current() {
    return root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }
  if (btn) btn.addEventListener("click", () => {
    const next = current() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) {}
  });

  // Lightbox: any <a class="lb-link" data-group="..." data-cap="..."> opens a viewer with prev/next.
  const lb = document.getElementById("lb");
  if (lb) {
    const img = document.getElementById("lb-img");
    const cap = document.getElementById("lb-cap");
    let items = [], idx = 0;
    function show(i) {
      idx = (i + items.length) % items.length;
      img.src = items[idx].href;
      cap.textContent = `${items[idx].dataset.cap || ""}  (${idx + 1} / ${items.length})`;
      document.getElementById("lb-prev").hidden = items.length < 2;
      document.getElementById("lb-next").hidden = items.length < 2;
    }
    function open(link) {
      const group = link.dataset.group || "";
      items = [...document.querySelectorAll("a.lb-link")].filter((a) => (a.dataset.group || "") === group);
      show(Math.max(0, items.indexOf(link)));
      lb.hidden = false;
    }
    function close() { lb.hidden = true; img.src = ""; }
    document.addEventListener("click", (e) => {
      const a = e.target.closest && e.target.closest("a.lb-link");
      if (a) { e.preventDefault(); open(a); }
    });
    document.getElementById("lb-prev").addEventListener("click", () => show(idx - 1));
    document.getElementById("lb-next").addEventListener("click", () => show(idx + 1));
    document.getElementById("lb-close").addEventListener("click", close);
    lb.addEventListener("click", (e) => { if (e.target === lb) close(); });
    document.addEventListener("keydown", (e) => {
      if (lb.hidden) return;
      if (e.key === "Escape") close();
      if (e.key === "ArrowLeft") show(idx - 1);
      if (e.key === "ArrowRight") show(idx + 1);
    });
  }

  // Notifications: a beep and/or a desktop notification, chosen on the Live page.
  window.notify = function (title, body) {
    let opt = {};
    try { opt = JSON.parse(localStorage.getItem("notify") || "{}"); } catch (e) {}
    if (opt.sound) beep();
    if (opt.desktop && "Notification" in window && Notification.permission === "granted") {
      try { new Notification(title, { body }); } catch (e) {}
    }
  };
  function beep() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.value = 880; g.gain.value = 0.08;
      o.connect(g); g.connect(ctx.destination); o.start();
      setTimeout(() => { o.stop(); ctx.close(); }, 260);
    } catch (e) {}
  }
})();

// Job page: poll /api/job/<id> until the job is done or failed.
(function () {
  const box = document.getElementById("job");
  if (!box) return;
  const id = box.dataset.job;
  const status = document.getElementById("status");
  const progress = document.getElementById("progress");
  const counts = document.getElementById("counts");
  const errorEl = document.getElementById("error");
  const outputs = document.getElementById("outputs");
  function render(j) {
    status.textContent = j.status;
    progress.textContent = j.progress;
    if (j.counts) {
      const scope = (j.outputs[0] && j.outputs[0].kind === "video") ? "in sampled frames" : "in this image";
      counts.textContent = `Animal detections ${scope}: ${j.counts.animal}, people: ${j.counts.person}, ` +
        `animal boxes dropped because they overlap a person: ${j.counts.suppressed}, rows logged: ${j.counts.rows_logged}`;
    }
    if (j.error) errorEl.textContent = j.error;
    if (j.status === "done") {
      outputs.innerHTML = "";
      for (const o of j.outputs) {
        if (o.kind === "video") {
          outputs.insertAdjacentHTML("beforeend",
            `<video class="out" controls src="/outputs/${o.url}"></video>`);
        } else {
          outputs.insertAdjacentHTML("beforeend",
            `<img class="frame" src="/outputs/${o.url}" alt="annotated image">`);
        }
      }
    }
    return j.status === "done" || j.status === "error";
  }
  async function tick() {
    try {
      const r = await fetch(`/api/job/${id}`, { cache: "no-store" });
      if (!r.ok) return;
      if (!render(await r.json())) setTimeout(tick, 1000);
    } catch (e) {
      setTimeout(tick, 2000);
    }
  }
  tick();
})();

// Live page: counters and an alert banner (polls /api/summary every 5 s).
(function () {
  const alertBox = document.getElementById("alert");
  const cAnimals = document.getElementById("c-animals");
  if (!cAnimals) return;
  const cPeople = document.getElementById("c-people");
  const cLast = document.getElementById("c-last");
  let seenTs = null;      // latest animal timestamp we have already reported
  let seenAlert = null;   // latest alert id we have already reported
  let first = true;
  function esc(t) { return String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  async function tick() {
    try {
      const r = await fetch("/api/summary", { cache: "no-store" });
      if (r.ok) {
        const s = await r.json();
        cAnimals.textContent = s.animal.count;
        cPeople.textContent = s.person.count;
        cLast.textContent = s.latest_animal ? `${s.latest_animal.class} · ${s.latest_animal.ts}` : "never";
        const latest = s.latest_animal;
        const alert = s.alert;
        let newAlert = false;
        if (alert && alert.id !== seenAlert) {
          // Alerts take the banner. On the first poll, old alerts are not shown.
          if (!first) {
            newAlert = true;
            alertBox.hidden = false;
            alertBox.innerHTML = `<strong>${esc(alert.title || "Alert")}:</strong> ${esc(alert.message)} at ${esc(alert.ts)} ` +
              `<a href="/animals?status=missing">open Animals</a>`;
            window.notify && window.notify(alert.title || "Alert", alert.message);
          }
          seenAlert = alert.id;
        }
        if (latest && latest.ts !== seenTs) {
          // On the first poll, only remember what is already there: no banner for old detections.
          if (!first && !newAlert) {
            alertBox.hidden = false;
            alertBox.innerHTML = `<strong>Animal detected:</strong> ${esc(latest.class)} at ${esc(latest.ts)} ` +
              `<a href="/log?kind=animal">open log</a>`;
            window.notify && window.notify("Animal detected", latest.class);
          }
          seenTs = latest.ts;
        }
        first = false;
      }
    } catch (e) { /* server busy or restarting: try again */ }
    setTimeout(tick, 5000);
  }
  tick();
})();

// Live page: remember the alert options (sound, desktop) and ask for notification permission once.
(function () {
  const sound = document.getElementById("opt-sound");
  const desktop = document.getElementById("opt-desktop");
  if (!sound || !desktop) return;
  let opt = {};
  try { opt = JSON.parse(localStorage.getItem("notify") || "{}"); } catch (e) {}
  sound.checked = !!opt.sound;
  desktop.checked = !!opt.desktop;
  function save() {
    try { localStorage.setItem("notify", JSON.stringify({ sound: sound.checked, desktop: desktop.checked })); } catch (e) {}
  }
  sound.addEventListener("change", save);
  desktop.addEventListener("change", () => {
    if (desktop.checked && "Notification" in window && Notification.permission === "default") {
      Notification.requestPermission().then((p) => { if (p !== "granted") desktop.checked = false; save(); });
    }
    save();
  });
})();
