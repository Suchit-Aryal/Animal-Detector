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
  // Timeline: a bar for the clip with a mark per animal detection; click a mark to seek the video.
  const COLORS = { cow: "#5b9dff", dog: "#f59e0b", goat: "#a78bfa", cat: "#f472b6", sheep: "#34d399" };
  let drawn = false;
  function drawTimeline(duration, markers) {
    const tl = document.getElementById("timeline");
    const track = document.getElementById("tl-track");
    if (!markers.length) { tl.hidden = true; return; }
    tl.hidden = false;
    if (drawn && track.dataset.n === String(markers.length)) return;
    drawn = true;
    track.dataset.n = String(markers.length);
    track.innerHTML = "";
    const classes = [...new Set(markers.map((m) => m.class))];
    for (const m of markers) {
      const d = document.createElement("button");
      d.type = "button";
      d.className = "tl-mark";
      d.style.left = `${Math.min(100, (100 * m.t) / duration)}%`;
      d.style.background = COLORS[m.class] || "#94a3b8";
      d.title = `${m.class} at ${m.t.toFixed(1)} s`;
      d.addEventListener("click", () => {
        const v = document.getElementById("clip");
        if (v) { v.currentTime = m.t; v.play().catch(() => {}); }
      });
      track.appendChild(d);
    }
    document.getElementById("tl-legend").innerHTML = classes.map((c) =>
      `<span class="tl-key"><i style="background:${COLORS[c] || "#94a3b8"}"></i>${c}</span>`).join("");
  }
  function render(j) {
    status.textContent = j.status;
    progress.textContent = j.progress;
    if (j.counts) {
      const scope = (j.outputs[0] && j.outputs[0].kind === "video") ? "in sampled frames" : "in this image";
      counts.textContent = `Animal detections ${scope}: ${j.counts.animal}, people: ${j.counts.person}, ` +
        `animal boxes dropped because they overlap a person: ${j.counts.suppressed}, rows logged: ${j.counts.rows_logged}`;
    }
    if (j.error) errorEl.textContent = j.error;
    if (j.duration && j.markers) drawTimeline(j.duration, j.markers);
    if (j.status === "done") {
      outputs.innerHTML = "";
      for (const o of j.outputs) {
        if (o.kind === "video") {
          outputs.insertAdjacentHTML("beforeend",
            `<video class="out" id="clip" controls src="/outputs/${o.url}"></video>`);
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

// Print buttons (report page).
document.querySelectorAll("[data-print]").forEach((b) => b.addEventListener("click", () => window.print()));

// Zones: draw a rectangle or polygon on a camera feed, name it, save it. Corners are stored as 0..1 of the frame.
(function () {
  const zCard = document.getElementById("zones");
  if (!zCard || !document.getElementById("z-save")) return;  // no zone editor on the read-only hosted site
  const nameEl = document.getElementById("z-name");
  const saveBtn = document.getElementById("z-save");
  const finishBtn = document.getElementById("z-finish");
  const msg = document.getElementById("z-msg");
  const list = document.getElementById("z-list");
  let mode = null, pts = [], saved = [];
  const canvases = () => [...document.querySelectorAll("canvas.zone-canvas")];
  function esc(t) { return String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function pending() {
    if (mode === "rect" && pts.length === 2) {
      const [a, b] = pts;
      return [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];
    }
    if (mode === "poly" && pts.length >= 3) return pts;
    return null;
  }
  function draw() {
    for (const c of canvases()) {
      const w = c.clientWidth, h = c.clientHeight;
      if (c.width !== w) c.width = w;
      if (c.height !== h) c.height = h;
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      for (const z of saved) shape(ctx, c, z.points, "#5b9dff", z.name, false);
      if (mode) {
        const p = pending() || pts;
        if (p.length) shape(ctx, c, p, "#fbbf24", null, true);
      }
    }
  }
  function shape(ctx, c, points, color, label, dashed) {
    if (!points.length) return;
    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.fillStyle = color + "33";
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    points.forEach(([x, y], i) => {
      const px = x * c.width, py = y * c.height;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    if (points.length > 2) ctx.closePath();
    ctx.fill(); ctx.stroke();
    if (label) {
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText(label, points[0][0] * c.width + 6, points[0][1] * c.height + 16);
    }
    ctx.restore();
  }
  function setMode(m) {
    mode = m; pts = [];
    finishBtn.hidden = m !== "poly";
    msg.textContent = m === "rect" ? "Click two opposite corners on the feed."
      : m === "poly" ? "Click each corner in turn, then press Close polygon (3 or more corners)."
      : "Pick Rectangle or Polygon, then click on a camera feed above.";
    refresh();
  }
  function refresh() {
    saveBtn.disabled = !(pending() && nameEl.value.trim());
    draw();
  }
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  document.getElementById("z-cancel").addEventListener("click", () => { setMode(null); nameEl.value = ""; });
  finishBtn.addEventListener("click", () => { if (pts.length >= 3) { msg.textContent = "Name it and save."; refresh(); } });
  nameEl.addEventListener("input", refresh);
  for (const c of canvases()) {
    c.addEventListener("click", (e) => {
      if (!mode) { msg.textContent = "Pick Rectangle or Polygon first."; return; }
      const r = c.getBoundingClientRect();
      const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
      if (mode === "rect") {
        if (pts.length === 2) pts = [];
        pts.push([x, y]);
        if (pts.length === 2) msg.textContent = "Name it and save.";
      } else {
        pts.push([x, y]);
      }
      refresh();
    });
  }
  window.addEventListener("resize", draw);
  if ("ResizeObserver" in window) {
    const ro = new ResizeObserver(draw);
    document.querySelectorAll(".feedwrap").forEach((w) => ro.observe(w));
  }
  document.querySelectorAll("img.feed").forEach((img) => img.addEventListener("load", draw));
  async function loadZones() {
    const r = await fetch("/api/zones", { cache: "no-store" });
    if (r.ok) saved = await r.json();
    list.innerHTML = saved.length ? saved.map((z) =>
      `<tr><td>${esc(z.name)}</td><td>${z.points.length}</td><td><button type="button" class="danger" data-del="${z.id}">Delete</button></td></tr>`
    ).join("") : `<tr><td colspan="3" class="muted">No zones yet.</td></tr>`;
    draw();
  }
  saveBtn.addEventListener("click", async () => {
    const p = pending();
    if (!p || !nameEl.value.trim()) return;
    const r = await fetch("/zones", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: nameEl.value.trim(), points: p }) });
    const j = await r.json();
    if (!r.ok) { msg.textContent = j.error || "Could not save."; return; }
    msg.textContent = `Saved zone "${j.name}". Animals entering it will raise an alert.`;
    nameEl.value = ""; pts = []; setMode(null); await loadZones();
  });
  list.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-del]");
    if (!b) return;
    await fetch(`/zones/${b.dataset.del}/delete`, { method: "POST", headers: { Accept: "application/json" } });
    await loadZones();
  });
  loadZones();
})();
