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

// Live page: counters and an alert banner for new animal detections (polls /api/summary every 5 s).
(function () {
  const alertBox = document.getElementById("alert");
  const cAnimals = document.getElementById("c-animals");
  if (!cAnimals) return;
  const cPeople = document.getElementById("c-people");
  const cLast = document.getElementById("c-last");
  let seenTs = null;      // latest animal timestamp we have already reported
  let seenAlert = null;   // latest missing-animal alert id we have already reported
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
          // A missing-animal alert takes the banner. On the first poll, old alerts are not shown.
          if (!first) {
            newAlert = true;
            alertBox.hidden = false;
            alertBox.innerHTML = `<strong>Missing animal:</strong> ${esc(alert.message)} at ${esc(alert.ts)} ` +
              `<a href="/animals?status=missing">open Animals</a>`;
          }
          seenAlert = alert.id;
        }
        if (latest && latest.ts !== seenTs) {
          // On the first poll, only remember what is already there: no banner for old detections.
          if (!first && !newAlert) {
            alertBox.hidden = false;
            alertBox.innerHTML = `<strong>Animal detected:</strong> ${esc(latest.class)} at ${esc(latest.ts)} ` +
              `<a href="/log?kind=animal">open log</a>`;
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
