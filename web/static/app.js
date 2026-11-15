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
      counts.textContent = `Animal detections (sampled frames): ${j.counts.animal}, people: ${j.counts.person}, ` +
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
