// Hosted site only: the visitor's own camera. One JPEG frame a second goes to /api/frame,
// which returns the detections and an annotated picture. The camera itself never leaves the browser.
(function () {
  const start = document.getElementById("bc-start");
  if (!start) return;
  const stop = document.getElementById("bc-stop");
  const status = document.getElementById("bc-status");
  const msg = document.getElementById("bc-msg");
  const wrap = document.getElementById("bc-wrap");
  const feed = document.getElementById("bc-feed");
  const video = document.getElementById("bc-video");
  const grab = document.getElementById("bc-grab");
  const counts = document.getElementById("bc-counts");

  // A random id per browser tab, so each visitor gets their own track IDs. No personal data.
  const session = (() => {
    const fresh = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
    try {
      let s = sessionStorage.getItem("bc-session");
      if (!s) { s = fresh(); sessionStorage.setItem("bc-session", s); }
      return s;
    } catch (e) { return fresh(); }
  })();

  let stream = null, timer = null, busy = false, frames = 0, people = 0, animals = 0;

  const setStatus = (text, cls) => { status.textContent = text; status.className = "pill " + (cls || ""); };
  const fail = (text) => { msg.hidden = false; msg.textContent = text; };

  async function sendFrame() {
    if (busy || !stream || !video.videoWidth) return;
    busy = true;
    try {
      const scale = Math.min(1, 640 / video.videoWidth);
      grab.width = Math.round(video.videoWidth * scale);
      grab.height = Math.round(video.videoHeight * scale);
      grab.getContext("2d").drawImage(video, 0, 0, grab.width, grab.height);
      const blob = await new Promise((resolve) => grab.toBlob(resolve, "image/jpeg", 0.8));
      const r = await fetch("/api/frame?session=" + encodeURIComponent(session), {
        method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob, cache: "no-store",
      });
      if (r.status === 429) return;            // previous frame still running: skip this one
      const j = await r.json();
      if (!r.ok) { fail(j.error || ("Error " + r.status)); return; }
      msg.hidden = true;
      feed.src = j.image;
      wrap.hidden = false;
      frames += 1;
      animals += j.animals.length;
      people += j.persons;
      counts.textContent = `Frames analysed: ${frames}. Animal boxes seen: ${animals}. People seen (never logged as animals): ${people}.`;
    } catch (e) {
      fail("Could not reach the site: " + e.message);
    } finally {
      busy = false;
    }
  }

  function stopCamera() {
    clearInterval(timer); timer = null;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    start.disabled = false; stop.disabled = true;
    setStatus("stopped");
  }

  start.addEventListener("click", async () => {
    msg.hidden = true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      fail("This browser cannot open a camera here (it needs https or localhost).");
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    } catch (e) {
      fail("Camera not started (" + e.name + "). Allow camera access for this site and try again.");
      return;
    }
    video.srcObject = stream;
    await video.play();
    start.disabled = true; stop.disabled = false;
    setStatus("running", "ok");
    timer = setInterval(sendFrame, 1000);
  });
  stop.addEventListener("click", stopCamera);
  window.addEventListener("pagehide", () => { if (stream) stopCamera(); });
})();
