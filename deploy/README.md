# Deploy the website to a Hugging Face Space (free CPU)

What you get: a public web page at `https://huggingface.co/spaces/<your-name>/<space-name>`.
It runs the website in a Docker container on the free CPU tier (2 vCPU, 16 GB RAM).
Files on it are wiped when the Space restarts. Nothing here needs a secret or a token.

Only the files in `deploy/` and `web/` are used. The Space folder is built from this repo
by `deploy/build_space.sh`, so the pushed copy always matches the repo.

## 1. Build the Space folder

```bash
cd <repo root>
deploy/build_space.sh          # writes deploy/hf_space/ (about 28 MB) and MODELS.sha256
```

It needs `results/weights/best.pt` and `yolov8n.pt` in the repo. It stops with a message if one is missing.

## 2. Test it on your computer first (needs Docker)

```bash
docker build -t animal-space deploy/hf_space
docker run --rm -p 127.0.0.1:7860:7860 animal-space
```

Open http://127.0.0.1:7860/ . Check that the Live page shows "Your camera (in your browser)", that an
upload works, and that the Stats page has demo data. Stop with Ctrl+C.

## 3. Create the Space

1. Sign in at huggingface.co (you do this in your browser; nothing is stored in this repo).
2. Click **New Space**.
3. Name: anything, for example `animal-surveillance`.
4. **SDK: Docker**, template **Blank**.
5. Hardware: **CPU basic** (free).
6. Visibility: **Public** (anyone with the link can use it) or **Private** (only you).
7. Create the Space. It starts empty.

## 4. Upload the folder

Use one of the two ways below. Model files are large, so Hugging Face stores them with Git LFS (the `.gitattributes` file in the folder does this).

**Option A: git (needs git and git-lfs installed)**

```bash
git lfs install
git clone https://huggingface.co/spaces/<your-name>/<space-name>
cp -r deploy/hf_space/. <space-name>/
cd <space-name>
git add .
git commit -m "Animal Surveillance: first version"
git push
```

When git asks for a password, use a Hugging Face access token that you create yourself in your account settings.
Type it at the prompt or let your git credential helper keep it. Do not paste it into any file or chat.

**Option B: the web page**

On the Space page, use **Files → Add file → Upload files** and upload everything in `deploy/hf_space/`
(including the hidden `.gitattributes` and `.dockerignore`; the page may hide them, so Option A is safer).

## 5. Secrets and settings

None are needed. Do not add any. The Space needs no API keys.

## 6. Check it started

- The Space page shows **Building**, then **Running** (the first build takes 10 to 20 minutes: it downloads torch).
- **Logs**: on the Space page, open **Logs**. Choose **Build** for the Docker build and **Container** for the running app.
  A line `Listening at: http://0.0.0.0:7860` means the website is up. `GET /live 200` lines show visitors.
- Open the Space, go to Live: the demo data should already be on Stats.

## 7. Restart it

- **Settings → Restart Space** restarts the container. Data resets, demo data loads again.
- If a push does not take effect, use **Settings → Factory reboot** (rebuilds from scratch).

## 8. Delete it

Delete the Space after 10 days unless you decided to keep it.
**Settings → Delete this Space**, type the name to confirm. Set a reminder for day 10 when you create it.

## Limits to tell visitors

- Free CPU is slow: one frame takes about a second, so the browser camera checks about once a second.
- On the public demo, Settings and Zones are read-only (403). Locally, anyone with access can change them.
- Uploads and camera frames are processed on the owner's computer, kept up to 24 hours, then deleted (see `web/README.md`).
- Per-visitor limits: 20 uploads an hour, 90 browser frames a minute, 2 analyses at once.
- Data (log, snapshots, zones, settings) is kept in memory and in `/tmp`; it is lost on restart.
- The free Space may go to sleep after a long time with no visitors; the first visit wakes it (a minute or two).
- The model was trained on a limited dataset and is not validated for real monitoring. See `web/README.md`.

## Public demo from your own computer (Cloudflare tunnel)

The tunnel is started by the owner and is not part of this folder. The site listens on `127.0.0.1` only,
so the only way in is through the tunnel.

Build the image from the folder that `deploy/build_space.sh` creates:

```bash
deploy/build_space.sh
docker build -t animal-watch:latest deploy/hf_space
```

Run it (the name, port and limits below are the ones to use):

```bash
docker run -d --name animal-watch --restart unless-stopped \
  -p 127.0.0.1:7860:7860 --memory 6g --cpus 3 animal-watch:latest
```

Check it: `curl -I http://127.0.0.1:7860/live` should give `200`, and `docker logs -f animal-watch` shows the start.
Stop it with `docker stop animal-watch`. The demo data is loaded on each start, and uploads, snapshots and
detection rows are deleted after 24 hours.
