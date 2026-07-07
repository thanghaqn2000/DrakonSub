# DrakonSub — Production deployment

Default stack: **Docker Compose** + **Caddy** reverse proxy (HTTPS) + persistent volumes.

## Prerequisites

- Linux server with Docker Engine and Docker Compose plugin
- `ffmpeg` is bundled in the app image (host `ffmpeg` optional for debugging)
- Minimum **4 GB RAM** recommended (`small` Whisper + translation); **8 GB+** for heavier workloads
- API keys in server-local `.env` (never commit)

## Quick start (internal test, no public HTTPS)

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and/or GEMINI_API_KEY

mkdir -p data/jobs data/whisper-cache
docker compose up -d --build drakonsub
```

App listens on `127.0.0.1:8000` only. SSH tunnel or local curl for smoke test:

```bash
curl -s http://127.0.0.1:8000/api/health | head

Or run the bundled script after `docker compose up`:

```bash
bash deploy/smoke-internal.sh
```
```

## Production with HTTPS (Caddy)

1. Point DNS A record for your domain to the server IP.
2. Set `DRAKONSUB_DOMAIN=sub.example.com` in `.env`.
3. Start app + proxy:

```bash
docker compose --profile proxy up -d --build
```

Caddy obtains TLS certificates automatically. Port **8000** stays bound to localhost; only **80/443** are public.

## Environment variables

See `.env.example`. Production-critical:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Translation / VI repair (when `TRANSLATION_ENGINE=openai`) |
| `GEMINI_API_KEY` | Translation (when `TRANSLATION_ENGINE=gemini`) |
| `TRANSLATION_ENGINE` | `openai` or `gemini` |
| `WHISPER_MODEL` | English ASR model (`small` default) |
| `DRAKONSUB_JOBS_ROOT` | Job uploads/outputs path inside container |
| `DRAKONSUB_DOMAIN` | Public hostname for Caddy |
| `YT_DLP_COOKIES_FILE` | YouTube cookies (Netscape format) for cloud-server imports |

## Storage & cleanup

- `./data/jobs` — uploaded videos, URL imports, rendered outputs, subtitle edits
- `./data/whisper-cache` — Whisper / HuggingFace model cache

Job folders are **not** served by Caddy. Back up or prune `data/jobs` on a schedule; generated media can be large.

## Server inventory (run after SSH access)

```bash
bash deploy/server-inventory.sh | tee inventory-$(date +%Y%m%d).txt
```

Or run commands manually:

```bash
uname -a
cat /etc/os-release
df -h && free -h && nproc
lscpu | head -40
which docker; docker --version; docker compose version
which ffmpeg; ffmpeg -version | head -3
which ffprobe; python3 --version
nvidia-smi || true
```

Sanitize output before sharing; do not paste secrets.

## Security checklist

- [ ] `.env` only on server, permissions `600`
- [ ] Do not expose port `8000` publicly when Caddy is enabled
- [ ] Do not mount `data/jobs` as a static web root
- [ ] `DRAKONSUB_DEBUG=false` in production
- [ ] Rotate API keys if ever leaked in logs

## Update / restart

```bash
git pull
docker compose build drakonsub
docker compose up -d drakonsub
```
