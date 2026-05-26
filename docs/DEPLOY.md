# Deploy guide

Production deploy uses two free-tier-friendly platforms:

| Layer    | Host    | Why                                                                                       |
|----------|---------|-------------------------------------------------------------------------------------------|
| Database | Neon    | Serverless Postgres + `pgvector` preinstalled, free 0.5 GB / 5 h compute per month        |
| Backend  | Fly.io  | Docker-native, has a region (`fra`) next to Neon `eu-central-1` so DB latency stays low   |
| Frontend | Vercel  | First-class Next.js, free Hobby plan, GitHub-PR previews                                  |

## Prerequisites

- `flyctl` installed locally — https://fly.io/docs/flyctl/install/
- A Fly.io account with a card on file (required since 2024 — pay-as-you-go, ~$2-4/month for 1 GB shared CPU machine that auto-stops on idle).
- A Vercel account (free Hobby plan is enough).
- `POSTGRES_DSN` from Neon already populated and the corpus already ingested (`cdrag ingest`). The BM25 index rebuilds itself from Postgres on first boot, so you do not need to ship `data/bm25_index/`.

## 1. Deploy the backend to Fly.io

The repo already contains `Dockerfile` (multi-stage, non-root, model weights pre-cached at build time) and `fly.toml` (pinned to `fra` region, 1 GB memory, auto-stop on idle).

```powershell
flyctl auth login

# First time only — creates the app on Fly.io without deploying. If the name is
# taken, edit `app =` in fly.toml.
flyctl launch --no-deploy --copy-config --name claude-docs-rag --region fra

# Set secrets (do NOT commit these).
flyctl secrets set `
    POSTGRES_DSN="postgresql://...neon.tech/neondb?sslmode=require" `
    ANTHROPIC_API_KEY="sk-ant-..." `
    CDRAG_CORS_ORIGINS="https://claude-docs-rag.vercel.app"

# Build + deploy. First build is ~6-8 min (pre-caches model weights ~150 MB).
flyctl deploy

# Watch logs.
flyctl logs

# Smoke-check from your laptop.
curl https://claude-docs-rag.fly.dev/healthz
```

`/healthz` returns the document count and BM25 status. First request after a cold start can take ~5-10 s while the BM25 index is loaded from disk and the models are paged into RAM; subsequent requests stay in the ~3-4 s range observed locally.

## 2. Deploy the frontend to Vercel

The frontend lives in `web/`. Vercel auto-detects Next.js.

1. https://vercel.com → New Project → import `alvarocanoo/claude-docs-rag`.
2. **Root Directory**: `web` (important — repo root is the Python project).
3. **Framework Preset**: Next.js (auto-detected).
4. **Environment Variables**:
   - `NEXT_PUBLIC_API_BASE_URL` = `https://claude-docs-rag.fly.dev`
5. Deploy.

After the first deploy, copy the Vercel URL back into the Fly.io secret so CORS allows it:

```powershell
flyctl secrets set CDRAG_CORS_ORIGINS="https://claude-docs-rag.vercel.app"
# Fly.io restarts the machine automatically when a secret changes.
```

## 3. Verify end-to-end

```powershell
# Backend
curl https://claude-docs-rag.fly.dev/healthz

# Hybrid search through the API
$body = @{ query = "How do I stream messages from the Claude API?"; k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri https://claude-docs-rag.fly.dev/search -Method POST -ContentType "application/json" -Body $body

# Frontend
Start-Process https://claude-docs-rag.vercel.app
```

## Operational notes

- **Cost on Fly.io**: a `shared-cpu-1x` / 1 GB machine costs roughly $1.94/month if it ran 24/7. With `auto_stop_machines = "stop"` (set in `fly.toml`) it idles to $0 when there's no traffic and spins back up on the next request (~5-10 s cold start).
- **Neon limits**: free tier sleeps the DB after 5 minutes of inactivity. First query after a sleep adds ~500 ms of wakeup latency. This is acceptable for portfolio traffic.
- **Cold start budget**: the Dockerfile pre-caches the embedder + reranker weights, so the first request only pays for BM25 index load (~1-2 s on 42k chunks) and Python interp warmup. Total first-request budget on a cold machine: ~5-10 s.
- **Rolling back**: `flyctl releases` lists deploys, `flyctl deploy --image registry.fly.io/claude-docs-rag:deployment-<id>` rolls to a specific image.
