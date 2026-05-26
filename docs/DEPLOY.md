# Deploy guide

The repo is set up for a two-host deploy:

| Layer    | Host                 | Why                                                                                          |
|----------|----------------------|----------------------------------------------------------------------------------------------|
| Database | Neon                 | Serverless Postgres + `pgvector` preinstalled. Free 0.5 GB / 5 h compute per month.          |
| Backend  | **Hugging Face Spaces (Docker SDK)** | Free, no card on file, always-on. URL `huggingface.co/spaces/<user>/claude-docs-rag` is a portfolio signal for AI-eng roles. |
| Frontend | Vercel               | First-class Next.js, free Hobby plan, GitHub-PR previews.                                    |

A Fly.io path is also wired (`fly.toml`, model pre-cache in `Dockerfile`) and is
documented at the bottom as an alternative if you want a custom subdomain or
prefer not to use HF.

## Prerequisites

- The corpus is ingested into Neon already (`uv run cdrag ingest`). The BM25
  index is rebuilt from Postgres on first boot, so the Space image does not
  need to ship `data/bm25_index/`.
- Hugging Face account at https://huggingface.co (free).
- Vercel account at https://vercel.com (free Hobby).
- A `HF_TOKEN` (Settings → Access Tokens → New token → role `write`).

## 1. Backend → Hugging Face Spaces

### 1.1. Create the Space

1. https://huggingface.co/new-space.
2. Owner: your HF user. Space name: `claude-docs-rag`.
3. **License**: MIT. **SDK**: **Docker** (blank template).
4. **Hardware**: free CPU basic (2 vCPU + 16 GB RAM is fine; embedder is 130 MB
   and reranker is 22 MB).
5. Visibility: Public.

The Space starts empty. Code arrives via the GitHub Action below.

### 1.2. Wire the GitHub → HF sync

The workflow `.github/workflows/sync-to-hf-space.yml` pushes the backend tree
to the Space on every commit to `main`. It needs:

- Repository secret `HF_TOKEN` — the write token from step above.
- Repository variable `HF_USER` — your HF username.
- Repository variable `HF_SPACE_NAME` — `claude-docs-rag`.

Set them at https://github.com/alvarocanoo/claude-docs-rag/settings :

- Secrets and variables → Actions → **New repository secret** → name `HF_TOKEN`.
- Secrets and variables → Actions → **Variables** tab → **New repository variable**
  twice, for `HF_USER` and `HF_SPACE_NAME`.

Then either push any commit to `main` or trigger manually:

```powershell
gh workflow run sync-to-hf-space.yml
gh run watch
```

The Space rebuild after sync takes ~6-10 min the first time (image build pulls
~150 MB of model weights). Watch progress at
`https://huggingface.co/spaces/<your-user>/claude-docs-rag` → "Logs".

### 1.3. Set Space-side secrets

In the Space UI → **Settings** → **Variables and secrets**, add:

| Type     | Name                   | Value                                                                                                            |
|----------|------------------------|------------------------------------------------------------------------------------------------------------------|
| Secret   | `POSTGRES_DSN`         | `postgresql://...neon.tech/neondb?sslmode=require`                                                               |
| Variable | `LLM_PROVIDER`         | `groq` (free, default in this repo) or `anthropic` (Haiku/Sonnet/Opus, paid)                                     |
| Secret   | `GROQ_API_KEY`         | `gsk_...` from https://console.groq.com/keys — only if `LLM_PROVIDER=groq`                                       |
| Variable | `GROQ_MODEL`           | `llama-3.3-70b-versatile` for best demo answers or `llama-3.1-8b-instant` for max free-tier quota. Optional.    |
| Secret   | `ANTHROPIC_API_KEY`    | `sk-ant-...` — only if `LLM_PROVIDER=anthropic`. `/search` works without any LLM key.                            |
| Variable | `CDRAG_CORS_ORIGINS`   | `https://<your-vercel-project>.vercel.app` (set after step 2)                                                    |

The Space restarts automatically when secrets change. Smoke-check:

```powershell
curl https://<hf-user>-claude-docs-rag.hf.space/healthz
```

First request after a fresh boot rebuilds the BM25 index from Postgres
(~30 s for 42 k chunks, observed locally). Subsequent requests hit the cached
index and return in the ~3-4 s range.

## 2. Frontend → Vercel

The Next.js app lives in `web/`.

1. https://vercel.com → **Add New… → Project** → import `alvarocanoo/claude-docs-rag`.
2. **Root Directory**: `web` (important — repo root is the Python project).
3. **Framework Preset**: Next.js (auto-detected).
4. **Environment Variables** → Add:
   - `NEXT_PUBLIC_API_BASE_URL` = `https://<hf-user>-claude-docs-rag.hf.space`
5. **Deploy**.

Vercel returns a URL like `https://claude-docs-rag-<hash>.vercel.app`. Copy it
back into the Space secret:

```powershell
# Or via the HF web UI as described in step 1.3.
# CDRAG_CORS_ORIGINS=https://<your-project>.vercel.app
```

Open the Vercel URL — type a question, hit search, you should see real
reranked hits coming from the HF Space backend.

## 3. Verify end-to-end

```powershell
# Backend
curl https://<hf-user>-claude-docs-rag.hf.space/healthz

# Hybrid search via API
$body = @{ query = "How do I stream messages from the Claude API?"; k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "https://<hf-user>-claude-docs-rag.hf.space/search" -Method POST -ContentType "application/json" -Body $body

# Frontend
Start-Process "https://<your-vercel-project>.vercel.app"
```

## Operational notes

- **HF Spaces free tier**: 2 vCPU + 16 GB RAM, always-on. No automatic sleep on
  Docker SDK Spaces. If quotas tighten in the future, the Space will go to
  sleep after long inactivity and wake on the next request.
- **Neon free tier**: the DB suspends after 5 minutes idle. First query after
  suspension adds ~500 ms of wakeup latency.
- **Cold start budget**: the Dockerfile pre-caches embedder + reranker
  (~150 MB), so a fresh container only pays for BM25 index build (~30 s on a
  42 k-chunk corpus) and FastAPI/Python warmup. Total first-request budget:
  ~30-40 s on the very first boot; ~5 s after.
- **Logs**: HF Space → "Logs" tab. Also visible via `huggingface-cli space-info
  <user>/claude-docs-rag`.

## Alternative: Fly.io (custom domain, ~$2/month)

If you'd rather have a `<app>.fly.dev` (or your own domain) and stay on a
"production" PaaS, the repo also contains `fly.toml`. Run:

```powershell
flyctl auth login
flyctl launch --no-deploy --copy-config --name claude-docs-rag --region fra
flyctl secrets set `
    POSTGRES_DSN="postgresql://..." `
    ANTHROPIC_API_KEY="sk-ant-..." `
    CDRAG_CORS_ORIGINS="https://<your-vercel-project>.vercel.app"
flyctl deploy
```

Then point Vercel's `NEXT_PUBLIC_API_BASE_URL` at `https://claude-docs-rag.fly.dev`
instead of the HF Space URL. Auto-stop on idle keeps the bill under ~$2/month
for portfolio-level traffic. Card on file is required since 2024.
