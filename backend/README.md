# Plexi Upload Backend

A lightweight Express server that powers the Plexi Materials upload portal. It handles GitHub OAuth authentication and securely creates GitHub Issues on behalf of authenticated users — keeping the existing maintainer-approval workflow fully intact.

## How It Fits Into the System

```
Browser → POST /api/upload → backend creates GitHub Issue → maintainer adds "approved" label
                                                          → add-material.yml fires → file processed
```

The backend never bypasses approval. It simply replaces the manual "redirect to GitHub Issues" step with a seamless in-browser upload experience.

## Prerequisites

- Node.js ≥ 18
- A GitHub **OAuth App** (for user sign-in)
- A GitHub **Personal Access Token** with `Contents: Write` and `Issues: Write` scopes (for the repo)

## Setup

### 1. Create a GitHub OAuth App

1. Go to **GitHub → Settings → Developer Settings → OAuth Apps → New OAuth App**
2. Fill in:
   - **Application name:** Plexi Upload Portal
   - **Homepage URL:** `https://plexi-material.mexus.tech`
   - **Authorization callback URL:** `https://your-backend-domain.com/api/auth/callback`
3. Copy the **Client ID** and generate a **Client Secret**

### 2. Create a Personal Access Token

1. Go to **GitHub → Settings → Developer Settings → Fine-grained tokens**
2. Set scope to the `KunalGupta25/plexi-materials` repository only
3. Grant **Contents: Read & Write** and **Issues: Read & Write**
4. Copy the token

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your values
```

### 4. Install dependencies and start

```bash
npm install
npm start          # production
npm run dev        # development (auto-restarts on file change, Node 18+)
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/auth/github` | Start GitHub OAuth flow |
| `GET` | `/api/auth/callback` | OAuth callback (GitHub redirects here) |
| `GET` | `/api/me` | Get signed-in user (401 if not authenticated) |
| `POST` | `/api/auth/logout` | Destroy session |
| `POST` | `/api/upload` | Upload files and create a GitHub Issue |

### POST `/api/upload`

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `semester` | string | ✅ | e.g. `Semester 5` |
| `subject` | string | ✅ | e.g. `[Sem 5] Enterprise Programming` |
| `fileType` | string | ✅ | e.g. `PDFs` |
| `notes` | string | — | Optional additional notes |
| `files` | File[] | ✅ | Up to 10 files, 25 MB each |

**Success response:**
```json
{
  "ok": true,
  "issueNumber": 42,
  "issueUrl": "https://github.com/KunalGupta25/plexi-materials/issues/42",
  "files": ["chapter1.pdf", "chapter2.pdf"]
}
```

## Deployment on Vercel *(recommended)*

> **Why Vercel works:** The server uses JWT httpOnly cookies instead of server-side sessions, so it's fully stateless and runs correctly on serverless infrastructure.

### 1. Push the backend to a GitHub repo

The `backend/` folder is inside `KunalGupta25/plexi-materials`. Vercel can deploy a subdirectory directly.

### 2. Import the project in Vercel

1. Go to [vercel.com/new](https://vercel.com/new) → **Import Git Repository**
2. Select `KunalGupta25/plexi-materials`
3. On the *Configure Project* screen, expand **Root Directory** and set it to **`backend`**
4. **Framework Preset:** Other
5. **Build Command:** leave blank (or `echo done`)
6. **Output Directory:** leave blank
7. Click **Deploy**

### 3. Add environment variables

In Vercel → Project → **Settings → Environment Variables**, add:

| Variable | Value |
|---|---|
| `GITHUB_CLIENT_ID` | Your OAuth App client ID |
| `GITHUB_CLIENT_SECRET` | Your OAuth App client secret |
| `GITHUB_TOKEN` | Fine-grained PAT (Contents + Issues: Write) |
| `GITHUB_REPO` | `KunalGupta25/plexi-materials` |
| `FRONTEND_URL` | `https://plexi-material.mexus.tech` |
| `JWT_SECRET` | Random 32-char hex (see below) |
| `NODE_ENV` | `production` |

Generate `JWT_SECRET`:
```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

### 4. Set the OAuth callback URL

Your Vercel backend URL will be something like `https://plexi-upload-api.vercel.app`.

Update the GitHub OAuth App callback to:
```
https://plexi-upload-api.vercel.app/api/auth/callback
```

### 5. Update the frontend API base URL

In [`docs/index.html`](../docs/index.html), update:
```javascript
return 'https://plexi-upload-api.vercel.app'; // ← your Vercel URL
```

Then push to trigger a GitHub Pages redeploy.

### 6. (Optional) Add a custom domain

In Vercel → Project → **Settings → Domains**, add `api.plexi-material.mexus.tech` and point a CNAME at `cname.vercel-dns.com`. Then update `API_BASE` and `FRONTEND_URL` to match.

---

## Other deployment options

| Platform | Notes |
|---|---|
| **Railway** | Connect repo, set root to `backend/`, add env vars, deploy |
| **Render** | Free web service, same setup as Railway |
| **Fly.io** | `fly launch` inside `backend/`, `fly secrets set KEY=value` |
| **VPS (PM2)** | `npm install -g pm2 && pm2 start server.js` + Nginx reverse proxy |

For VPS/Railway/Render, set `NODE_ENV=production` and use a process manager that keeps the server alive.

## Security Notes

- User GitHub tokens are **never stored** — only a minimal session (login, name, avatar URL) is kept server-side.
- The repo PAT (`GITHUB_TOKEN`) never leaves the server.
- Sessions expire after 24 hours.
- CORS is restricted to `FRONTEND_URL` only.
