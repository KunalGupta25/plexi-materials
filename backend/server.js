/**
 * Plexi Upload Backend
 *
 * Vercel-compatible entry point. Uses JWT httpOnly cookies instead of
 * server-side sessions so it works correctly in a stateless serverless
 * environment.
 *
 * When run directly (local dev): starts an HTTP server.
 * When imported by Vercel:       exports the Express app as a handler.
 */

'use strict';

require('dotenv').config();

const path         = require('path');
const express      = require('express');
const cookieParser = require('cookie-parser');
const multer       = require('multer');
const cors         = require('cors');
const jwt          = require('jsonwebtoken');

// ── Config ─────────────────────────────────────────────────────────────────────
const {
  GITHUB_CLIENT_ID,
  GITHUB_CLIENT_SECRET,
  GITHUB_TOKEN,                                     // PAT: Contents Write + Issues Write
  GITHUB_REPO     = 'KunalGupta25/plexi-materials',
  FRONTEND_URL    = 'http://localhost:5500',
  JWT_SECRET      = 'dev-secret-change-in-production',
  NODE_ENV        = 'development',
  PORT            = '3001',
} = process.env;

// The only GitHub login that may access manage endpoints
const OWNER_LOGIN = 'KunalGupta25';

const [REPO_OWNER, REPO_NAME] = GITHUB_REPO.split('/');

// ── App ────────────────────────────────────────────────────────────────────────
const app = express();

// Required when running behind Vercel's / Nginx's reverse proxy
app.set('trust proxy', 1);

app.use(cors({ origin: FRONTEND_URL, credentials: true }));
app.use(express.json());
app.use(cookieParser());

// Serve the bundled frontend from public/
// express.static handles GET / → public/index.html automatically.
app.use(express.static(path.join(__dirname, 'public')));

// 25 MB per file, up to 10 files per request
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024, files: 10 },
});

// ── JWT cookie auth ────────────────────────────────────────────────────────────
const COOKIE_NAME = 'plexi_auth';
const IS_PROD     = NODE_ENV === 'production';

const COOKIE_OPTS = {
  httpOnly: true,
  secure:   IS_PROD,
  sameSite: 'lax',              // same-origin deployment — 'lax' is safe in all envs
  maxAge:   24 * 60 * 60 * 1000, // 1 day
  path:     '/',
};

/** Sign a JWT containing the user's public profile (no secrets). */
function signToken(user) {
  return jwt.sign(
    { login: user.login, name: user.name, avatar_url: user.avatar_url },
    JWT_SECRET,
    { expiresIn: '1d' },
  );
}

/** Verify the auth cookie and return the decoded user, or null. */
function getUser(req) {
  try {
    const token = req.cookies?.[COOKIE_NAME];
    if (!token) return null;
    return jwt.verify(token, JWT_SECRET);
  } catch {
    return null;
  }
}

// ── GitHub API helper ──────────────────────────────────────────────────────────
async function gh(path, opts = {}) {
  const url = path.startsWith('http') ? path : `https://api.github.com${path}`;
  const res = await fetch(url, {
    ...opts,
    headers: {
      Authorization:          `Bearer ${GITHUB_TOKEN}`,
      Accept:                 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      ...(opts.body && !opts.headers?.['Content-Type']
        ? { 'Content-Type': 'application/json' }
        : {}),
      ...opts.headers,
    },
  });
  if (opts.method === 'DELETE') return null;
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message || `GitHub API ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

function sanitize(name) {
  return name.replace(/[^\w.\-]/g, '_');
}

// Release cache (lives for the duration of a warm serverless instance)
const releaseCache = new Map();

async function ensureRelease(tag, description = '') {
  if (releaseCache.has(tag)) return releaseCache.get(tag);
  try {
    const r = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/tags/${tag}`);
    releaseCache.set(tag, r);
    return r;
  } catch {
    const r = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases`, {
      method: 'POST',
      body:   JSON.stringify({ tag_name: tag, name: tag, body: description }),
    });
    releaseCache.set(tag, r);
    return r;
  }
}

// ── Auth Routes ────────────────────────────────────────────────────────────────

/** Step 1 — Start GitHub OAuth. */
app.get('/api/auth/github', (req, res) => {
  if (!GITHUB_CLIENT_ID) {
    return res.status(500).json({ error: 'GITHUB_CLIENT_ID is not configured.' });
  }
  // Build the callback URL from the incoming request so it works on any domain
  const callback = `${req.protocol}://${req.get('host')}/api/auth/callback`;
  const params = new URLSearchParams({
    client_id:    GITHUB_CLIENT_ID,
    redirect_uri: callback,
    scope:        'read:user',
  });
  res.redirect(`https://github.com/login/oauth/authorize?${params}`);
});

/** Step 2 — GitHub redirects here with ?code. */
app.get('/api/auth/callback', async (req, res) => {
  const { code } = req.query;
  if (!code) return res.redirect('/?auth=error&reason=no_code');

  try {
    // Exchange code for access token
    const tokenRes = await fetch('https://github.com/login/oauth/access_token', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body:    JSON.stringify({ client_id: GITHUB_CLIENT_ID, client_secret: GITHUB_CLIENT_SECRET, code }),
    });
    const { access_token, error, error_description } = await tokenRes.json();
    if (error) throw new Error(error_description || error);

    // Fetch user profile — we only store public info, never the token
    const userRes = await fetch('https://api.github.com/user', {
      headers: { Authorization: `Bearer ${access_token}`, Accept: 'application/vnd.github+json' },
    });
    const user = await userRes.json();

    // Issue a JWT and set it as an httpOnly cookie
    const token = signToken({ login: user.login, name: user.name || user.login, avatar_url: user.avatar_url });
    res.cookie(COOKIE_NAME, token, COOKIE_OPTS);
    res.redirect('/?auth=success');

  } catch (err) {
    console.error('[OAuth] Error:', err.message);
    res.redirect('/?auth=error');
  }
});

/** Return the currently signed-in user (decoded from JWT). */
app.get('/api/me', (req, res) => {
  const user = getUser(req);
  if (!user) return res.status(401).json({ error: 'Unauthenticated' });
  res.json({ login: user.login, name: user.name, avatar_url: user.avatar_url });
});

/** Clear the auth cookie (sign out). */
app.post('/api/auth/logout', (req, res) => {
  res.clearCookie(COOKIE_NAME, { path: '/' });
  res.json({ ok: true });
});

// ── Upload Route ───────────────────────────────────────────────────────────────
app.post('/api/upload', upload.array('files', 10), async (req, res) => {
  const user = getUser(req);
  if (!user) return res.status(401).json({ error: 'You must be signed in to upload.' });

  const { semester, subject, fileType, notes } = req.body;
  const files = req.files;

  if (!semester || !subject || !fileType) {
    return res.status(400).json({ error: 'Semester, subject, and material type are required.' });
  }
  if (!files?.length) {
    return res.status(400).json({ error: 'At least one file is required.' });
  }

  try {
    // 1. Ensure staging release exists
    const stagingRelease = await ensureRelease(
      'staging-uploads',
      'Temporary staging area for pending uploads awaiting maintainer approval.',
    );

    // 2. Upload files to the staging release
    const uploaded = [];
    for (const file of files) {
      const safeName = sanitize(file.originalname);

      // Clobber any existing asset with the same name
      try {
        const assets = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/${stagingRelease.id}/assets`);
        const existing = assets.find(a => a.name === safeName);
        if (existing) {
          await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/assets/${existing.id}`, { method: 'DELETE' });
        }
      } catch { /* best-effort clobber */ }

      const uploadUrl = `https://uploads.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/${stagingRelease.id}/assets?name=${encodeURIComponent(safeName)}`;
      const assetRes = await fetch(uploadUrl, {
        method:  'POST',
        headers: {
          Authorization:    `Bearer ${GITHUB_TOKEN}`,
          Accept:           'application/vnd.github+json',
          'Content-Type':   file.mimetype || 'application/octet-stream',
          'Content-Length': String(file.buffer.length),
        },
        body: file.buffer,
      });

      if (!assetRes.ok) {
        const err = await assetRes.json().catch(() => ({}));
        throw new Error(`Failed to upload "${file.originalname}": ${err.message || assetRes.statusText}`);
      }

      const downloadUrl = `https://github.com/${GITHUB_REPO}/releases/download/staging-uploads/${safeName}`;
      uploaded.push({ originalName: file.originalname, downloadUrl });
    }

    // 3. Build issue body (matches upload-material.yml template format)
    const cleanSubject = subject.replace(/^\[Sem \d+\]\s*/, '');
    let body = `### Semester\n\n${semester}\n\n`;
    body    += `### Subject\n\n${subject}\n\n`;
    body    += `### Material Type\n\n${fileType}\n\n`;
    body    += `### File\n\n${uploaded.map(f => f.downloadUrl).join('\n')}\n\n`;
    body    += `### Additional Notes (optional)\n\n${notes?.trim() || '_No response_'}\n`;
    body    += `\n---\n_Submitted by [@${user.login}](https://github.com/${user.login}) via [Plexi Upload Portal](${FRONTEND_URL})_`;

    // 4. Create the GitHub Issue
    const issue = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/issues`, {
      method: 'POST',
      body:   JSON.stringify({
        title:  `[Upload] ${cleanSubject} — ${fileType}`,
        body,
        labels: ['upload-material'],
      }),
    });

    console.log(`[Upload] Issue #${issue.number} by @${user.login}`);
    res.json({ ok: true, issueNumber: issue.number, issueUrl: issue.html_url, files: uploaded.map(f => f.originalName) });

  } catch (err) {
    console.error('[Upload] Error:', err.message);
    res.status(500).json({ error: err.message || 'Upload failed. Please try again.' });
  }
});

// ── Manage Routes (owner-only) ──────────────────────────────────────────────────

function requireOwner(req, res) {
  const user = getUser(req);
  if (!user) { res.status(401).json({ error: 'Authentication required.' }); return null; }
  if (user.login !== OWNER_LOGIN) { res.status(403).json({ error: 'Forbidden.' }); return null; }
  return user;
}

/** List all assets in the staging-uploads release. */
app.get('/api/manage/assets', async (req, res) => {
  if (!requireOwner(req, res)) return;
  try {
    const release = await ensureRelease('staging-uploads', 'Staging area for pending uploads.');
    const assets  = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/${release.id}/assets?per_page=100`);
    res.json(assets.map(a => ({
      id:           a.id,
      name:         a.name,
      size:         a.size,
      created_at:   a.created_at,
      download_url: a.browser_download_url,
    })));
  } catch (err) {
    console.error('[Manage] list assets error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/** Delete a staging release asset by ID. */
app.delete('/api/manage/asset/:id', async (req, res) => {
  if (!requireOwner(req, res)) return;
  try {
    await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/assets/${req.params.id}`, { method: 'DELETE' });
    res.json({ ok: true });
  } catch (err) {
    console.error('[Manage] delete asset error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/** Rename a staging release asset by ID (GitHub supports PATCH on asset name). */
app.patch('/api/manage/asset/:id', async (req, res) => {
  if (!requireOwner(req, res)) return;
  const { newName } = req.body;
  if (!newName || typeof newName !== 'string' || !newName.trim()) {
    return res.status(400).json({ error: 'newName is required.' });
  }
  try {
    const updated = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/assets/${req.params.id}`, {
      method: 'PATCH',
      body:   JSON.stringify({ name: sanitize(newName.trim()) }),
    });
    res.json({ ok: true, name: updated.name });
  } catch (err) {
    console.error('[Manage] rename asset error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Health ─────────────────────────────────────────────────────────────────────
app.get('/api/health', (_req, res) => res.json({ ok: true }));

// ── SPA fallback ───────────────────────────────────────────────────────────────
// Unmatched /api/* routes return a JSON 404.
// Everything else (deep links, unknown paths) serves index.html so the
// frontend can handle routing client-side.
app.use('/api', (_req, res) => res.status(404).json({ error: 'API route not found' }));
app.get('*', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

// ── Local dev server ───────────────────────────────────────────────────────────
// When run directly (npm start / npm run dev), start a real HTTP server.
// When imported by Vercel's runtime, only the exported `app` is used.
if (require.main === module) {
  const port = parseInt(PORT, 10);
  app.listen(port, () => {
    console.log(`✓ Plexi Upload Backend → http://localhost:${port}`);
    if (!GITHUB_CLIENT_ID) console.warn('⚠  GITHUB_CLIENT_ID not set');
    if (!GITHUB_TOKEN)     console.warn('⚠  GITHUB_TOKEN not set');
  });
}

module.exports = app;
