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
  if (!code) return res.redirect(`${FRONTEND_URL}/?auth=error&reason=no_code`);

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
    res.redirect(`${FRONTEND_URL}/?auth=success`);

  } catch (err) {
    console.error('[OAuth] Error:', err.message);
    res.redirect(`${FRONTEND_URL}/?auth=error`);
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
app.post('/api/upload/file', upload.single('file'), async (req, res) => {
  const user = getUser(req);
  if (!user) return res.status(401).json({ error: 'You must be signed in to upload.' });

  const file = req.file;
  if (!file) return res.status(400).json({ error: 'No file provided.' });

  try {
    // 1. Ensure staging release exists
    const stagingRelease = await ensureRelease(
      'staging-uploads',
      'Temporary staging area for pending uploads awaiting maintainer approval.',
    );

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
    res.json({ ok: true, originalName: file.originalname, downloadUrl });

  } catch (err) {
    console.error('[Upload File] Error:', err.message);
    res.status(500).json({ error: err.message || 'Upload failed. Please try again.' });
  }
});

app.post('/api/upload/submit', async (req, res) => {
  const user = getUser(req);
  if (!user) return res.status(401).json({ error: 'You must be signed in to submit.' });

  const { semester, subject, fileType, notes, uploadedFiles } = req.body;

  if (!semester || !subject || !fileType || !uploadedFiles?.length) {
    return res.status(400).json({ error: 'Missing required fields or uploaded files.' });
  }

  try {
    // 3. Build issue body (matches upload-material.yml template format)
    const cleanSubject = subject.replace(/^\[Sem \d+\]\s*/, '');
    let body = `### Semester\n\n${semester}\n\n`;
    body    += `### Subject\n\n${subject}\n\n`;
    body    += `### Material Type\n\n${fileType}\n\n`;
    body    += `### File\n\n${uploadedFiles.map(f => f.downloadUrl).join('\n')}\n\n`;
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

    console.log(`[Upload Submit] Issue #${issue.number} by @${user.login}`);
    res.json({ ok: true, issueNumber: issue.number, issueUrl: issue.html_url });

  } catch (err) {
    console.error('[Upload Submit] Error:', err.message);
    res.status(500).json({ error: err.message || 'Submit failed. Please try again.' });
  }
});

// ── Manage Routes (owner-only) ──────────────────────────────────────────────────
// These operate on the LIVE manifest.json (published materials) and the semester
// releases on GitHub — NOT the staging area.

function requireOwner(req, res) {
  const user = getUser(req);
  if (!user) { res.status(401).json({ error: 'Authentication required.' }); return null; }
  if (user.login !== OWNER_LOGIN) { res.status(403).json({ error: 'Forbidden.' }); return null; }
  return user;
}

// ── Manifest helpers ────────────────────────────────────────────────────────────
// manifest.json lives in the repo root. We read/write it via GitHub Contents API
// so the change is committed and the main Plexi site picks it up automatically.

async function readManifest() {
  const data = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/contents/manifest.json`);
  const content = Buffer.from(data.content, 'base64').toString('utf-8');
  return { manifest: JSON.parse(content), sha: data.sha };
}

async function writeManifest(manifest, sha, message) {
  const content = Buffer.from(JSON.stringify(manifest, null, 2) + '\n').toString('base64');
  await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/contents/manifest.json`, {
    method: 'PUT',
    body: JSON.stringify({ message, content, sha }),
  });
}

// Given a download_url, find the release asset ID so we can delete/rename it.
async function findAssetByUrl(downloadUrl) {
  // URL format: https://github.com/OWNER/REPO/releases/download/TAG/FILENAME
  const parts = downloadUrl.split('/');
  const tag      = parts[parts.length - 2];
  const filename = decodeURIComponent(parts[parts.length - 1]);
  try {
    const release = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/tags/${tag}`);
    const assets  = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/${release.id}/assets?per_page=100`);
    const asset   = assets.find(a => a.name === filename);
    return asset ? { asset, release, tag } : null;
  } catch {
    return null;
  }
}

/** GET /api/manage/materials — Return the full manifest. */
app.get('/api/manage/materials', async (req, res) => {
  if (!requireOwner(req, res)) return;
  try {
    const { manifest } = await readManifest();
    res.json(manifest);
  } catch (err) {
    console.error('[Manage] read manifest error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/** PATCH /api/manage/material/rename — Rename display name only (manifest only). */
app.patch('/api/manage/material/rename', async (req, res) => {
  if (!requireOwner(req, res)) return;
  const { semester, subject, type, oldName, newName } = req.body;
  if (!semester || !subject || !type || !oldName || !newName) {
    return res.status(400).json({ error: 'semester, subject, type, oldName, and newName are required.' });
  }
  try {
    const { manifest, sha } = await readManifest();
    const files = manifest?.[semester]?.[subject]?.[type];
    if (!files) return res.status(404).json({ error: 'Section not found in manifest.' });

    const entry = files.find(f => f.name === oldName);
    if (!entry) return res.status(404).json({ error: `File "${oldName}" not found.` });

    entry.name = newName.trim();
    await writeManifest(manifest, sha, `rename: ${oldName} → ${newName} in ${semester}/${subject}/${type}`);
    res.json({ ok: true, name: entry.name });
  } catch (err) {
    console.error('[Manage] rename error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/** DELETE /api/manage/material — Delete file (manifest entry + release asset). */
app.delete('/api/manage/material', async (req, res) => {
  if (!requireOwner(req, res)) return;
  const { semester, subject, type, name } = req.body;
  if (!semester || !subject || !type || !name) {
    return res.status(400).json({ error: 'semester, subject, type, and name are required.' });
  }
  try {
    const { manifest, sha } = await readManifest();
    const files = manifest?.[semester]?.[subject]?.[type];
    if (!files) return res.status(404).json({ error: 'Section not found.' });

    const idx = files.findIndex(f => f.name === name);
    if (idx === -1) return res.status(404).json({ error: `File "${name}" not found.` });

    const entry = files[idx];

    // Delete the release asset (best-effort — don't fail if asset already gone)
    try {
      const found = await findAssetByUrl(entry.download_url);
      if (found) {
        await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/assets/${found.asset.id}`, { method: 'DELETE' });
      }
    } catch (assetErr) {
      console.warn('[Manage] could not delete release asset:', assetErr.message);
    }

    // Remove from manifest
    files.splice(idx, 1);
    // Clean up empty containers
    if (files.length === 0) delete manifest[semester][subject][type];
    if (manifest[semester][subject] && Object.keys(manifest[semester][subject]).length === 0) delete manifest[semester][subject];
    if (manifest[semester] && Object.keys(manifest[semester]).length === 0) delete manifest[semester];

    await writeManifest(manifest, sha, `delete: ${name} from ${semester}/${subject}/${type}`);
    res.json({ ok: true });
  } catch (err) {
    console.error('[Manage] delete error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/** POST /api/manage/material/move — Move file to a different section.
 *  This is a complex operation:
 *    1. Download the file from the old release
 *    2. Upload to the target release with new asset name
 *    3. Delete the old asset
 *    4. Update manifest (remove from old, add to new)
 *    5. Commit manifest
 */
app.post('/api/manage/material/move', async (req, res) => {
  if (!requireOwner(req, res)) return;
  const { semester, subject, type, name, targetSemester, targetSubject, targetType } = req.body;
  if (!semester || !subject || !type || !name || !targetSemester || !targetSubject || !targetType) {
    return res.status(400).json({ error: 'All source and target fields are required.' });
  }

  // No-op if same location
  if (semester === targetSemester && subject === targetSubject && type === targetType) {
    return res.json({ ok: true, message: 'Source and target are the same.' });
  }

  try {
    const { manifest, sha } = await readManifest();
    const srcFiles = manifest?.[semester]?.[subject]?.[type];
    if (!srcFiles) return res.status(404).json({ error: 'Source section not found.' });

    const idx = srcFiles.findIndex(f => f.name === name);
    if (idx === -1) return res.status(404).json({ error: `File "${name}" not found in source.` });

    const entry = srcFiles[idx];
    const oldDownloadUrl = entry.download_url;

    // Parse old URL to get tag and asset filename
    const urlParts    = oldDownloadUrl.split('/');
    const oldTag      = urlParts[urlParts.length - 2];
    const oldAssetName = decodeURIComponent(urlParts[urlParts.length - 1]);

    // Compute new release tag and asset name
    const newTag       = targetSemester.toLowerCase().replace(/ /g, '-');
    const newAssetName = `${sanitize(targetSubject)}_${sanitize(targetType)}_${sanitize(name)}`;
    const newDownloadUrl = `https://github.com/${GITHUB_REPO}/releases/download/${newTag}/${newAssetName}`;

    // 1. Download the file from the old asset
    const found = await findAssetByUrl(oldDownloadUrl);
    if (!found) {
      return res.status(404).json({ error: 'Could not find the release asset to move.' });
    }

    // Download using the asset's API URL (not browser URL)
    const downloadRes = await fetch(found.asset.url, {
      headers: {
        Authorization: `Bearer ${GITHUB_TOKEN}`,
        Accept:        'application/octet-stream',
      },
      redirect: 'follow',
    });
    if (!downloadRes.ok) throw new Error(`Failed to download asset: ${downloadRes.status}`);
    const fileBuffer = Buffer.from(await downloadRes.arrayBuffer());

    // 2. Ensure target release exists and upload
    const targetRelease = await ensureRelease(newTag, `Study materials for ${targetSemester}`);

    // Clobber any existing asset with the same name
    try {
      const targetAssets = await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/${targetRelease.id}/assets?per_page=100`);
      const existing = targetAssets.find(a => a.name === newAssetName);
      if (existing) {
        await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/assets/${existing.id}`, { method: 'DELETE' });
      }
    } catch { /* best-effort clobber */ }

    const uploadUrl = `https://uploads.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/${targetRelease.id}/assets?name=${encodeURIComponent(newAssetName)}`;
    const uploadRes = await fetch(uploadUrl, {
      method: 'POST',
      headers: {
        Authorization:  `Bearer ${GITHUB_TOKEN}`,
        Accept:         'application/vnd.github+json',
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(fileBuffer.length),
      },
      body: fileBuffer,
    });
    if (!uploadRes.ok) {
      const err = await uploadRes.json().catch(() => ({}));
      throw new Error(`Upload failed: ${err.message || uploadRes.statusText}`);
    }

    // 3. Delete old asset
    try {
      await gh(`/repos/${REPO_OWNER}/${REPO_NAME}/releases/assets/${found.asset.id}`, { method: 'DELETE' });
    } catch (delErr) {
      console.warn('[Manage] could not delete old asset:', delErr.message);
    }

    // 4. Update manifest: remove from source, add to target
    srcFiles.splice(idx, 1);
    if (srcFiles.length === 0) delete manifest[semester][subject][type];
    if (manifest[semester][subject] && Object.keys(manifest[semester][subject]).length === 0) delete manifest[semester][subject];
    if (manifest[semester] && Object.keys(manifest[semester]).length === 0) delete manifest[semester];

    // Ensure target path exists
    if (!manifest[targetSemester]) manifest[targetSemester] = {};
    if (!manifest[targetSemester][targetSubject]) manifest[targetSemester][targetSubject] = {};
    if (!manifest[targetSemester][targetSubject][targetType]) manifest[targetSemester][targetSubject][targetType] = [];

    manifest[targetSemester][targetSubject][targetType].push({
      name:         entry.name,
      download_url: newDownloadUrl,
    });

    // 5. Commit
    await writeManifest(manifest, sha,
      `move: ${name} from ${semester}/${subject}/${type} → ${targetSemester}/${targetSubject}/${targetType}`);

    res.json({ ok: true, newDownloadUrl });
  } catch (err) {
    console.error('[Manage] move error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Health ─────────────────────────────────────────────────────────────────────
app.get('/api/health', (_req, res) => res.json({ ok: true }));

// ── Fallbacks ──────────────────────────────────────────────────────────────────
app.get('/', (_req, res) => res.json({ service: 'Plexi API', status: 'online' }));
app.use('*', (_req, res) => res.status(404).json({ error: 'API route not found' }));

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
