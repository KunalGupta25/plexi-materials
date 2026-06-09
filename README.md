# Plexi Materials

Welcome to the **Plexi Materials** repository — the central storage and index for all study materials on the Plexi platform.

## 📂 Key Files

| File | Purpose |
|---|---|
| `manifest.json` | Master index: semester → subject → type → file list |
| `indexed_files.json` | Tracks files already processed into the RAG vector store |
| `docs/index.html` | **Plexi Upload Portal** — the GitHub OAuth web UI |
| `backend/` | Node.js server powering the upload portal |

## 🛠 Adding New Materials

### Option A — Web Upload Portal *(recommended)*

1. Visit **[plexi-material.mexus.tech](https://plexi-material.mexus.tech)**
2. Sign in with GitHub (read-only access — no repo permissions required)
3. Fill in semester, subject, type, and attach your files
4. Submit — an issue is created automatically and a maintainer will approve it

### Option B — Direct GitHub Issue *(legacy)*

1. [Open a new issue](https://github.com/KunalGupta25/plexi-materials/issues/new?template=upload-material.yml) using the **Upload Study Material** template
2. Fill in the form fields and drag-and-drop your file into the *File* field
3. Submit — a maintainer will add the `approved` label to trigger processing

Both paths go through the same approval workflow. Once a maintainer adds the `approved` label, the file is published automatically.

## ⚙️ How the Pipeline Works

```
Upload Portal  ──┐
                 ├─→ GitHub Issue (label: upload-material)
GitHub Issue  ───┘         │
                     Maintainer adds "approved" label
                           │
                    add-material.yml fires
                           │
                    process_upload.py:
                      • Downloads file
                      • Uploads to semester Release
                      • Updates manifest.json
                           │
                    build_index.py:
                      • Rebuilds LlamaIndex vector store
                      • Commits updated index
```

## 🚀 Running the Upload Backend locally

```bash
cd backend
cp .env.example .env   # fill in your credentials
npm install
npm run dev
```

See [`backend/README.md`](backend/README.md) for full setup and deployment instructions.
