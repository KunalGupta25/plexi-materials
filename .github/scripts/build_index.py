"""Build the LlamaIndex vector index from study materials in the manifest.

Run after process_upload.py to rebuild the index with all current materials.
The index files are committed to the repo so the app can fetch them directly.
"""

import io
import json
import mimetypes
import os
import urllib.request

import PyPDF2
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MANIFEST_PATH = "manifest.json"
INDEX_DIR = "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_pdf_content_safe(pdf_bytes):
    """Extract text from PDF bytes."""
    text = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    filtered = page_text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
                    text.append(filtered)
            except Exception as e:
                print(f"  Page extraction warning: {e}")
    except Exception as e:
        print(f"  PDF extraction error: {e}")
    return "\n".join(text)


def download_file(url, max_retries=3):
    """Download a file from URL with retry logic."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as e:
            print(f"  Download error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
    return None


def get_mime_type(filename):
    """Guess MIME type from filename."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def main():
    if not os.path.exists(MANIFEST_PATH):
        print("No manifest.json found. Nothing to index.")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    if not manifest:
        print("Manifest is empty. Nothing to index.")
        return

    # Collect all files from manifest
    documents = []
    for semester, subjects in manifest.items():
        for subject, types in subjects.items():
            for file_type, file_list in types.items():
                for file_entry in file_list:
                    name = file_entry["name"]
                    url = file_entry["download_url"]
                    mime = get_mime_type(name)

                    if not (mime.startswith("text/") or mime == "application/pdf"):
                        print(f"Skipping unsupported: {name} ({mime})")
                        continue

                    print(f"Processing: {name}")
                    try:
                        content = download_file(url)
                        if not content:
                            continue

                        metadata = {
                            "filename": name,
                            "semester": semester,
                            "subject": subject,
                            "type": file_type,
                        }

                        if mime.startswith("text/"):
                            text = content.decode("utf-8", errors="ignore")
                            if text.strip():
                                documents.append(Document(text=text, metadata=metadata))

                        elif mime == "application/pdf":
                            text = read_pdf_content_safe(content)
                            if text.strip():
                                documents.append(Document(text=text, metadata=metadata))
                            else:
                                print(f"  No text extracted from: {name}")

                    except Exception as e:
                        print(f"  Error processing {name}: {e}")

    if not documents:
        print("No documents to index.")
        return

    print(f"\nBuilding index from {len(documents)} documents...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"Index persisted to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
