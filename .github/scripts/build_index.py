"""Build the LlamaIndex vector index from study materials in the manifest.

Run after process_upload.py to rebuild the index with all current materials.
The index files are committed to the repo so the app can fetch them directly.

This builder uses normal PDF text extraction first, then falls back to OCR for
scanned pages and supplements low-text pages with sparse OCR labels. The sparse
OCR pass helps recover text from flowcharts, ER diagrams, and other visual pages
where the important signal is scattered around the page.
"""

import io
import json
import mimetypes
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import PyPDF2
import pypdfium2 as pdfium
import pytesseract
from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MANIFEST_PATH = "manifest.json"
INDEX_DIR = "index"
INDEX_CACHE_PATH = "indexed_files.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_DIRECT_TEXT_CHARS = 120
OCR_RENDER_SCALE = 2
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_TIMEOUT_SECONDS = int(os.getenv("OCR_TIMEOUT_SECONDS", "90"))
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx"}


def normalize_text(text):
    """Normalize text extracted from PDFs or OCR."""
    if not text:
        return ""
    filtered = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    lines = [" ".join(line.split()) for line in filtered.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def alpha_count(text):
    """Return the count of alpha-numeric characters in text."""
    return sum(1 for char in text if char.isalnum())


def read_pdf_content_safe(pdf_bytes):
    """Extract direct text from PDF bytes via PyPDF2."""
    pages = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            try:
                page_text = normalize_text(page.extract_text())
                pages.append(page_text)
            except Exception as e:
                print(f"  Page extraction warning: {e}")
                pages.append("")
    except Exception as e:
        print(f"  PDF extraction error: {e}")
    return pages


def pil_image_from_pdf_page(page):
    """Render a PDF page to a PIL image for OCR."""
    bitmap = page.render(scale=OCR_RENDER_SCALE)
    return bitmap.to_pil()


def collect_sparse_ocr_lines(page_image):
    """Collect sparse OCR labels from visually dense pages."""
    data = pytesseract.image_to_data(
        page_image,
        lang=OCR_LANG,
        config="--oem 3 --psm 11",
        output_type=pytesseract.Output.DICT,
        timeout=OCR_TIMEOUT_SECONDS,
    )

    rows = {}
    count = len(data.get("text", []))
    for index in range(count):
        text = normalize_text(data["text"][index])
        confidence = data["conf"][index]
        if not text:
            continue
        try:
            conf_value = float(confidence)
        except (TypeError, ValueError):
            conf_value = -1
        if conf_value < 35:
            continue

        top = int(data["top"][index])
        line_key = round(top / 20)
        rows.setdefault(line_key, []).append((int(data["left"][index]), text))

    lines = []
    for _, words in sorted(rows.items()):
        line = " ".join(word for _, word in sorted(words))
        if line:
            lines.append(line)

    deduped = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped


def ocr_pdf_pages(pdf_bytes, direct_pages):
    """Run OCR on low-text pages and gather sparse labels for diagram pages."""
    combined_pages = []
    try:
        pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    except Exception as e:
        print(f"  PDF render error: {e}")
        return "\n".join(page for page in direct_pages if page)

    try:
        for page_index in range(len(pdf)):
            direct_text = (
                direct_pages[page_index] if page_index < len(direct_pages) else ""
            )
            page = pdf[page_index]
            try:
                page_image = pil_image_from_pdf_page(page)
                direct_length = alpha_count(direct_text)

                parts = []
                if direct_text:
                    parts.append(direct_text)

                if direct_length < MIN_DIRECT_TEXT_CHARS:
                    ocr_text = normalize_text(
                        pytesseract.image_to_string(
                            page_image,
                            lang=OCR_LANG,
                            config="--oem 3 --psm 6",
                            timeout=OCR_TIMEOUT_SECONDS,
                        )
                    )
                    if ocr_text:
                        parts.append("[OCR TEXT]")
                        parts.append(ocr_text)

                sparse_lines = collect_sparse_ocr_lines(page_image)
                if sparse_lines:
                    sparse_block = "\n".join(f"- {line}" for line in sparse_lines[:50])
                    parts.append("[DIAGRAM LABELS]")
                    parts.append(sparse_block)

                page_text = "\n".join(part for part in parts if part).strip()
                if page_text:
                    combined_pages.append(f"[Page {page_index + 1}]\n{page_text}")
            except Exception as e:
                print(f"  OCR warning on page {page_index + 1}: {e}")
                if direct_text:
                    combined_pages.append(f"[Page {page_index + 1}]\n{direct_text}")
    finally:
        pdf.close()

    return "\n\n".join(combined_pages).strip()


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


def get_extension(filename):
    """Return the lowercase file extension."""
    return Path(filename).suffix.lower()


def is_supported_file(filename, mime):
    """Return True when a file can be indexed."""
    extension = get_extension(filename)
    return (
        mime.startswith("text/")
        or mime == "application/pdf"
        or extension in OFFICE_EXTENSIONS
    )


def convert_office_to_pdf(file_bytes, filename):
    """Convert Office files to PDF via LibreOffice headless."""
    extension = get_extension(filename)
    if extension not in OFFICE_EXTENSIONS:
        return None

    with tempfile.TemporaryDirectory(prefix="plexi_office_") as temp_dir:
        input_path = os.path.join(temp_dir, f"source{extension}")
        with open(input_path, "wb") as file_handle:
            file_handle.write(file_bytes)

        command = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            temp_dir,
            input_path,
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=180
            )
        except Exception as e:
            print(f"  LibreOffice conversion error: {e}")
            return None

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or "unknown conversion failure"
            print(f"  LibreOffice conversion failed: {detail}")
            return None

        output_path = os.path.splitext(input_path)[0] + ".pdf"
        if not os.path.exists(output_path):
            pdf_candidates = list(Path(temp_dir).glob("*.pdf"))
            if not pdf_candidates:
                print("  LibreOffice conversion did not produce a PDF.")
                return None
            output_path = str(pdf_candidates[0])

        with open(output_path, "rb") as file_handle:
            return file_handle.read()


def main():
    if not os.path.exists(MANIFEST_PATH):
        print("No manifest.json found. Nothing to index.")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    if not manifest:
        print("Manifest is empty. Nothing to index.")
        return

    indexed_urls = []
    if os.path.exists(INDEX_CACHE_PATH):
        try:
            with open(INDEX_CACHE_PATH, "r") as f:
                indexed_urls = json.load(f)
        except Exception as e:
            print(f"Warning: could not read cache: {e}")

    # Collect all files from manifest
    documents = []
    newly_indexed_urls = []
    for semester, subjects in manifest.items():
        for subject, types in subjects.items():
            for file_type, file_list in types.items():
                for file_entry in file_list:
                    name = file_entry["name"]
                    url = file_entry["download_url"]

                    if url in indexed_urls:
                        print(f"Skipping already indexed file: {name}")
                        continue

                    mime = get_mime_type(name)

                    if not is_supported_file(name, mime):
                        print(f"Skipping unsupported: {name} ({mime})")
                        continue

                    print(f"Processing: {name}")
                    try:
                        prev_doc_count = len(documents)
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
                            direct_pages = read_pdf_content_safe(content)
                            text = ocr_pdf_pages(content, direct_pages)
                            if text.strip():
                                documents.append(Document(text=text, metadata=metadata))
                            else:
                                print(f"  No text extracted from: {name}")

                        elif get_extension(name) in OFFICE_EXTENSIONS:
                            pdf_bytes = convert_office_to_pdf(content, name)
                            if not pdf_bytes:
                                print(f"  Could not convert Office file: {name}")
                                continue

                            direct_pages = read_pdf_content_safe(pdf_bytes)
                            text = ocr_pdf_pages(pdf_bytes, direct_pages)
                            if text.strip():
                                documents.append(Document(text=text, metadata=metadata))
                            else:
                                print(f"  No text extracted after conversion: {name}")

                        if len(documents) > prev_doc_count:
                            newly_indexed_urls.append(url)

                    except Exception as e:
                        print(f"  Error processing {name}: {e}")

    if not documents:
        print("No new documents to index.")
        return

    print(f"\nBuilding/Updating index with {len(documents)} new documents...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.embed_model = embed_model

    if os.path.exists(INDEX_DIR) and len(os.listdir(INDEX_DIR)) > 0:
        print("Loading existing index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        for doc in documents:
            index.insert(doc)
    else:
        print("Creating new index...")
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"Index persisted to {INDEX_DIR}/")

    all_indexed = list(set(indexed_urls + newly_indexed_urls))
    with open(INDEX_CACHE_PATH, "w") as f:
        json.dump(all_indexed, f, indent=2)
    print(f"Cache updated with {len(newly_indexed_urls)} new URLs.")


if __name__ == "__main__":
    main()
