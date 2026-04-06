"""Process a GitHub issue created via the upload-material template.

Parses the issue body for semester, subject, type, and file attachment URL.
Uploads the file as a GitHub Release asset and updates manifest.json.
"""

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request


def parse_issue_body(body):
    """Extract form fields from the GitHub issue forms body."""
    fields = {}
    current_heading = None

    for line in body.splitlines():
        line = line.strip()
        if line.startswith("### "):
            current_heading = line[4:].strip()
        elif current_heading and line and line != "_No response_":
            if current_heading not in fields:
                fields[current_heading] = line
            else:
                fields[current_heading] += "\n" + line

    return fields


def extract_attachment_urls(text):
    """Extract all file attachment URLs from text."""
    pattern = r"https://github\.com/[^\s\)]+/(?:files|assets)/[^\s\)]+"
    return re.findall(pattern, text)


def gh(*args, input_data=None):
    """Run a gh CLI command and return output."""
    cmd = ["gh"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, input=input_data)
    if result.returncode != 0:
        print(f"gh command failed: {' '.join(cmd)}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def ensure_release(tag, repo):
    """Create a release if it doesn't exist, return the tag."""
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        gh(
            "release",
            "create",
            tag,
            "--repo",
            repo,
            "--title",
            tag,
            "--notes",
            f"Study materials for {tag}",
        )
    return tag


def sanitize_filename(name):
    """Sanitize a filename for use as a release asset name."""
    return re.sub(r"[^\w.\-]", "_", name)


def download_with_retry(url, dest, chunk_size=1024 * 1024):
    """Download a URL to dest, retrying cleanly on HTTP 416."""
    existing_size = os.path.getsize(dest) if os.path.exists(dest) else 0

    def _download(range_start=None):
        headers = {}
        if range_start is not None and range_start > 0:
            headers["Range"] = f"bytes={range_start}-"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            mode = "ab" if (range_start is not None and range_start > 0) else "wb"
            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)

    try:
        if existing_size > 0:
            _download(range_start=existing_size)
        else:
            _download(range_start=None)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            if os.path.exists(dest):
                os.remove(dest)
            _download(range_start=None)
        else:
            raise


def main():
    body = os.environ["ISSUE_BODY"]
    issue_number = os.environ["ISSUE_NUMBER"]
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    fields = parse_issue_body(body)

    semester = fields.get("Semester", "").strip()
    subject = fields.get("Subject", "").strip()
    # Strip semester prefix (e.g. "[Sem 3] Database Management Systems" -> "Database Management Systems")
    subject = re.sub(r"^\[Sem \d+\]\s*", "", subject)
    file_type = fields.get("Material Type", "").strip()
    file_text = fields.get("File", "")
    notes = fields.get("Additional Notes (optional)", "").strip()

    if not all([semester, subject, file_type, file_text]):
        gh(
            "issue",
            "comment",
            str(issue_number),
            "--repo",
            repo,
            "--body",
            "Could not parse all required fields from the issue. "
            "Please ensure semester, subject, material type, and file are provided.",
        )
        sys.exit(1)

    attachment_urls = extract_attachment_urls(file_text)
    if not attachment_urls:
        gh(
            "issue",
            "comment",
            str(issue_number),
            "--repo",
            repo,
            "--body",
            "No file attachment found. Please drag-and-drop a file into the File field.",
        )
        sys.exit(1)

    # Create release tag from semester (e.g., "Semester 5" -> "sem-5")
    tag = semester.lower().replace(" ", "-")
    ensure_release(tag, repo)

    # Load manifest
    manifest_path = "manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest.setdefault(semester, {})
    manifest[semester].setdefault(subject, {})
    manifest[semester][subject].setdefault(file_type, [])

    uploaded_files = []

    for attachment_url in attachment_urls:
        # Derive the original filename from the attachment URL
        original_name = attachment_url.split("/")[-1]
        # URL-decode common patterns
        original_name = urllib.request.url2pathname(original_name.split("?")[0])
        original_name = os.path.basename(original_name)

        # Upload as release asset with a prefixed name to avoid collisions
        asset_prefix = f"{sanitize_filename(subject)}_{sanitize_filename(file_type)}_"
        asset_name = asset_prefix + sanitize_filename(original_name)

        local_path = f"/tmp/{asset_name}"

        # Download the attachment
        print(f"Downloading: {attachment_url}")
        download_with_retry(attachment_url, local_path)

        print(f"Uploading {asset_name} to release {tag}...")
        gh("release", "upload", tag, local_path, "--repo", repo, "--clobber")

        # Build download URL
        download_url = f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"

        # Add to manifest if not already present
        existing_names = [e["name"] for e in manifest[semester][subject][file_type]]
        if original_name not in existing_names:
            manifest[semester][subject][file_type].append(
                {
                    "name": original_name,
                    "download_url": download_url,
                }
            )

        uploaded_files.append(original_name)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # Close issue with success comment
    file_list = "\n".join(f"  - `{name}`" for name in uploaded_files)
    comment = (
        f"**{len(uploaded_files)} file(s) uploaded successfully!**\n\n"
        f"- **Semester:** {semester}\n"
        f"- **Subject:** {subject}\n"
        f"- **Type:** {file_type}\n"
        f"- **Files:**\n{file_list}\n"
    )
    if notes:
        comment += f"- **Notes:** {notes}\n"
    comment += f"\nThe manifest has been updated and the files are now available in the Study Materials Hub."

    gh("issue", "comment", str(issue_number), "--repo", repo, "--body", comment)
    gh("issue", "close", str(issue_number), "--repo", repo)

    print("Done!")


if __name__ == "__main__":
    main()
