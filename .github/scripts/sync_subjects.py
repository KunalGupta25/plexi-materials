"""Regenerate .github/ISSUE_TEMPLATE/upload-material.yml from subjects.json.

Run this script whenever you add, rename, or remove a subject:

    python .github/scripts/sync_subjects.py

The script reads backend/public/subjects.json (the single source of truth)
and overwrites the GitHub Issue form template so both the web portal and the
manual-issue path always show the same subjects.
"""

import json
import re
from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parents[2]
SUBJECTS_FILE   = REPO_ROOT / "backend" / "public" / "subjects.json"
TEMPLATE_FILE   = REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "upload-material.yml"

MATERIAL_TYPES = [
    "PDFs",
    "PPTs",
    "Notes",
    "Question Papers",
    "Question banks",
    "Assignments",
    "Syllabus",
    "Other",
]


def sem_label(semester: str) -> str:
    """Convert 'Semester 5' → 'Sem 5'."""
    return re.sub(r"Semester\s+(\d+)", r"Sem \1", semester)


def build_template(subjects: dict) -> str:
    """Build the full YAML content for the upload-material issue form."""

    # ── Semester options ───────────────────────────────────────────────────
    sem_options = "\n".join(
        f"        - {sem}" for sem in subjects
    )

    # ── Subject options ────────────────────────────────────────────────────
    subject_options_lines = []
    for sem, names in subjects.items():
        label = sem_label(sem)
        for name in names:
            subject_options_lines.append(f'        - "[{label}] {name}"')
    subject_options = "\n".join(subject_options_lines)

    # ── Material type options ──────────────────────────────────────────────
    type_options = "\n".join(
        f"        - {t}" for t in MATERIAL_TYPES
    )

    return f"""\
name: \U0001f4e4 Upload Study Material
description: Submit a study material file to be added to the repository.
title: "[Upload] "
labels: ["upload-material"]
body:
  - type: dropdown
    id: semester
    attributes:
      label: Semester
      description: Which semester does this material belong to?
      options:
{sem_options}
    validations:
      required: true

  - type: dropdown
    id: subject
    attributes:
      label: Subject
      description: "The subject name"
      options:
{subject_options}
    validations:
      required: true

  - type: dropdown
    id: file_type
    attributes:
      label: Material Type
      description: What type of material is this?
      options:
{type_options}
    validations:
      required: true

  - type: textarea
    id: file_upload
    attributes:
      label: File
      description: |
        Drag and drop (or paste) your file here. GitHub supports up to 25 MB per attachment.
        Accepted formats: (Topic name or Chapter number) PDF, PPT, PPTX, DOCX.
      placeholder: "Drag and drop your file here..."
    validations:
      required: true

  - type: textarea
    id: notes
    attributes:
      label: Additional Notes (optional)
      description: Any extra info about this material (chapter number, topic, etc.)
      placeholder: "Chapter 3 — Linked Lists"
    validations:
      required: false
"""


def main():
    print(f"Reading subjects from: {SUBJECTS_FILE}")
    with open(SUBJECTS_FILE, encoding="utf-8") as f:
        subjects = json.load(f)

    content = build_template(subjects)

    print(f"Writing template to:   {TEMPLATE_FILE}")
    with open(TEMPLATE_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    total = sum(len(v) for v in subjects.values())
    print(f"Done — {len(subjects)} semesters, {total} subjects total.")


if __name__ == "__main__":
    main()
