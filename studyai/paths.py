"""Project root and standard folders (lecture_notes, exam)."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """
    Repo root: directory that contains `lecture_notes/`.

    - If STUDYAI_PROJECT_ROOT is set, use that.
    - Else walk upward from cwd until `lecture_notes` exists.
    - This works for editable installs, site-packages installs, and subfolders.
    """
    env = os.getenv("STUDYAI_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "lecture_notes").is_dir():
            return p

    raise FileNotFoundError(
        "Hittade ingen mapp 'lecture_notes/'. "
        "Kör kommandot från StudyAI-projektmappen (eller en undermapp) "
        "eller sätt miljövariabeln STUDYAI_PROJECT_ROOT."
    )


def lecture_notes_dir() -> Path:
    return project_root() / "lecture_notes"


def exam_dir() -> Path:
    d = project_root() / "exam"
    d.mkdir(parents=True, exist_ok=True)
    return d
