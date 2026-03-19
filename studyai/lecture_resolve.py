"""Resolve user input to a lecture .txt path under lecture_notes/."""

from __future__ import annotations

from pathlib import Path


def list_lecture_files(notes_dir: Path) -> list[Path]:
    if not notes_dir.is_dir():
        return []
    return sorted(notes_dir.glob("*.txt"))


def resolve_lecture_path(notes_dir: Path, user_input: str) -> Path | None:
    """
    Accepts e.g. "1", "lecture_1", "lecture_1.txt", "Lecture_1.TXT".
    Returns absolute path if file exists, else None.
    """
    raw = user_input.strip()
    if not raw:
        return None

    base = notes_dir
    candidates: list[Path] = []

    lower = raw.lower()
    if lower.endswith(".txt"):
        candidates.append(base / raw)
    elif raw.isdigit():
        candidates.append(base / f"lecture_{raw}.txt")
    else:
        # strip optional .txt
        stem = lower.removesuffix(".txt")
        candidates.append(base / f"{stem}.txt")
        candidates.append(base / f"lecture_{stem}.txt")

    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None
