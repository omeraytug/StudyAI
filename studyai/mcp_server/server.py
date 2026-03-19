"""
StudyAI MCP-server (FastMCP), mönster från nackademin-mcp-demo.

Miljö:
  STUDYAI_PROJECT_ROOT — repo-root (annars sätts cwd från paketplats + lecture_notes).
  STUDYAI_MCP_TRANSPORT — stdio (default) eller http
  STUDYAI_MCP_HTTP_HOST, STUDYAI_MCP_HTTP_PORT — för http (default 0.0.0.0:8003)
"""

from __future__ import annotations

import asyncio
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Annotated

from pydantic import Field

from fastmcp import FastMCP

from studyai.lecture_resolve import list_lecture_files, resolve_lecture_path
from studyai.paths import exam_dir, lecture_notes_dir, project_root
from studyai.mcp_server.logging_config import configure_logging
from studyai.mcp_server.middleware import RequestLoggingMiddleware
from studyai.summarize_agent import _norm_basename, summarize_agent_run
from studyai.tenta_rag import _run_tenta_once

_MAX_READ_CHARS = 250_000


def _repo_root() -> Path:
    env = os.getenv("STUDYAI_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # studyai/mcp_server/server.py -> parents[2] = StudyAI repo
    return Path(__file__).resolve().parents[2]


def _ensure_cwd_repo() -> Path:
    root = _repo_root()
    if not (root / "lecture_notes").is_dir():
        raise FileNotFoundError(
            f"STUDYAI_PROJECT_ROOT / repo saknar lecture_notes/: {root}"
        )
    os.chdir(root)
    return root


def _exam_read_path(filename: str) -> Path:
    """Endast basnamn under exam/."""
    exam = exam_dir().resolve()
    safe = Path(filename).name
    p = (exam / safe).resolve()
    if p.parent.resolve() != exam:
        raise ValueError(f"Ogiltig sökväg (måste ligga under exam/): {filename!r}")
    if not p.is_file():
        raise FileNotFoundError(f"Finns ej: {safe}")
    return p


def _resolve_summarize_paths(rel_paths: list[str]) -> list[Path]:
    """
    Lös PDF/TXT (eller mapp) för summarize. Tillåtna träd: lecture_notes, raw_lecture_notes, exam.

    - Exakt sökväg om den finns.
    - Annars: matcha **basnamn** i ordning raw_lecture_notes → lecture_notes → exam
      (så fel som lecture_notes/foo.txt hittas i raw_lecture_notes/foo.txt om den bara finns där).
    - Bara filnamn utan mapp: samma sökordning + tolerant bindestreck (_norm_basename).
    """
    root = project_root()
    allowed_roots = [
        (root / "lecture_notes").resolve(),
        (root / "raw_lecture_notes").resolve(),
        (root / "exam").resolve(),
    ]
    search_dirs_order = [
        root / "raw_lecture_notes",
        root / "lecture_notes",
        root / "exam",
    ]

    def _under_allowed(p: Path) -> bool:
        pr = p.resolve()
        for ar in allowed_roots:
            if ar.is_dir():
                try:
                    pr.relative_to(ar)
                    return True
                except ValueError:
                    continue
        return False

    def _resolve_one(raw: str) -> Path:
        s = raw.strip().lstrip("/")
        if ".." in s or s.startswith("/"):
            raise ValueError(f"Ogiltig sökväg: {raw!r}")

        candidate = (root / s).resolve()

        # Endast filnamn (inga delimiters): sök bara i tillåtna mappar
        bare_name = "/" not in s and "\\" not in s

        if _under_allowed(candidate):
            if candidate.is_file() or candidate.is_dir():
                return candidate

        if not bare_name and not _under_allowed(candidate):
            raise ValueError(
                f"Sökvägen måste ligga under lecture_notes/, raw_lecture_notes/ eller exam/: {raw!r}"
            )

        basename = Path(s).name
        if not basename:
            raise ValueError(f"Ogiltig sökväg: {raw!r}")

        want = _norm_basename(basename)

        for d in search_dirs_order:
            if not d.is_dir():
                continue
            dr = d.resolve()
            hit = (dr / basename).resolve()
            try:
                hit.relative_to(dr)
            except ValueError:
                continue
            if hit.is_file():
                return hit

        for d in search_dirs_order:
            if not d.is_dir():
                continue
            dr = d.resolve()
            for c in sorted(dr.iterdir(), key=lambda x: x.name.lower()):
                if c.is_file() and _norm_basename(c.name) == want:
                    return c.resolve()

        hint = ""
        raw_d = root / "raw_lecture_notes"
        if raw_d.is_dir():
            names = [p.name for p in raw_d.iterdir() if p.is_file()]
            names.sort(key=str.lower)
            if names:
                preview = ", ".join(names[:10])
                if len(names) > 10:
                    preview += " …"
                hint = f" Filer i raw_lecture_notes/: {preview}"

        raise FileNotFoundError(
            f"Finns ej: {raw!r} (sökte basnamn {basename!r} i raw_lecture_notes, lecture_notes, exam).{hint}"
        )

    return [_resolve_one(raw) for raw in rel_paths]


def _summarize_run_captured(paths: list[Path], out_path: Path | None) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        try:
            text = summarize_agent_run(paths, out_path=out_path)
        except SystemExit as e:
            return f"[fel code={e.code}]\n{buf.getvalue()}"
        except Exception as e:
            return f"[exception: {e}]\n{buf.getvalue()}"
    tail = buf.getvalue()
    if tail.strip():
        return f"{tail}\n\n---\n\n{text}"
    return text


configure_logging()
mcp = FastMCP(
    "StudyAI",
    instructions=(
        "StudyAI: läs/lista föreläsningar (lecture_notes), rå-PDF/TXT (raw_lecture_notes), "
        "tentor (exam); sammanfatta en eller flera PDF/TXT (ofta raw_lecture_notes — lista först); "
        "generera tentafrågor (RAG) från lecture_notes/*.txt. Kräver OPENAI_API_KEY för summarize och tenta."
    ),
)
mcp.add_middleware(RequestLoggingMiddleware())


@mcp.tool()
def studyai_get_project_root() -> str:
    """Returnerar absolut sökväg till StudyAI-repot och bekräftar att lecture_notes finns."""
    root = _ensure_cwd_repo()
    return str(root)


@mcp.tool()
def studyai_list_lecture_notes() -> str:
    """Listar .txt-filer i lecture_notes/."""
    _ensure_cwd_repo()
    notes = lecture_notes_dir()
    files = list_lecture_files(notes)
    if not files:
        return f"(tomt) {notes}"
    return "\n".join(p.name for p in files)


@mcp.tool()
def studyai_read_lecture_note(
    lecture: Annotated[
        str,
        Field(description="t.ex. 1, lecture_1 eller lecture_1.txt"),
    ],
) -> str:
    """Läser en föreläsningstext från lecture_notes/ (begränsad längd)."""
    _ensure_cwd_repo()
    notes = lecture_notes_dir()
    path = resolve_lecture_path(notes, lecture)
    if path is None:
        return f"Hittade ingen fil för: {lecture!r}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > _MAX_READ_CHARS:
        return text[:_MAX_READ_CHARS] + f"\n\n[… avkortat, {len(text)} tecken totalt]"
    return text


@mcp.tool()
def studyai_list_raw_lecture_notes() -> str:
    """Listar .pdf och .txt i raw_lecture_notes/ om mappen finns."""
    _ensure_cwd_repo()
    raw = project_root() / "raw_lecture_notes"
    if not raw.is_dir():
        return f"(saknas) {raw}"
    parts = sorted(raw.glob("*.txt")) + sorted(raw.glob("*.pdf"))
    if not parts:
        return f"(tomt) {raw}"
    return "\n".join(p.name for p in parts)


@mcp.tool()
def studyai_list_exam_files() -> str:
    """Listar filer i exam/."""
    _ensure_cwd_repo()
    exam = exam_dir()
    files = sorted(exam.iterdir()) if exam.is_dir() else []
    files = [p for p in files if p.is_file()]
    if not files:
        return f"(tomt) {exam}"
    return "\n".join(p.name for p in files)


@mcp.tool()
def studyai_read_exam_file(
    filename: Annotated[str, Field(description="Basnamn, t.ex. lecture_1_tentafrågor.md")],
) -> str:
    """Läser en fil under exam/ (endast basnamn)."""
    _ensure_cwd_repo()
    path = _exam_read_path(filename)
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > _MAX_READ_CHARS:
        return text[:_MAX_READ_CHARS] + f"\n\n[… avkortat, {len(text)} tecken totalt]"
    return text


@mcp.tool()
def studyai_summarize_sources(
    relative_paths: Annotated[
        list[str],
        Field(
            description=(
                "En eller flera källor: raw_lecture_notes/… och/eller lecture_notes/… (PDF/TXT). "
                "Lista gärna studyai_list_raw_lecture_notes först. Ex: raw_lecture_notes/Verktygsanvändning (1).pdf. "
                "Basnamn utan mapp matchas automatiskt (raw först, sedan lecture_notes, sedan exam)."
            ),
        ),
    ],
    output_relative: Annotated[
        str | None,
        Field(
            description=(
                "Valfritt: spara sammanfattning till denna sökväg relativt repo (t.ex. lecture_notes/summary_1.txt)"
            ),
        ),
    ] = None,
) -> str:
    """
    Sammanfattar en eller flera PDF/TXT via summarize-agenten (OpenAI).
    """
    _ensure_cwd_repo()
    paths = _resolve_summarize_paths(relative_paths)
    out: Path | None = None
    if output_relative and output_relative.strip():
        root = project_root()
        out = (root / output_relative.strip().lstrip("/")).resolve()
        allowed = [
            (root / "lecture_notes").resolve(),
            (root / "exam").resolve(),
        ]
        ok = False
        for ar in allowed:
            if ar.is_dir():
                try:
                    out.relative_to(ar)
                    ok = True
                    break
                except ValueError:
                    continue
        if not ok:
            return "output_relative måste ligga under lecture_notes/ eller exam/."
    return _summarize_run_captured(paths, out_path=out)


@mcp.tool()
def studyai_generate_tenta_questions(
    lecture: Annotated[str, Field(description="Föreläsning: 1, lecture_1, lecture_1.txt")],
    prompt: Annotated[
        str,
        Field(description="Vad agenten ska göra, t.ex. 'Skapa 5 tentafrågor med facit på svenska'"),
    ],
    num_questions: Annotated[
        int | None,
        Field(description="Ungefärligt antal frågor (tips till agenten)"),
    ] = None,
) -> str:
    """
    Kör Tenta-RAG en gång: indexerar vald lecture_notes/*.txt, söker i FAISS, sparar under exam/.
    Kräver OPENAI_API_KEY. Sparad fil: först alla frågor, sedan facit (samma numrering), inte facit under varje fråga.
    """
    _ensure_cwd_repo()
    notes = lecture_notes_dir()
    path = resolve_lecture_path(notes, lecture)
    if path is None:
        return f"Hittade ingen föreläsning för: {lecture!r}"
    result = _run_tenta_once(
        path,
        prompt.strip() or "Skapa tentafrågor med facit på svenska.",
        num_questions=num_questions,
        stream=False,
        silent=True,
    )
    if not result.get("ok"):
        return f"Fel: {result.get('error', 'okänt')}\nAssistent (delvis): {result.get('assistant_text', '')}"
    saved = result.get("saved_relative") or "(ingen fil sparad — se assistant_text)"
    return f"Sparat: {saved}\n\nAssistent: {result.get('assistant_text', '')}"


def main() -> None:
    _ensure_cwd_repo()
    transport = os.getenv("STUDYAI_MCP_TRANSPORT", "stdio").strip().lower()
    if transport in ("http", "sse"):
        host = os.getenv("STUDYAI_MCP_HTTP_HOST", "0.0.0.0")
        port = int(os.getenv("STUDYAI_MCP_HTTP_PORT", "8003"))
        asyncio.run(
            mcp.run_http_async(
                host=host,
                port=port,
                log_level="warning",
            )
        )
    else:
        asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
