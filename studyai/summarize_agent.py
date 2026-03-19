"""
PDF/TXT → en sammanslagen LECTURE SUMMARY i ren text (koncept per block, RAG-vänlig).

Pipeline: chunk-extraktion → merge per fil → sista anrop med fast mall (se summary_prompts.py).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _prep_sys_path_when_run_as_script() -> None:
    """Gör `import studyai` möjligt vid `uv run summarize_agent.py` från valfri cwd."""
    repo = Path(__file__).resolve().parent.parent
    if (repo / "pyproject.toml").is_file():
        rs = str(repo)
        if rs not in sys.path:
            sys.path.insert(0, rs)


_prep_sys_path_when_run_as_script()

import argparse
import unicodedata

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from studyai.paths import project_root
from studyai.summary_prompts import (
    CHUNK_EXTRACT_SYSTEM,
    FILE_MERGE_SYSTEM,
    LECTURE_SUMMARY_SYSTEM,
)
from studyai.util.models import get_model
from studyai.util.pretty_print import Colors, print_welcome

# --- Token-/kostnadsgränser (justera via miljö om du vill) ---
import os

CHUNK_SIZE = int(os.getenv("STUDYAI_SUMMARY_CHUNK_CHARS", "4500"))
CHUNK_OVERLAP = int(os.getenv("STUDYAI_SUMMARY_CHUNK_OVERLAP", "200"))
MAX_CHUNKS_PER_FILE = int(os.getenv("STUDYAI_SUMMARY_MAX_CHUNKS", "8"))

MAX_OUT_CHUNK = int(os.getenv("STUDYAI_SUMMARY_MAX_TOKENS_CHUNK", "420"))
MAX_OUT_FILE = int(os.getenv("STUDYAI_SUMMARY_MAX_TOKENS_FILE", "900"))
MAX_OUT_FINAL = int(os.getenv("STUDYAI_SUMMARY_MAX_TOKENS_FINAL", "4096"))


def _norm_basename(name: str) -> str:
    """Jämför filnamn tolerant: Unicode-/ASCII-bindestreck + NFC-normalisering (ö vs ö)."""
    s = name
    for ch in (
        "\u2011",  # narrow no-break hyphen (vanligt i ChatGPT-exporter)
        "\u2010",
        "\u2013",
        "\u2014",
        "\u2212",
    ):
        s = s.replace(ch, "-")
    return unicodedata.normalize("NFC", s).casefold()


def _resolve_user_path(raw: Path) -> tuple[Path | None, str | None]:
    """
    Lös fil eller mapp. Om exakt sökväg saknas men mappen finns, matcha filnamn
    tolerant (Unicode-bindestreck vs ASCII-minus).
    Returnerar (path, hint) där hint är t.ex. vilken fil som matchades.
    """
    p = raw.expanduser()
    try:
        p = p.resolve()
    except OSError:
        return None, f"Ogiltig sökväg: {raw}"

    if p.is_dir():
        return p, None
    if p.is_file():
        return p, None

    parent = p.parent
    try:
        parent = parent.resolve()
    except OSError:
        return None, f"Hittades inte (mapp saknas?): {raw}"

    if not parent.is_dir():
        return None, f"Hittades inte: {raw}"

    want = _norm_basename(p.name)
    if not want:
        return None, f"Hittades inte: {raw}"

    candidates: list[Path] = [c for c in parent.iterdir() if c.is_file()]
    for c in sorted(candidates, key=lambda x: x.name.lower()):
        if _norm_basename(c.name) == want:
            hint = f"matchade {c.name!r} (du skrev {raw.name!r} — ofta olika bindestreck/tecken)"
            return c.resolve(), hint

    return None, f"Hittades inte i {parent}: {raw.name!r}"


def _expand_paths(paths: list[Path]) -> tuple[list[Path], list[str]]:
    """Returnerar (filer, felmeddelanden / tips)."""
    files: list[Path] = []
    notes: list[str] = []

    for raw in paths:
        resolved, hint = _resolve_user_path(raw)
        if resolved is None:
            notes.append(hint or str(raw))
            continue
        if hint:
            print(f"{Colors.DIM}{hint}{Colors.RESET}")

        if resolved.is_dir():
            files.extend(sorted(resolved.glob("*.pdf")))
            files.extend(sorted(resolved.glob("*.txt")))
        else:
            files.append(resolved)

    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        if f not in seen and f.suffix.lower() in (".pdf", ".txt"):
            seen.add(f)
            unique.append(f)
    return unique, notes


def _load_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(path)).load()
        return "\n\n".join((d.page_content or "").strip() for d in docs if d.page_content)
    if path.suffix.lower() == ".txt":
        docs = TextLoader(str(path), encoding="utf-8").load()
        return "\n\n".join((d.page_content or "").strip() for d in docs if d.page_content)
    raise ValueError(f"Stöds ej: {path.suffix}")


def _split_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text] if text.strip() else []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text)


def _response_text(resp: object) -> str:
    return (getattr(resp, "content", None) or str(resp)).strip()


def summarize_agent_run(
    paths: list[Path],
    *,
    out_path: Path | None = None,
) -> str:
    files, path_notes = _expand_paths(paths)
    if not files:
        print(f"{Colors.RED}Inga .pdf- eller .txt-filer hittades.{Colors.RESET}")
        for line in path_notes:
            print(f"  {Colors.YELLOW}•{Colors.RESET} {line}")
        # tips: lista mappinnehåll om någon sökväg pekade in under raw_lecture_notes
        for raw in paths:
            par = Path(raw).expanduser().parent
            if par.is_dir() and par.name == "raw_lecture_notes":
                listed = sorted(par.glob("*.txt")) + sorted(par.glob("*.pdf"))
                if listed:
                    print(f"\n{Colors.DIM}Filer i {par}:{Colors.RESET}")
                    for x in listed:
                        print(f"  {x.name}")
                break
        raise SystemExit(1)

    if path_notes:
        print(f"{Colors.YELLOW}Varning — vissa argument användes inte:{Colors.RESET}")
        for line in path_notes:
            print(f"  {Colors.YELLOW}•{Colors.RESET} {line}")

    # Steg 1–2: billiga extraktioner per chunk / per fil. Steg 3: ett strukturerat slutformat (fler tokens).
    llm_chunk = get_model(temperature=0.12, max_tokens=MAX_OUT_CHUNK)
    llm_file = get_model(temperature=0.18, max_tokens=MAX_OUT_FILE)
    llm_final = get_model(temperature=0.22, max_tokens=MAX_OUT_FINAL)

    sys_chunk = SystemMessage(content=CHUNK_EXTRACT_SYSTEM)
    sys_file = SystemMessage(content=FILE_MERGE_SYSTEM)
    sys_final = SystemMessage(content=LECTURE_SUMMARY_SYSTEM)

    per_file_blocks: list[str] = []

    for fp in files:
        print(f"{Colors.DIM}Läser: {fp.name} …{Colors.RESET}")
        try:
            full = _load_file(fp)
        except Exception as e:
            print(f"{Colors.YELLOW}Hoppar över {fp.name}: {e}{Colors.RESET}")
            continue
        if not full.strip():
            print(f"{Colors.YELLOW}Tom text: {fp.name}{Colors.RESET}")
            continue

        chunks = _split_text(full)
        if len(chunks) > MAX_CHUNKS_PER_FILE:
            print(
                f"{Colors.YELLOW}  Varning: {fp.name} har många delar — bearbetar bara "
                f"de första {MAX_CHUNKS_PER_FILE} för att spara tokens.{Colors.RESET}"
            )
            chunks = chunks[:MAX_CHUNKS_PER_FILE]

        chunk_notes: list[str] = []
        for i, ch in enumerate(chunks, start=1):
            msg = HumanMessage(
                content=(
                    f"Source file: {fp.name} (part {i}/{len(chunks)})\n\n"
                    f"--- EXCERPT ---\n{ch}\n--- END EXCERPT ---"
                ),
            )
            resp = llm_chunk.invoke([sys_chunk, msg])
            piece = _response_text(resp)
            if piece and piece.strip().upper() not in ("NONE", "NONE."):
                chunk_notes.append(piece)

        if not chunk_notes:
            continue

        if len(chunk_notes) == 1:
            merged_file = chunk_notes[0]
        else:
            joined = "\n\n".join(
                f"--- PART {j} ---\n{t}" for j, t in enumerate(chunk_notes, start=1)
            )
            resp_f = llm_file.invoke(
                [
                    sys_file,
                    HumanMessage(
                        content=f"Document: {fp.name}\n\n{joined}",
                    ),
                ]
            )
            merged_file = _response_text(resp_f)

        per_file_blocks.append(f"=== SOURCE: {fp.name} ===\n{merged_file}")

    if not per_file_blocks:
        raise SystemExit("Kunde inte skapa några sammanfattningar.")

    bundle = "\n\n".join(per_file_blocks)
    final_resp = llm_final.invoke(
        [
            sys_final,
            HumanMessage(
                content=(
                    "Produce the complete LECTURE SUMMARY from these pipeline notes. "
                    "Use the same language as the notes for all free text (e.g. Swedish).\n\n"
                    f"{bundle}"
                ),
            ),
        ],
    )
    final = _response_text(final_resp)

    # Endast slutdokumentet (ren .txt, inga extra rubriker från oss)
    full_output = final if final.endswith("\n") else final + "\n"

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(full_output, encoding="utf-8")
        print(f"{Colors.BRIGHT_GREEN}Sparat:{Colors.RESET} {out_path}")

    return full_output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PDF/TXT → en konceptbaserad LECTURE SUMMARY (plain text, RAG-vänlig). "
            "Miljö: STUDYAI_SUMMARY_MAX_TOKENS_FINAL m.m."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Filer och/eller mappar (söker *.pdf och *.txt i mappar).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Spara resultat till fil (ren text, t.ex. .txt).",
    )
    parser.add_argument(
        "--lecture-notes",
        action="store_true",
        help="Använd projektets lecture_notes/ + raw_lecture_notes/ om de finns.",
    )
    args = parser.parse_args(argv)

    print_welcome(
        title="StudyAI — LECTURE SUMMARY",
        description=(
            "Konceptbaserad plain text (ingen markdown). Flera små steg + ett avslutande format-steg."
        ),
        version="0.2.0",
    )

    roots: list[Path] = []
    if args.paths:
        roots.extend(Path(p) for p in args.paths)
    if args.lecture_notes:
        r = project_root()
        for name in ("lecture_notes", "raw_lecture_notes"):
            d = r / name
            if d.is_dir():
                roots.append(d)

    if not roots:
        print(
            f"{Colors.YELLOW}Ange minst en sökväg eller --lecture-notes. "
            f"Exempel:{Colors.RESET}\n"
            f"  uv run studyai-summarize --lecture-notes -o sammanfattning.txt\n"
            f"  uv run studyai-summarize fil1.pdf fil2.txt notes/\n"
        )
        raise SystemExit(1)

    out = args.out
    if out and not out.is_absolute():
        out = project_root() / out

    text = summarize_agent_run(roots, out_path=out)
    if not args.out:
        print(f"\n{Colors.BOLD}--- Utdrag (sista delen om långt) ---{Colors.RESET}\n")
        if "FINAL QUICK REVIEW" in text:
            tail = text[text.index("FINAL QUICK REVIEW") :].strip()
            print(tail)
        else:
            print(text[-3500:] if len(text) > 3500 else text)


if __name__ == "__main__":
    main()
