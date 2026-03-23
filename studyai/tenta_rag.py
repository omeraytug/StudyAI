"""
RAG-agent: skapar tentafrågor utifrån en vald föreläsning (.txt i lecture_notes/).

Bygger på mönster från nackademin-langchain-demo (rag_agent + FAISS + verktyg);
använder OpenAI (Chat + embeddings) via langchain-openai.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _prep_sys_path_when_run_as_script() -> None:
    repo = Path(__file__).resolve().parent.parent
    if (repo / "pyproject.toml").is_file():
        rs = str(repo)
        if rs not in sys.path:
            sys.path.insert(0, rs)


_prep_sys_path_when_run_as_script()

import argparse
import asyncio
import hashlib
import os
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, wrap_tool_call
from langchain.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.errors import GraphRecursionError
from pydantic import Field

from studyai.lecture_resolve import list_lecture_files, resolve_lecture_path
from studyai.paths import exam_dir, lecture_notes_dir, project_root
from studyai.util.embeddings import get_embeddings
from studyai.util.models import get_model
from studyai.util.pretty_print import Colors, get_user_input, print_tool_summary, print_welcome
from studyai.util.streaming_utils import STREAM_MODES, handle_stream

# LangChain agents compile with recursion_limit=10_000 by default — too high; allows runaway loops.
AGENT_RECURSION_LIMIT = 28

# Begränsa hur mycket kontext varje sökning returnerar (minskar upprepning/eko av samma malltext).
_MAX_RETRIEVAL_CHARS = 5200
_MAX_CHUNK_CHARS = 950

# Minsta storlek för att räkna som lyckad sparning (tom fil = misslyckat).
_MIN_SAVED_CONTENT_CHARS = 120
_MCP_TOOL_OUTPUT_MAX_CHARS = 1500


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p.strip())
    return str(content)


def _sanitize_tool_output(text: str, *, limit: int = _MCP_TOOL_OUTPUT_MAX_CHARS) -> str:
    cleaned = text.replace("\r", "\n").strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()} ... [truncated to {limit} chars]"


@wrap_tool_call
def sanitize_mcp_tool_output(request, handler):
    """Normalisera MCP tool-output innan den går tillbaka till agenten."""
    result = handler(request)
    if isinstance(result, ToolMessage):
        tool_name = request.tool.name if request.tool else request.tool_call.get("name", "mcp_tool")
        original = _extract_text(result.content)
        result.content = f"[MCP:{tool_name}] {_sanitize_tool_output(original)}"
    return result


def _filter_allowed_tools(tools: list[BaseTool], allowed_names: set[str]) -> list[BaseTool]:
    if not allowed_names:
        return tools
    return [tool for tool in tools if tool.name in allowed_names]


async def _load_mcp_tools(
    *,
    server_command: str,
    server_args: list[str],
    server_cwd: Path | None,
    allowed_tools: set[str],
) -> list[BaseTool]:
    connections: dict[str, Any] = {
        "study_mcp": {
            "transport": "stdio",
            "command": server_command,
            "args": server_args,
        }
    }
    if server_cwd is not None:
        connections["study_mcp"]["cwd"] = str(server_cwd)

    async with MultiServerMCPClient(connections=connections) as client:
        all_tools = await client.get_tools(server_name="study_mcp")
        selected = _filter_allowed_tools(all_tools, allowed_tools)
        return selected


def _message_text(msg: object) -> str:
    """Plain text från AIMessage/HumanMessage etc."""
    t = getattr(msg, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _save_tool_content_from_messages(msgs: list) -> str | None:
    """Hitta save_tentafragor/save_tentafrågor-innehåll i AI tool_calls (även om verktyget aldrig kördes)."""
    save_names = {"save_tentafrågor", "save_tentafragor"}
    for m in reversed(msgs):
        if not isinstance(m, AIMessage):
            continue
        calls = getattr(m, "tool_calls", None) or []
        for tc in calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name not in save_names:
                continue
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            if not isinstance(args, dict):
                continue
            raw = args.get("content")
            if raw is None:
                continue
            text = str(raw).strip()
            if len(text) >= _MIN_SAVED_CONTENT_CHARS:
                return text
    return None


def _longest_assistant_markdown(msgs: list) -> str | None:
    """Längsta AI-text (fallback om verktyget inte körts)."""
    best = ""
    for m in msgs:
        if not isinstance(m, AIMessage):
            continue
        text = _message_text(m).strip()
        if len(text) > len(best):
            best = text
    if len(best) < _MIN_SAVED_CONTENT_CHARS:
        return None
    return best


def _tool_confirmed_save(msgs: list) -> bool:
    """True om save-verktyget returnerat vår bekräftelsetext."""
    for m in msgs:
        if not isinstance(m, ToolMessage):
            continue
        txt = _message_text(m)
        if "Sparat:" in txt or "Sparat " in txt:
            return True
    return False


def _ensure_exam_file_written(
    *,
    exam: Path,
    default_save_name: str,
    lecture_stem: str,
    messages: list | None,
    stream_final_text: str | None,
    silent: bool = False,
) -> Path | None:
    """
    Säkerställ att tentafrågor landar på disk. Modellen glömmer ofta verktyget;
    då plockar vi text från tool_calls eller sista AI-svaret.
    """
    chosen = Path(default_save_name).name
    if not chosen.endswith((".md", ".txt")):
        chosen = f"{lecture_stem}_tentafrågor.md"
    dest = exam / chosen

    def _write(body: str, note: str) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body.strip() + "\n", encoding="utf-8")
        if not silent:
            print(f"{Colors.BRIGHT_GREEN}{note}{Colors.RESET} {dest.relative_to(project_root())}")
        return dest

    if dest.is_file():
        existing = dest.read_text(encoding="utf-8").strip()
        if len(existing) >= _MIN_SAVED_CONTENT_CHARS:
            if not silent:
                print(
                    f"{Colors.DIM}Sparad fil finns redan ({len(existing)} tecken): "
                    f"{dest.relative_to(project_root())}{Colors.RESET}"
                )
            return dest

    body: str | None = None
    source = ""

    if messages:
        if _tool_confirmed_save(messages):
            if dest.is_file() and len(dest.read_text(encoding="utf-8").strip()) >= _MIN_SAVED_CONTENT_CHARS:
                if not silent:
                    print(
                        f"{Colors.DIM}Verifierat via verktyg: "
                        f"{dest.relative_to(project_root())}{Colors.RESET}"
                    )
                return dest
        body = _save_tool_content_from_messages(messages)
        if body:
            source = " (från verktygsanrop i historiken)"
        if not body:
            body = _longest_assistant_markdown(messages)
            if body:
                source = " (fallback: sista AI-text — granska gärna)"

    if not body and stream_final_text and len(stream_final_text.strip()) >= _MIN_SAVED_CONTENT_CHARS:
        body = stream_final_text.strip()
        source = " (fallback: streamat svar — granska gärna)"

    if body:
        return _write(
            body,
            f"✓ Tentafrågor sparade{source} →",
        )

    if not silent:
        print(
            f"{Colors.YELLOW}Inga tentafrågor kunde sparas automatiskt. "
            f"Be modellen uttryckligen: 'Anropa save_tentafragor med hela markdown-innehållet'.{Colors.RESET}"
        )
    return None


def _serialize_unique_docs(docs: list[Document]) -> tuple[str, list[Document]]:
    """Deduplicera överlappande chunks och begränsa total längd."""
    seen_hashes: set[str] = set()
    kept: list[Document] = []
    parts: list[str] = []
    total_len = 0

    for doc in docs:
        raw = (doc.page_content or "").strip()
        if not raw:
            continue
        h = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        text = raw if len(raw) <= _MAX_CHUNK_CHARS else raw[:_MAX_CHUNK_CHARS] + "…"
        block = f"Källa: {doc.metadata.get('source', 'Okänd')}\nInnehåll: {text}"
        if total_len + len(block) > _MAX_RETRIEVAL_CHARS:
            break
        parts.append(block)
        total_len += len(block)
        kept.append(doc)

    if not parts:
        return "Hittade ingen relevant passage (efter deduplicering).", []
    return "\n\n".join(parts), kept


def load_lecture_vectorstore(lecture_path: Path) -> FAISS | None:
    """Chunka en föreläsning och bygg FAISS-index."""
    if not lecture_path.is_file():
        return None

    loader = TextLoader(str(lecture_path), encoding="utf-8")
    docs = loader.load()
    if not docs:
        return None

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = str(lecture_path.name)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = get_embeddings()
    return FAISS.from_documents(splits, embeddings)


def _run_tenta_once(
    lecture_path: Path,
    user_message: str,
    *,
    num_questions: int | None = None,
    stream: bool = False,
    silent: bool = False,
    extra_tools: list[BaseTool] | None = None,
    middleware: list[AgentMiddleware] | None = None,
) -> dict[str, Any]:
    """
    Kör tenta-agenten en gång (ingen stdin). Används från CLI; `silent=True` är för programmatisk körning.
    """
    root = project_root()
    exam = exam_dir()
    lecture_stem = lecture_path.stem
    default_save_name = f"{lecture_stem}_tentafrågor.md"

    vector_store = load_lecture_vectorstore(lecture_path)
    if vector_store is None:
        msg = f"Kunde inte ladda eller indexera: {lecture_path}"
        if not silent:
            print(f"{Colors.RED}{msg}{Colors.RESET}")
        return {"ok": False, "error": msg, "assistant_text": "", "saved_relative": None}

    @tool(response_format="content_and_artifact")
    def search_documents(
        query: Annotated[str, Field(description="Sökfråga på svenska eller engelska")],
    ):
        """Sök i den valda föreläsningen för relevanta utdrag att basera tentafrågor på."""
        retrieved = vector_store.similarity_search(query, k=8)
        serialized, kept = _serialize_unique_docs(retrieved)
        if not kept:
            return "Hittade ingen relevant passage.", []
        return serialized + "\n\n(Använd utdragen som stöd — kopiera inte ordagrant hela mallar.)", kept

    # Namn måste vara ASCII (OpenAI tools: ^[a-zA-Z0-9_-]+$) — inte save_tentafrågor.
    @tool
    def save_tentafragor(
        content: Annotated[
            str,
            Field(
                description=(
                    "Komplett markdown: först alla tentafrågor (numrerade), sedan avdelare, sedan facit med samma numrering — "
                    "inte facit direkt under varje fråga."
                )
            ),
        ],
        filename: Annotated[
            str,
            Field(
                description=(
                    "Filnamn endast (inga sökvägar), t.ex. lecture_1_tentafrågor.md. "
                    "Ska sluta på .md eller .txt. Lämna tomt för att använda standardfilnamn (se systemprompten)."
                )
            ),
        ] = "",
    ):
        """Spara genererade tentafrågor under projektets mapp exam/."""
        stripped = (content or "").strip()
        if len(stripped) < _MIN_SAVED_CONTENT_CHARS:
            return (
                f"Fel: 'content' måste vara minst {_MIN_SAVED_CONTENT_CHARS} tecken (markdown med tentafrågor). "
                f"Du skickade {len(stripped)} tecken — skriv ut hela tentafrågorna här, inte bara i chatten."
            )
        chosen = filename.strip() or default_save_name
        safe = Path(chosen).name
        if not safe.endswith((".md", ".txt")):
            safe = f"{lecture_stem}_tentafrågor.md"
        dest = exam / safe
        dest.write_text(stripped + "\n", encoding="utf-8")
        return f"Sparat: {dest.relative_to(project_root())}"

    model = get_model(
        temperature=0.35,
        max_tokens=3200,
    )

    n_hint = ""
    if num_questions is not None:
        n_hint = f" Användaren vill ha ungefär {num_questions} frågor."

    agent_tools: list = [search_documents, save_tentafragor]
    if extra_tools:
        agent_tools.extend(extra_tools)

    mcp_instruction = ""
    if extra_tools:
        mcp_instruction = (
            "\n\nDu har även tillgång till externa MCP-verktyg. "
            "Använd dem bara när de hjälper uppgiften och följ deras argumentkontrakt exakt."
        )

    agent = create_agent(
        model=model,
        tools=agent_tools,
        middleware=middleware or [],
        system_prompt=(
            "Du är en examinator som skapar **nya** **tentafrågor** på **svenska** utifrån kursmaterial.\n"
            f"Aktuell föreläsning (fil): **{lecture_path.name}**.\n\n"
            "VIKTIGT — undvik loopar och spam:\n"
            "- Kopiera **aldrig** långa färdiga block från föreläsningen (t.ex. färdiga Quiz/Assessment/Discussion-mallar på engelska).\n"
            "- **Upprepa inte** samma stycke, lista eller frågor om och om igen i chatten.\n"
            "- Håll svaret i chatten **kort** (1–3 meningar). Det långa innehållet ska bara skickas in i verktyget **save_tentafragor**.\n\n"
            "Arbetsflöde:\n"
            "1. Anropa `search_documents` **högst 5 gånger** med **olika** sökfrågor (nyckelbegrepp). Upprepa inte samma query.\n"
            "2. Formulera **nya** varierade tentafrågor (förklara, jämför, tillämpa) utifrån utdragen.\n"
            "3. **Obligatoriskt:** Anropa **save_tentafragor** exakt en gång och skicka **hela** tentafrågorna i parametern **content**. "
            "Utan detta verktyg sparas inget till disk — chattsvar räcker inte.\n"
            "4. Avsluta med en **kort** bekräftelse i chatten (var filen sparades).\n\n"
            "**Utdataformat i `content` (markdown):**\n"
            "- Först **endast frågorna**: en rubrik t.ex. `## Tentafrågor`, sedan numrerade frågor (1., 2., …). "
            "**Inga svar eller facit** under respektive fråga.\n"
            "- Sedan en **tydlig avdelare** (t.ex. horisontell linje `---` eller rubrik `## Facit`).\n"
            "- Därefter **facit/svar**: rubrik `## Facit` och samma numrering (1., 2., …) så att svar 1 hör till fråga 1.\n\n"
            f"Standardfilnamn: `{default_save_name}`.{n_hint}{mcp_instruction}"
        ),
    )

    run_cfg = {"recursion_limit": AGENT_RECURSION_LIMIT}
    stream_final: str | None = None
    invoke_messages: list | None = None
    assistant_text = ""
    try:
        if not stream:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=run_cfg,
            )
            invoke_messages = result.get("messages", []) or []
            if invoke_messages:
                last = invoke_messages[-1]
                assistant_text = _message_text(last)
                if not silent:
                    print(f"\n{Colors.GREEN}{assistant_text}{Colors.RESET}\n")
        else:
            process_stream = agent.stream(
                {"messages": [{"role": "user", "content": user_message}]},
                stream_mode=STREAM_MODES,
                config=run_cfg,
            )
            stream_final = handle_stream(process_stream, agent_name="Tenta-RAG")
    except GraphRecursionError:
        msg = (
            f"Agenten stoppades: för många steg (recursion_limit={AGENT_RECURSION_LIMIT}). "
            "Kör utan stream eller be om färre sökningar."
        )
        if not silent:
            print(f"\n{Colors.YELLOW}{msg}{Colors.RESET}\n")
        return {"ok": False, "error": msg, "assistant_text": assistant_text, "saved_relative": None}

    saved = _ensure_exam_file_written(
        exam=exam,
        default_save_name=default_save_name,
        lecture_stem=lecture_stem,
        messages=invoke_messages,
        stream_final_text=stream_final,
        silent=silent,
    )
    saved_rel = str(saved.relative_to(root)) if saved else None
    return {
        "ok": True,
        "error": None,
        "assistant_text": assistant_text,
        "saved_relative": saved_rel,
    }


def run_interactive(
    lecture_path: Path,
    num_questions: int | None = None,
    *,
    stream: bool = False,
    extra_tools: list[BaseTool] | None = None,
    middleware: list[AgentMiddleware] | None = None,
) -> None:
    user_ask = get_user_input(
        "Vad vill du att jag gör? (t.ex. 'Skapa 5 tentafrågor med facit', 'Bara diskussionsfrågor')"
    )
    if not user_ask:
        return
    _run_tenta_once(
        lecture_path,
        user_ask,
        num_questions=num_questions,
        stream=stream,
        silent=False,
        extra_tools=extra_tools,
        middleware=middleware,
    )


def _print_available_lectures() -> None:
    notes = lecture_notes_dir()
    files = list_lecture_files(notes)
    if not files:
        print(f"{Colors.YELLOW}Inga .txt-filer i {notes}{Colors.RESET}")
        return
    print(f"\n{Colors.BOLD}Tillgängliga föreläsningar:{Colors.RESET}")
    for p in files:
        print(f"  {Colors.CYAN}•{Colors.RESET} {p.name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="RAG-agent som skapar tentafrågor från en föreläsning och sparar under exam/.",
    )
    parser.add_argument(
        "lecture",
        nargs="?",
        default=None,
        help="Föreläsning: t.ex. 1, lecture_1, lecture_1.txt",
    )
    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        default=None,
        help="Ungefärligt antal frågor (tips till agenten).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_lectures",
        help="Lista .txt i lecture_notes/ och avsluta.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Visa token-för-token (spar-logik har då bara sista svaret som fallback — mindre tillförlitligt).",
    )
    parser.add_argument(
        "--invoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mcp-server-command",
        default=None,
        help="Kommando för extern MCP-server (t.ex. uv eller python).",
    )
    parser.add_argument(
        "--mcp-server-arg",
        action="append",
        default=[],
        help="Argument till --mcp-server-command. Upprepa flaggan för flera argument.",
    )
    parser.add_argument(
        "--mcp-server-cwd",
        type=Path,
        default=None,
        help="Arbetskatalog där MCP-servern körs (t.ex. /home/.../mcp-project).",
    )
    parser.add_argument(
        "--mcp-allow-tool",
        action="append",
        default=[],
        help="MCP-verktyg som agenten får använda. Upprepa för flera.",
    )
    parser.add_argument(
        "--list-mcp-tools",
        action="store_true",
        help="Lista tillgängliga MCP-verktyg (efter filtrering) och avsluta.",
    )
    args = parser.parse_args(argv)

    try:
        root = project_root()
    except FileNotFoundError as e:
        print(f"{Colors.RED}{e}{Colors.RESET}")
        raise SystemExit(1) from e

    os.chdir(root)

    if not args.mcp_server_command and (
        args.mcp_server_arg or args.mcp_server_cwd is not None or args.mcp_allow_tool or args.list_mcp_tools
    ):
        print(
            f"{Colors.RED}MCP-flaggor angavs utan --mcp-server-command. "
            f"Ange även kommando som startar servern.{Colors.RESET}"
        )
        raise SystemExit(1)

    mcp_tools: list[BaseTool] = []
    agent_middleware: list[AgentMiddleware] = []
    if args.mcp_server_command:
        allow = {name.strip() for name in args.mcp_allow_tool if name.strip()}
        try:
            mcp_tools = asyncio.run(
                _load_mcp_tools(
                    server_command=args.mcp_server_command,
                    server_args=args.mcp_server_arg,
                    server_cwd=args.mcp_server_cwd,
                    allowed_tools=allow,
                )
            )
        except Exception as e:
            print(f"{Colors.RED}Kunde inte ladda MCP-verktyg: {e}{Colors.RESET}")
            raise SystemExit(1) from e

        if not mcp_tools:
            print(
                f"{Colors.YELLOW}Inga MCP-verktyg tillgängliga efter filtrering. "
                f"Kontrollera --mcp-allow-tool.{Colors.RESET}"
            )
            if args.list_mcp_tools:
                return
        else:
            print_tool_summary(mcp_tools)
            agent_middleware = [sanitize_mcp_tool_output]
            names = ", ".join(t.name for t in mcp_tools)
            print(f"{Colors.DIM}MCP-verktyg aktiva i agenten: {names}{Colors.RESET}\n")

        if args.list_mcp_tools:
            return

    if args.list_lectures:
        _print_available_lectures()
        return

    print_welcome(
        title="StudyAI — Tenta-RAG",
        description=(
            "Välj en föreläsning, agenten söker i texten (FAISS) och skriver tentafrågor till mappen exam/."
        ),
        version="0.1.0",
    )

    notes = lecture_notes_dir()
    lecture_arg = args.lecture
    if not lecture_arg:
        _print_available_lectures()
        lecture_arg = get_user_input("Ange föreläsning (t.ex. 1 eller lecture_1.txt)")

    path = resolve_lecture_path(notes, lecture_arg) if lecture_arg else None
    if path is None:
        print(f"{Colors.RED}Hittade ingen föreläsning för: {lecture_arg!r}{Colors.RESET}")
        _print_available_lectures()
        raise SystemExit(1)

    print(f"{Colors.DIM}Indexerar: {path.relative_to(project_root())}{Colors.RESET}\n")
    use_stream = bool(args.stream) and not bool(args.invoke)
    run_interactive(
        path,
        num_questions=args.num_questions,
        stream=use_stream,
        extra_tools=mcp_tools,
        middleware=agent_middleware,
    )


if __name__ == "__main__":
    main()
