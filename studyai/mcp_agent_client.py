"""
LangChain-agent som ansluter till StudyAI MCP-servern och **väljer verktyg själv**.

Samma idé som ofta visas i kursen: MCP-server (verktyg) + separat **agent** (LLM som
anropar tools/call via langchain-mcp-adapters).

Kör:
  # stdio: startar MCP-servern som subprocess (ingen separat terminal)
  uv run python -m studyai.mcp_agent_client

  # http: starta först MCP-servern med STUDYAI_MCP_TRANSPORT=http, sedan:
  STUDYAI_MCP_CLIENT_TRANSPORT=http uv run python -m studyai.mcp_agent_client
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from langchain_mcp_adapters.client import MultiServerMCPClient

from studyai.util.models import get_model
from studyai.util.pretty_print import Colors, print_mcp_tools, print_welcome

AGENT_RECURSION_LIMIT = 40

# Miljövariabler som inte ska skickas till MCP-subprocessen (zsh/powerlevel10k PS1 innehåller `${…}`
# som varnar och kan störa; MCP SDK förväntar sig en enkel miljö).
_STDIO_ENV_DROP = frozenset(
    {"PS1", "PS2", "PS3", "PS4", "PROMPT_COMMAND", "RPROMPT", "RPS1"}
)


def _repo_root() -> Path:
    """
    Repo-root (mapp med pyproject.toml + lecture_notes/).

    `mcp_agent_client.py` ligger i `studyai/` — `parents[2]` skulle bli fel (t.ex. Desktop).
    """
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").is_file() and (p / "lecture_notes").is_dir():
            return p
    raise FileNotFoundError(
        "Hittade inte StudyAI-repo (sökte efter pyproject.toml + lecture_notes/ "
        "uppåt från studyai/mcp_agent_client.py)."
    )


def _stdio_subprocess_env(repo: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _STDIO_ENV_DROP and not k.startswith("P9K_")
    }
    env["STUDYAI_PROJECT_ROOT"] = str(repo)
    # Under stdio-MCP måste subprocess alltid vara stdio. Om shell har STUDYAI_MCP_TRANSPORT=http
    # (t.ex. efter tester) ärvs det annars hit → servern försöker binda :8003 → Errno 48.
    env["STUDYAI_MCP_TRANSPORT"] = "stdio"
    # Undvik stor FastMCP-ruta på stderr + eventuella stdio-konflikter vid MCP-handshake
    env.setdefault("FASTMCP_SHOW_SERVER_BANNER", "false")
    return env


def _message_text(msg: object) -> str:
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


def _build_mcp_client(*, transport: str, http_url: str) -> MultiServerMCPClient:
    repo = _repo_root()
    transport = transport.strip().lower()
    if transport == "http":
        return MultiServerMCPClient(
            {
                "studyai": {
                    "transport": "http",
                    "url": http_url,
                }
            }
        )
    # stdio: spawn `uv run python -m studyai.mcp_server` i repot
    env = _stdio_subprocess_env(repo)
    return MultiServerMCPClient(
        {
            "studyai": {
                "transport": "stdio",
                "command": "uv",
                "args": ["run", "python", "-m", "studyai.mcp_server"],
                "cwd": str(repo),
                "env": env,
            }
        }
    )


async def _async_chat_loop(*, transport: str, http_url: str) -> None:
    load_dotenv(_repo_root() / ".env")

    print_welcome(
        title="StudyAI — MCP-agent (LangChain)",
        description=(
            "Ansluter till StudyAI MCP-servern, laddar alla verktyg, och låter modellen "
            "välja vilket verktyg som ska anropas. Skriv quit för att avsluta."
        ),
        version="0.1.0",
    )

    if transport == "http":
        print(
            f"{Colors.DIM}HTTP-transport → {http_url} "
            f"(se till att MCP-servern kör med STUDYAI_MCP_TRANSPORT=http){Colors.RESET}\n"
        )
    else:
        print(
            f"{Colors.DIM}stdio-transport → startar MCP-server som subprocess (cwd: {_repo_root()}){Colors.RESET}\n"
        )

    client = _build_mcp_client(transport=transport, http_url=http_url)
    tools = await client.get_tools()
    print_mcp_tools(tools, server_name="studyai")

    model = get_model(temperature=0.25, max_tokens=2500)
    system_prompt = (
        "Du är en studieassistent med tillgång till StudyAI via MCP-verktyg.\n"
        "Svara på svenska om användaren skriver svenska.\n\n"
        "Regler:\n"
        "- Välj **själv** vilket verktyg som passar frågan (lista filer, läsa text, sammanfatta, tentafrågor).\n"
        "- Anropa verktyg när du behöver fakta från disk — gissa inte filnamn; lista först om du är osäker.\n"
        "- **Sammanfattning:** PDF/original ligger oftast i **raw_lecture_notes/**. Anropa "
        "**studyai_list_raw_lecture_notes** (och vid behov **studyai_list_lecture_notes**) innan "
        "**studyai_summarize_sources** så `relative_paths` får exakta namn (t.ex. med `(1).pdf`). "
        "Servern matchar också basnamn i raw → lecture_notes → exam om mappen är fel.\n"
        "- `studyai_summarize_sources` och `studyai_generate_tenta_questions` kostar API-anrop och tid — "
        "använd dem när användaren ber om sammanfattning eller tentafrågor.\n"
        "- Håll svar kort i chatten om användaren inte ber om lång utskrift.\n"
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    cfg = {"recursion_limit": AGENT_RECURSION_LIMIT}
    prompt = f"\n{Colors.BOLD}Du:{Colors.RESET} "
    while True:
        try:
            # input() i en tråd så event loop inte blockar (MCP-verktyg är async).
            line = (await asyncio.to_thread(input, prompt)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break

        try:
            # MCP-verktyg från langchain-mcp-adapters är async → måste använda ainvoke, inte invoke.
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": line}]},
                config=cfg,
            )
        except Exception as e:
            print(f"{Colors.RED}Fel: {e}{Colors.RESET}")
            continue

        msgs = result.get("messages") or []
        text = ""
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                text = _message_text(m)
                if text.strip():
                    break
        print(f"\n{Colors.BRIGHT_GREEN}Agent:{Colors.RESET} {text or '(tomt svar)'}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="LangChain-agent mot StudyAI MCP (modellen väljer verktyg).",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Anslut till redan körande HTTP-MCP (STUDYAI_MCP_URL, default http://127.0.0.1:8003/mcp).",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("STUDYAI_MCP_URL", "http://127.0.0.1:8003/mcp"),
        help="HTTP endpoint för MCP (vid --http).",
    )
    args = parser.parse_args(argv)

    transport = os.getenv("STUDYAI_MCP_CLIENT_TRANSPORT", "").strip().lower()
    if args.http:
        transport = "http"
    if not transport:
        transport = "stdio"

    asyncio.run(_async_chat_loop(transport=transport, http_url=args.url))


if __name__ == "__main__":
    main()
