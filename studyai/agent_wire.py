"""
Unified CLI to run existing StudyAI agents with optional MCP wiring.

Usage:
  - Choose agent: --agent tenta | summarize
  - Forward the agent's own flags after a double dash `--`
    Example:
      uv run studyai-agents --agent tenta -- 1 --mcp-server-command uv --mcp-server-arg run ...
      uv run studyai-agents --agent summarize -- --lecture-notes -o lecture_notes/out.txt
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from studyai.util.pretty_print import Colors, print_welcome


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Välj vilken befintlig StudyAI-agent som ska köras (tenta eller summarize).",
        add_help=True,
    )
    parser.add_argument(
        "--agent",
        choices=("tenta", "summarize"),
        required=True,
        help="Agent att köra.",
    )
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="Argument efter '--' skickas vidare oförändrade till vald agent.",
    )
    args = parser.parse_args(argv)

    print_welcome(
        title="StudyAI — Agent Wire",
        description="Kör en av de befintliga agenterna. MCP-flaggor förs vidare till Tenta-RAG.",
        version="0.1.0",
    )

    # Strip a leading `--` if provided by user before the forwarded args
    forwarded: List[str] = list(args.rest)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if args.agent == "tenta":
        from studyai.tenta_rag import main as tenta_main

        # Tenta-agenten förstår redan MCP-flaggor, så vi skickar dem vidare som är.
        tenta_main(forwarded or None)
        return

    if args.agent == "summarize":
        from studyai.summarize_agent import main as summarize_main

        # summarize-agenten har ingen MCP-integration. Eventuella MCP-flaggor
        # i forwarded args ignoreras av dess parser.
        summarize_main(forwarded or None)
        return

    print(f"{Colors.RED}Okänd agent: {args.agent}{Colors.RESET}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
