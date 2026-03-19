"""Request logging middleware (anpassad från nackademin-mcp-demo/config/custom_logging_config.py)."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_WHITE = "\033[97m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_RED = "\033[91m"
    RED = "\033[31m"
    BRIGHT_BLACK = "\033[90m"


class RequestLoggingMiddleware(Middleware):
    """Logga tools/list och tools/call (samma mönster som demo-projektet)."""

    def _format_data(self, data: Any, indent: int = 0) -> str:
        try:
            if isinstance(data, (dict, list)):
                json_str = json.dumps(data, indent=2, ensure_ascii=False)
                lines = json_str.split("\n")
                if len(lines) > 10:
                    return (
                        "\n".join(lines[:10])
                        + f"\n{Colors.DIM}... ({len(lines) - 10} more lines){Colors.RESET}"
                    )
                return json_str
            return str(data)
        except Exception:
            return str(data)

    def _log_separator(self, char: str = "─", length: int = 80, color: str = Colors.BRIGHT_BLACK) -> None:
        logging.info(f"{color}{char * length}{Colors.RESET}")

    def _extract_tools_from_result(self, result: Any) -> list[str]:
        tools: list[str] = []
        if isinstance(result, dict) and "tools" in result:
            for tool in result["tools"]:
                if isinstance(tool, dict):
                    name = tool.get("name", "unknown")
                    desc = tool.get("description", "")
                    tools.append(f"{name}: {desc}")
        elif isinstance(result, list):
            for tool in result:
                if hasattr(tool, "name"):
                    name = tool.name
                    desc = getattr(tool, "description", "")
                    tools.append(f"{name}: {desc}")
                elif isinstance(tool, dict):
                    name = tool.get("name", "unknown")
                    desc = tool.get("description", "")
                    tools.append(f"{name}: {desc}")
        return tools

    def _extract_tool_call_info(self, message: Any) -> tuple[str, dict]:
        if hasattr(message, "name") and hasattr(message, "arguments"):
            return message.name, message.arguments
        if isinstance(message, dict):
            return message.get("name", "unknown"), message.get("arguments", {})
        return "unknown", {}

    def _extract_tool_result(self, result: Any) -> Any:
        if hasattr(result, "content"):
            if isinstance(result.content, list) and len(result.content) > 0:
                first_item = result.content[0]
                if hasattr(first_item, "text"):
                    return first_item.text
            return result.content
        if isinstance(result, dict):
            if "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        return first_item["text"]
                return content
            if "result" in result:
                return result["result"]
        return result

    async def on_message(self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any]) -> Any:
        method = context.method
        try:
            result = await call_next(context)
            if method == "tools/list":
                logging.info(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}📋 TOOLS LIST{Colors.RESET}")
                self._log_separator("═", 80, Colors.BRIGHT_CYAN)
                for i, tool in enumerate(self._extract_tools_from_result(result), 1):
                    logging.info(f"{Colors.CYAN}{i}.{Colors.RESET} {Colors.BRIGHT_WHITE}{tool}{Colors.RESET}")
                self._log_separator("═", 80, Colors.BRIGHT_CYAN)
                logging.info("")
            elif method == "tools/call":
                tool_name, arguments = self._extract_tool_call_info(context.message)
                tool_result = self._extract_tool_result(result)
                logging.info(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}🔧 TOOL CALL{Colors.RESET}")
                self._log_separator("═", 80, Colors.BRIGHT_GREEN)
                logging.info(f"{Colors.BOLD}Tool:{Colors.RESET} {Colors.BRIGHT_YELLOW}{tool_name}{Colors.RESET}")
                logging.info(f"{Colors.CYAN}┌─ Arguments{Colors.RESET}")
                for line in self._format_data(arguments).split("\n"):
                    logging.info(f"{Colors.CYAN}│{Colors.RESET} {line}")
                logging.info(f"{Colors.CYAN}──────────────────────{Colors.RESET}")
                logging.info(f"{Colors.MAGENTA}┌─ Response{Colors.RESET}")
                for line in self._format_data(tool_result).split("\n"):
                    logging.info(f"{Colors.MAGENTA}│{Colors.RESET} {line}")
                logging.info(f"{Colors.MAGENTA}──────────────────────{Colors.RESET}")
                self._log_separator("═", 80, Colors.BRIGHT_GREEN)
                logging.info("")
            return result
        except Exception as e:
            logging.info(f"{Colors.BRIGHT_RED}{Colors.BOLD}❌ ERROR: {method}{Colors.RESET} {e}")
            raise
