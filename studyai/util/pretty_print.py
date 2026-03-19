"""
Utilities for pretty printing various objects in the terminal.

DO NOT CHANGE THIS FILE.
"""

from typing import Any, List, Optional
from langchain_core.tools import BaseTool


# ANSI color codes
class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    
    # Colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


def _format_type(type_info: Any) -> str:
    """Format a type annotation for display."""
    if isinstance(type_info, dict):
        type_name = type_info.get("type", "any")
        if type_name == "object":
            return "object"
        elif type_name == "array":
            items = type_info.get("items", {})
            if isinstance(items, dict):
                item_type = items.get("type", "any")
                return f"array[{item_type}]"
            return "array"
        return type_name
    return str(type_info)


def _format_parameter(name: str, schema: dict, required: bool = False) -> str:
    """Format a single parameter for display."""
    type_str = _format_type(schema)
    description = schema.get("description", "")
    
    # Build the parameter line
    parts = []
    
    # Parameter name with required indicator
    if required:
        parts.append(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{name}{Colors.RESET}")
        parts.append(f"{Colors.RED}*{Colors.RESET}")
    else:
        parts.append(f"{Colors.CYAN}{name}{Colors.RESET}")
    
    # Type
    parts.append(f"{Colors.GRAY}:{Colors.RESET}")
    parts.append(f"{Colors.YELLOW}{type_str}{Colors.RESET}")
    
    param_line = " ".join(parts)
    
    # Add description if available
    if description:
        return f"{param_line}\n      {Colors.DIM}{description}{Colors.RESET}"
    
    return param_line


def print_mcp_tools(tools: List[BaseTool], server_name: Optional[str] = None) -> None:
    """Pretty print a list of MCP tools with their schemas.
    
    Args:
        tools: List of LangChain tools from an MCP server
        server_name: Optional name of the MCP server
    """
    if not tools:
        print(f"{Colors.YELLOW}No tools found{Colors.RESET}")
        return
    
    # Header
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}")
    if server_name:
        print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}  MCP Tools from: {Colors.BRIGHT_CYAN}{server_name}{Colors.RESET}")
    else:
        print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}  MCP Tools{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}\n")
    
    # Print each tool
    for i, tool in enumerate(tools, 1):
        # Tool name and description
        print(f"{Colors.BOLD}{Colors.GREEN}{i}. {tool.name}{Colors.RESET}")
        
        if tool.description:
            print(f"   {Colors.DIM}{tool.description}{Colors.RESET}")
        
        # Get the tool's input schema
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                if hasattr(tool.args_schema, "schema"):
                    schema = tool.args_schema.schema()  # type: ignore
                else:
                    schema = tool.args_schema  # type: ignore
                
                if isinstance(schema, dict):
                    properties = schema.get("properties", {})
                    required_fields = schema.get("required", [])
                else:
                    properties = {}
                    required_fields = []
            except Exception:
                properties = {}
                required_fields = []
            
            if properties:
                print(f"\n   {Colors.BOLD}Parameters:{Colors.RESET}")
                for param_name, param_schema in properties.items():
                    is_required = param_name in required_fields
                    formatted = _format_parameter(param_name, param_schema, is_required)
                    print(f"     {formatted}")
        
        # Separator between tools
        if i < len(tools):
            print(f"\n{Colors.GRAY}{'─' * 70}{Colors.RESET}\n")
    
    # Footer
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}Total: {Colors.BRIGHT_CYAN}{len(tools)}{Colors.RESET} {Colors.BOLD}tool(s){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}\n")
    
    # Legend
    print(f"{Colors.DIM}Legend: {Colors.BRIGHT_CYAN}{Colors.BOLD}parameter{Colors.RESET}{Colors.RED}*{Colors.RESET}{Colors.DIM} = required, {Colors.CYAN}parameter{Colors.RESET}{Colors.DIM} = optional{Colors.RESET}\n")


def print_tool_summary(tools: List[BaseTool]) -> None:
    """Print a compact summary of tools (just names and descriptions).
    
    Args:
        tools: List of LangChain tools
    """
    if not tools:
        print(f"{Colors.YELLOW}No tools available{Colors.RESET}")
        return
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Available Tools ({len(tools)}):{Colors.RESET}")
    for tool in tools:
        desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
        print(f"  {Colors.GREEN}•{Colors.RESET} {Colors.BOLD}{tool.name}{Colors.RESET}")
        if desc:
            print(f"    {Colors.DIM}{desc}{Colors.RESET}")
    print()


def get_user_input(
    prompt: Optional[str] = "Input",
    agent_name: Optional[str] = None,
    show_help: bool = True
) -> str:
    """Get user input with a pretty formatted prompt.
    
    Args:
        prompt: The prompt text to show
        agent_name: Optional name of the agent being addressed
        show_help: Whether to show help text about available commands
    
    Returns:
        The user's input string
    """
    # Get input with colored prompt
    try:
        user_input = input(
            f"\n{Colors.BOLD}{Colors.BRIGHT_GREEN}➤ {Colors.BRIGHT_CYAN}{prompt}:{Colors.RESET} "
        ).strip()
        print()  # Add spacing after input
        return user_input
    except (EOFError, KeyboardInterrupt):
        print(f"\n\n{Colors.YELLOW}Avslutar...{Colors.RESET}")
        return ""


def print_welcome(
    title: str = "LangChain Agent",
    description: Optional[str] = None,
    version: Optional[str] = None
) -> None:
    """Print a welcome banner for the application.
    
    Args:
        title: The title to display
        description: Optional description text
        version: Optional version string
    """
    width = 70
    
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}  🤖 {title}{Colors.RESET}")
    
    if version:
        print(f"{Colors.DIM}  v{version}{Colors.RESET}")
    
    if description:
        # Word wrap the description
        words = description.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width - 4:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        print()
        for line in lines:
            print(f"{Colors.DIM}  {line}{Colors.RESET}")
    
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * width}{Colors.RESET}\n")


def get_user_decision() -> dict:
    """Hämta användarens beslut för en interrupt."""
    print(f"\n{Colors.BOLD}Välj ett alternativ:{Colors.RESET}")
    print(f"  {Colors.BRIGHT_GREEN}1.{Colors.RESET} Godkänn och kör")
    print(f"  {Colors.BRIGHT_RED}2.{Colors.RESET} Avvisa")

    while True:
        try:
            choice = input(
                f"\n{Colors.BOLD}{Colors.BRIGHT_GREEN}➤ {Colors.BRIGHT_CYAN}Ditt val (1/2):{Colors.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{Colors.YELLOW}Avslutar...{Colors.RESET}")
            return {"type": "reject", "feedback": "Användaren avbröt"}

        if choice == "1":
            return {"type": "approve"}
        elif choice == "2":
            return {"type": "reject", "feedback": "User rejected the operation"}
        else:
            print(f"{Colors.YELLOW}Ogiltigt val. Välj 1 eller 2.{Colors.RESET}")


def print_interrupt_info(result) -> bool:
    """Visa information om en interrupt som kräver godkännande."""
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return False

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_YELLOW}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}  ⚠  HUMAN-IN-THE-LOOP: Godkännande krävs!{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}{'═' * 60}{Colors.RESET}\n")

    for interrupt in interrupts:
        for action in interrupt.value.get("action_requests", []):
            print(f"  {Colors.BOLD}Verktyg:{Colors.RESET}   {Colors.BRIGHT_CYAN}{action.get('name', 'N/A')}{Colors.RESET}")
            args = action.get('arguments') or action.get('args', {})
            print(f"  {Colors.BOLD}Argument:{Colors.RESET}  {Colors.DIM}{args}{Colors.RESET}")

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_YELLOW}{'═' * 60}{Colors.RESET}")
    return True


def print_goodbye(message: str = "Tack för att du använde agenten!") -> None:
    """Print a goodbye message.
    
    Args:
        message: The goodbye message to display
    """
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}👋 {message}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'─' * 70}{Colors.RESET}\n")
