"""
Utilities for streaming agent output with rich terminal logging.

DO NOT CHANGE THIS FILE.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Union

from langchain.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import StreamMode


# ---------------------------------------------------------------------------
# Stream modes used throughout the project
# ---------------------------------------------------------------------------

STREAM_MODES: Sequence[StreamMode] = ["messages", "updates"]

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------


class _C:
    """ANSI escape codes for terminal colours."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _divider(label: str, color: str = _C.BLUE) -> None:
    ts = _ts()
    label = f"{_C.BOLD}{ts} {label}{_C.RESET}"
    line = "\u2500" * 60
    print(f"\n{color}{_C.BOLD}{line}{_C.RESET}")
    print(f"{color}{_C.BOLD}  {label}{_C.RESET}")
    print(f"{color}{_C.BOLD}{line}{_C.RESET}")


def _log(icon: str, label: str, detail: str = "", color: str = _C.GRAY) -> None:
    ts = _ts()
    prefix = f"  {_C.DIM}[{ts}]{_C.RESET} {icon} {color}{label}{_C.RESET}"
    if detail:
        print(f"{prefix} {detail}")
    else:
        print(prefix)


def _log_simple(detail: str = "", color: str = _C.GRAY) -> None:
    print(f"{color}{detail}{_C.RESET}")


class _LoadingSpinner:
    """Animated loading spinner that runs in a separate thread."""

    def __init__(self, message: str):
        self.message = message
        self.running = False
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.frame_idx = 0

    def _animate(self):
        while self.running:
            frame = self.frames[self.frame_idx % len(self.frames)]
            sys.stdout.write(f"\r{_C.CYAN}{frame} {self.message}{_C.RESET}")
            sys.stdout.flush()
            self.frame_idx += 1
            time.sleep(0.1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=0.5)
            # Clear the line
            sys.stdout.write("\r" + " " * (len(self.message) + 5) + "\r")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Public helpers – call before / after streaming
# ---------------------------------------------------------------------------


def log_input(content: str, agent_name: str = "Agent") -> None:
    """Print the user's input message before the stream starts."""
    _divider(f"| INPUT \u2192 {agent_name}", _C.BLUE)
    print(f"  {content}")


def log_output(content: str, agent_name: str = "Agent") -> None:
    """Print the agent's final response after the stream ends."""
    _divider(f"\u25c0 FINISHED OUTPUT \u2190 {agent_name}", _C.GREEN)
    print(f"  {content}\n")


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------


def _msg_text(msg: Any) -> str:
    """Extract plain text from a message object."""
    # Prefer the .text property (newer LangChain)
    text = getattr(msg, "text", None)
    if text:
        return text
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


def _extract_reasoning(msg: Any) -> str:
    """Extract reasoning/thinking content from a message chunk.

    Supports:
    - OpenAI reasoning models (additional_kwargs.reasoning.summary)
    - Anthropic extended thinking (content blocks with type "thinking")
    - LangChain standard format (content_blocks with type "reasoning")
    """
    # Check for OpenAI reasoning in additional_kwargs
    additional_kwargs = getattr(msg, "additional_kwargs", {})
    if isinstance(additional_kwargs, dict):
        reasoning = additional_kwargs.get("reasoning", {})
        if isinstance(reasoning, dict) and "summary" in reasoning:
            summary = reasoning["summary"]
            if isinstance(summary, str):
                return summary

    # Check for LangChain standard content_blocks
    content_blocks = getattr(msg, "content_blocks", None)
    if content_blocks:
        for block in content_blocks:
            if isinstance(block, dict):
                if block.get("type") == "reasoning" and "reasoning" in block:
                    return str(block["reasoning"])
                if block.get("type") == "thinking" and "thinking" in block:
                    return str(block["thinking"])

    # Check for thinking/reasoning in message.content array
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning" and "reasoning" in block:
                    return str(block["reasoning"])
                if block.get("type") == "thinking" and "thinking" in block:
                    return str(block["thinking"])

    return ""


# ---------------------------------------------------------------------------
# Main stream handler
# ---------------------------------------------------------------------------


def _handle_stream_sync(
    chunks: Iterator[Any],
    agent_name: Optional[str] = None,
) -> str:
    """Internal synchronous stream handler."""
    streaming_text = False
    streaming_reasoning = False
    current_node: str | None = None
    final_text = ""
    first_chunk = True

    spinner = _LoadingSpinner(f"Skickar till {agent_name or 'Agent'}...")
    spinner.start()

    for mode, data in chunks:
        # Stop spinner on first chunk
        if first_chunk:
            spinner.stop()
            first_chunk = False
        # ==============================================================
        # messages mode – real-time token streaming
        # ==============================================================
        if mode == "messages":
            token, metadata = data
            node = metadata.get("langgraph_node", "")
            name = metadata.get("lc_agent_name", agent_name)

            # Detect node transitions
            if node != current_node:
                if streaming_text:
                    print(_C.RESET)
                    streaming_text = False
                if streaming_reasoning:
                    print(_C.RESET)
                    streaming_reasoning = False
                current_node = node

            if not isinstance(token, AIMessageChunk):
                continue

            # --- stream reasoning tokens (OpenAI, Anthropic thinking) ---
            reasoning_content = _extract_reasoning(token)
            if reasoning_content:
                if not streaming_reasoning:
                    if streaming_text:
                        print(_C.RESET)
                        streaming_text = False
                    _divider(f"| REASONING \u2190 {agent_name or 'Agent'}", _C.MAGENTA)
                    sys.stdout.write(f"  {_C.DIM}{_C.GRAY}")
                    streaming_reasoning = True
                sys.stdout.write(reasoning_content)
                sys.stdout.flush()

            # --- stream text tokens ---
            if token.text:
                if not streaming_text:
                    if streaming_reasoning:
                        print(_C.RESET)
                        streaming_reasoning = False
                    _divider(
                        f"| {agent_name or 'Agent'} | OUTPUT", _C.BLUE
                    )
                    sys.stdout.write(f"  {_C.CYAN}")
                    streaming_text = True
                sys.stdout.write(token.text)
                sys.stdout.flush()

        # ==============================================================
        # updates mode – completed agent steps
        # ==============================================================
        elif mode == "updates":
            if streaming_text:
                print(_C.RESET)
                streaming_text = False
            if streaming_reasoning:
                print(_C.RESET)
                streaming_reasoning = False

            if not isinstance(data, dict):
                continue

            for source, update in data.items():
                if source == "__interrupt__":
                    _log("\u23f8\ufe0f ", "Interrupt received", color=_C.RED)
                    continue

                if not isinstance(update, dict):
                    continue

                messages = update.get("messages", [])
                for msg in messages:
                    # -- AI message with tool calls (intermediate step) --
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args_str = ", ".join(
                                f"{k}={v!r}" for k, v in tc["args"].items()
                            )
                            _divider(
                                f"| {agent_name or 'Agent'}, TOOL CALL: {tc['name']}",
                                _C.BLUE,
                            )
                            _log_simple(
                                f"{_C.YELLOW}{_C.BOLD}{tc['name']}"
                                f"{_C.RESET}({_C.GRAY}{args_str}{_C.RESET})"
                            )

                    # -- AI message with text only (final response) --
                    elif isinstance(msg, AIMessage):
                        text = _msg_text(msg)
                        if text.strip():
                            final_text = text

                    # -- Tool response --
                    elif isinstance(msg, ToolMessage):
                        content = _msg_text(msg)
                        _divider(f"| TOOL, {tc['name']} \u2192 {agent_name or 'Agent'}", _C.BLUE)
                        _log_simple(f"{_C.GREEN}{content}{_C.RESET}")

        # ==============================================================
        # custom mode (if included)
        # ==============================================================
        elif mode == "custom":
            if streaming_text:
                print(_C.RESET)
                streaming_text = False
            if streaming_reasoning:
                print(_C.RESET)
                streaming_reasoning = False
            _log_simple(f"{_C.MAGENTA}{str(data)}{_C.RESET}")

    # Clean up trailing colour codes
    if streaming_text:
        print(_C.RESET)
    if streaming_reasoning:
        print(_C.RESET)

    # Ensure spinner is stopped
    spinner.stop()

    return final_text


async def _handle_stream_async(
    chunks: AsyncIterator[Any],
    agent_name: Optional[str] = None,
) -> str:
    """Internal asynchronous stream handler."""
    streaming_text = False
    streaming_reasoning = False
    current_node: str | None = None
    final_text = ""
    first_chunk = True

    spinner = _LoadingSpinner(f"Skickar till {agent_name or 'Agent'}...")
    spinner.start()

    async for mode, data in chunks:
        if first_chunk:
            spinner.stop()
            first_chunk = False

        if mode == "messages":
            token, metadata = data
            node = metadata.get("langgraph_node", "")
            name = metadata.get("lc_agent_name", agent_name)

            if node != current_node:
                if streaming_text:
                    print(_C.RESET)
                    streaming_text = False
                if streaming_reasoning:
                    print(_C.RESET)
                    streaming_reasoning = False
                current_node = node

            if not isinstance(token, AIMessageChunk):
                continue

            reasoning_content = _extract_reasoning(token)
            if reasoning_content:
                if not streaming_reasoning:
                    if streaming_text:
                        print(_C.RESET)
                        streaming_text = False
                    _divider(f"| REASONING \u2190 {agent_name or 'Agent'}", _C.MAGENTA)
                    sys.stdout.write(f"  {_C.DIM}{_C.GRAY}")
                    streaming_reasoning = True
                sys.stdout.write(reasoning_content)
                sys.stdout.flush()

            if token.text:
                if not streaming_text:
                    if streaming_reasoning:
                        print(_C.RESET)
                        streaming_reasoning = False
                    _divider(
                        f"| {agent_name or 'Agent'} \u2192 STREAMING OUTPUT", _C.BLUE
                    )
                    sys.stdout.write(f"  {_C.CYAN}")
                    streaming_text = True
                sys.stdout.write(token.text)
                sys.stdout.flush()

        elif mode == "updates":
            if streaming_text:
                print(_C.RESET)
                streaming_text = False
            if streaming_reasoning:
                print(_C.RESET)
                streaming_reasoning = False

            if not isinstance(data, dict):
                continue

            for source, update in data.items():
                if source == "__interrupt__":
                    _log("\u23f8\ufe0f ", "Interrupt received", color=_C.RED)
                    continue

                if not isinstance(update, dict):
                    continue

                messages = update.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args_str = ", ".join(
                                f"{k}={v!r}" for k, v in tc["args"].items()
                            )
                            _divider(
                                f"| TOOL CALL \u2192 {agent_name or 'Agent'} \u2192 {tc['name']}",
                                _C.BLUE,
                            )
                            _log_simple(
                                f"{_C.YELLOW}{_C.BOLD}{tc['name']}"
                                f"{_C.RESET}({_C.GRAY}{args_str}{_C.RESET})"
                            )

                    elif isinstance(msg, AIMessage):
                        text = _msg_text(msg)
                        if text.strip():
                            final_text = text

                    elif isinstance(msg, ToolMessage):
                        content = _msg_text(msg)
                        _divider(f"| TOOL \u2192 {agent_name or 'Agent'}", _C.BLUE)
                        _log_simple(f"{_C.GREEN}{content}{_C.RESET}")

        elif mode == "custom":
            if streaming_text:
                print(_C.RESET)
                streaming_text = False
            if streaming_reasoning:
                print(_C.RESET)
                streaming_reasoning = False
            _log_simple(f"{_C.MAGENTA}{str(data)}{_C.RESET}")

    if streaming_text:
        print(_C.RESET)
    if streaming_reasoning:
        print(_C.RESET)

    spinner.stop()

    return final_text


def handle_stream(
    chunks: Union[Iterator[Any], AsyncIterator[Any]],
    agent_name: Optional[str] = None,
) -> str:
    """Consume a ``stream_mode=["messages", "updates"]`` stream and print
    rich, colour-coded progress to the terminal.

    Automatically detects and handles both synchronous and asynchronous iterators.

    What is displayed:
    * **messages mode** – real-time text tokens as they are generated
    * **updates mode** – completed tool calls (name + args) and tool
      responses, plus the final AI response (used as return value)

    Returns:
        The text content of the agent's final AI response.
    """
    # Check if the chunks is an async iterator
    if hasattr(chunks, "__anext__"):
        # It's an async iterator - run it in the event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context, create a task
            raise RuntimeError(
                "handle_stream called with async iterator from within async context. "
                "Use 'await handle_stream_async()' instead or call from sync context."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No event loop running, we can create one
                return asyncio.run(_handle_stream_async(chunks, agent_name))  # type: ignore
            else:
                # Re-raise if it's the error we raised above
                raise
    else:
        # It's a sync iterator
        return _handle_stream_sync(chunks, agent_name)  # type: ignore


async def handle_stream_async(
    chunks: AsyncIterator[Any],
    agent_name: Optional[str] = None,
) -> str:
    """Async version of handle_stream for use in async contexts.

    Use this when you're already in an async function and have an async iterator.
    """
    return await _handle_stream_async(chunks, agent_name)


# ---------------------------------------------------------------------------
# Legacy wrapper – keeps the old function name working
# ---------------------------------------------------------------------------


def handle_stream_chunks(
    chunks: Iterator[Any],
    agent_name: str = "Agent",
    stream_mode: Sequence[str] | None = None,
    show_metadata: bool = False,
) -> str:
    """Backwards-compatible wrapper around :func:`handle_stream`."""
    return handle_stream(chunks, agent_name=agent_name)
