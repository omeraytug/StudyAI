"""Logging (samma idé som nackademin-mcp-demo/config/logging_config.py)."""

import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="\033[2m%(asctime)s\033[0m %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("fastmcp", "uvicorn", "uvicorn.access", "mcp"):
        logging.getLogger(name).setLevel(logging.WARNING)
