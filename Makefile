# Editable installs break `studyai-summarize` / `studyai-tenta` / `studyai-mcp` on some Python versions
# (console scripts don't see the `studyai` package). Use this once after clone or `uv sync`:
.PHONY: sync sync-cli
sync:
	uv sync

sync-cli:
	uv sync --no-editable
