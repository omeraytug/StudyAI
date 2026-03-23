# StudyAI

RAG-based study helper for generating **exam questions** from lecture notes and creating **lecture summaries** from PDF/TXT. Built with **LangChain**, **OpenAI**, and **FAISS**.

## Features

| Component | Description |
|---|---|
| **Tenta-RAG** | Select a `.txt` lecture note from `lecture_notes/`, index it with embeddings, retrieve relevant passages, generate exam questions, and save them to `exam/` (markdown). |
| **Summarization** | Summarize one or more PDF/TXT files (often from `raw_lecture_notes/`) into a **LECTURE SUMMARY** in plain text. Uses multiple LLM steps and a final formatting step. |

## Requirements

- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)**
- **OpenAI API key** (`OPENAI_API_KEY`)

Defaults:
- Chat model: **`gpt-4o-mini`**
- Embeddings (FAISS): **`text-embedding-3-small`**

You can override via environment variables.

## Installation

```bash
git clone git@github.com:omeraytug/StudyAI.git
cd StudyAI
cp .env.example .env   # fill in OPENAI_API_KEY
uv sync
```

Run commands from the project root (the directory containing `lecture_notes/`).

## Usage

### Exam questions (CLI)

```bash
uv run python -m studyai.tenta_rag --list
uv run python -m studyai.tenta_rag 1
uv run python -m studyai.tenta_rag 1 --stream   # optional
```

After `uv sync`, the convenience script `studyai-tenta` should be available. If it’s not, use the `python -m studyai.tenta_rag ...` commands above.

### Summarization (CLI)

Summarize sources from `lecture_notes/` and `raw_lecture_notes/` (PDF/TXT):

```bash
uv run python -m studyai.summarize_agent --lecture-notes -o lecture_notes/min_sammanfattning.txt
```

For your own files (quote paths if they contain spaces):

```bash
uv run python -m studyai.summarize_agent \
  "raw_lecture_notes/min_fil.pdf" \
  -o lecture_notes/out.txt
```

Same as: `studyai-summarize`. Prompts and output format live in `studyai/summary_prompts.py`. Token limits are configured in `.env.example` (`STUDYAI_SUMMARY_*`).

### Unified agent wire (choose agent; optional MCP)

Run existing agents via one entrypoint:

```bash
# Tenta-RAG without MCP (no extra flags)
uv run studyai-agents --agent tenta -- 1

# Tenta-RAG with running MCP server (simple mode: one MCP URL flag)
uv run studyai-agents --agent tenta -- \
  1 \
  --mcp-url http://127.0.0.1:8003/mcp \
  --mcp-allow-tool tool_name_1

# Summarize without MCP
uv run studyai-agents --agent summarize -- \
  --lecture-notes -o lecture_notes/out.txt

# Summarize with running MCP server
uv run studyai-agents --agent summarize -- \
  --lecture-notes \
  --mcp-url http://127.0.0.1:8003/mcp \
  --mcp-allow-tool tool_name_1 \
  -o lecture_notes/out.txt
```

Basic flow:
- You can run `studyai-tenta` and `studyai-summarize` exactly as before (no MCP required).
- If you want MCP, run your MCP server and then use `--mcp-url` (or set env `MCP_SERVER_URL`).
- The selected agent will run with MCP tools only when MCP flags are provided.
- MCP server project: [https://github.com/omeraytug/mcp-project](https://github.com/omeraytug/mcp-project)

### MCP via existing Tenta-RAG agent

`studyai.tenta_rag` can connect to an already running MCP server via URL and include a filtered subset of MCP tools in the same existing agent run.

List MCP tools (after optional filtering):

```bash
uv run python -m studyai.tenta_rag \
  --mcp-url http://127.0.0.1:8003/mcp \
  --mcp-allow-tool tool_name_1 \
  --mcp-allow-tool tool_name_2 \
  --list-mcp-tools
```

Run the normal Tenta-RAG flow with MCP tools enabled:

```bash
uv run python -m studyai.tenta_rag 1 \
  --mcp-url http://127.0.0.1:8003/mcp \
  --mcp-allow-tool tool_name_1
```

Summarizer supports the same MCP flags:

```bash
uv run python -m studyai.summarize_agent --lecture-notes \
  --mcp-url http://127.0.0.1:8003/mcp \
  --mcp-allow-tool tool_name_1 \
  -o lecture_notes/out.txt
```

## Directory layout

| Directory | Role |
|---|---|
| `lecture_notes/` | `.txt` lecture notes used by Tenta-RAG (and sometimes summary output targets) |
| `raw_lecture_notes/` | Source PDF/TXT files used for summarization |
| `exam/` | Saved exam questions (markdown/text) |

## Code structure

```text
studyai/
├── tenta_rag.py            # Tenta-RAG agent (FAISS, tools, saving)
├── summarize_agent.py      # PDF/TXT -> LECTURE SUMMARY
├── summary_prompts.py      # Prompt templates for the summarization pipeline
├── paths.py                # Project root + standard directories
├── lecture_resolve.py      # Resolve lectures by name/number
└── util/                   # OpenAI models, embeddings, and utilities
```

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | **Required** |
| `OPENAI_MODEL` | Chat model (default `gpt-4o-mini`) |
| `OPENAI_EMBEDDING_MODEL` | Embeddings model (default `text-embedding-3-small`) |
| `OPENAI_BASE_URL` | Optional (proxy / OpenAI-compatible endpoint) |
| `STUDYAI_PROJECT_ROOT` | Optional: repo root if you run outside the project directory |
| `STUDYAI_SUMMARY_*` | Summarization pipeline token/chunk limits (see `.env.example`) |
