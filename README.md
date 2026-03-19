# StudyAI

RAG-based study helper for generating **exam questions** from lecture notes and creating **lecture summaries** from PDF/TXT. Built with **LangChain**, **OpenAI**, and **FAISS**. Includes an optional **MCP (FastMCP) server** to expose tools to MCP-compatible clients.

## Features

| Component | Description |
|---|---|
| **Tenta-RAG** | Select a `.txt` lecture note from `lecture_notes/`, index it with embeddings, retrieve relevant passages, generate exam questions, and save them to `exam/` (markdown). |
| **Summarization** | Summarize one or more PDF/TXT files (often from `raw_lecture_notes/`) into a **LECTURE SUMMARY** in plain text. Uses multiple LLM steps and a final formatting step. |
| **MCP** | Exposes the same capabilities as tools (list/read files, summarize, generate exam questions) for MCP clients. |
| **MCP-agent** | A terminal-based LangChain agent that connects to the StudyAI MCP server and lets the model choose which tool to call. |

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

### MCP server

**Stdio** (clients that start the process themselves, e.g. Cursor):

```bash
uv run python -m studyai.mcp_server
```

**HTTP** (external/local clients):

```bash
STUDYAI_MCP_TRANSPORT=http STUDYAI_MCP_HTTP_PORT=8003 uv run python -m studyai.mcp_server
```

Tools include: listing/reading `lecture_notes`, `raw_lecture_notes`, and `exam`; summarizing sources; generating exam questions for a selected `.txt` lecture note.

**Cursor example** (adjust the path):

```json
{
  "mcpServers": {
    "studyai": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/StudyAI", "python", "-m", "studyai.mcp_server"],
      "env": { "OPENAI_API_KEY": "…" }
    }
  }
}
```

### MCP-agent (LLM + tools in terminal)

Runs the MCP server as a subprocess and chats with the model (which calls tools):

```bash
uv run python -m studyai.mcp_agent_client
```

For an already running HTTP MCP server:
```bash
uv run python -m studyai.mcp_agent_client --http
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
├── mcp_server/             # FastMCP server + tools
├── mcp_agent_client.py     # LangChain agent that uses MCP tools
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
| `STUDYAI_MCP_TRANSPORT` | `stdio` or `http` for the MCP server |
| `STUDYAI_MCP_HTTP_HOST` / `STUDYAI_MCP_HTTP_PORT` | HTTP MCP host/port (default port `8003`) |
| `STUDYAI_MCP_URL` | MCP URL for `mcp_agent_client` when using `--http` |
| `STUDYAI_SUMMARY_*` | Summarization pipeline token/chunk limits (see `.env.example`) |
