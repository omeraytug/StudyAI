# StudyAI

- **Tenta-RAG** — skapar **tentafrågor** från `lecture_notes/*.txt` och sparar i **`exam/`** (OpenAI + FAISS).
- **Sammanfattning** — läser **PDF + TXT** (flera filer/mappar) och sammanfattar med **små `max_tokens`-steg** för att hålla nere kostnad.

## Krav

- Python 3.14+ (enligt `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/)
- **OpenAI API-nyckel** (`OPENAI_API_KEY` i `.env`)

Standardmodeller (billiga):

- Chat: **`gpt-4o-mini`** (`OPENAI_MODEL`)
- Embeddings (FAISS): **`text-embedding-3-small`** (`OPENAI_EMBEDDING_MODEL`)

## Snabbstart

```bash
cd /path/to/StudyAI
cp .env.example .env   # sätt OPENAI_API_KEY
uv sync
```

### Om du får `ModuleNotFoundError: No module named 'studyai'`

Kortkommandona `studyai-tenta` / `studyai-summarize` fungerar inte alltid med **editable** `uv sync` (särskilt Python 3.14). Gör **något av följande**:

1. Installera projektet **icke-editably** (rekommenderat om du vill använda kortkommandona):

   ```bash
   uv sync --no-editable
   # eller: make sync-cli
   ```

2. Eller använd **modulform** (fungerar alltid från projektmappen):

   ```bash
   uv run python -m studyai.tenta_rag --list
   uv run python -m studyai.tenta_rag 1
   uv run python -m studyai.summarize_agent --lecture-notes -o sammanfattning.txt
   ```

### Tentafrågor (RAG)

```bash
uv run studyai-tenta --list
uv run studyai-tenta 1
uv run studyai-tenta 1 --stream   # valfri live-streaming
```

(Samma som: `uv run python -m studyai.tenta_rag …`)

### Sammanfatta PDF + TXT → LECTURE SUMMARY (plain text)

En **konceptbaserad** sammanfattning i **ren text** (ingen markdown från verktyget), optimerad för studier/RAG. Mallen styrs i `studyai/summary_prompts.py`. Sista steget använder fler tokens (`STUDYAI_SUMMARY_MAX_TOKENS_FINAL`, default 4096).

Allt i `lecture_notes/` och `raw_lecture_notes/` som är `.pdf` eller `.txt`:

```bash
uv run studyai-summarize --lecture-notes -o sammanfattning.txt
```

Egna filer (citattecken vid mellanslag):

```bash
uv run studyai-summarize \
  "raw_lecture_notes/AI-agenter - Föreläsning.txt" \
  "raw_lecture_notes/AI-agenter - Föreläsning (1).pdf" \
  -o lecture_notes/lecture3_agent.txt
```

(Samma som: `uv run python -m studyai.summarize_agent …`)

Övriga token-gränser: se `.env.example`.

### Köra från annan katalog

```bash
export STUDYAI_PROJECT_ROOT=/path/to/StudyAI
uv run --directory /path/to/StudyAI studyai-tenta 1
```

### MCP-server (Cursor / Claude Desktop), likt nackademin-mcp-demo

**FastMCP** med stdio (standard) eller HTTP. Verktyg: lista/läs `lecture_notes`, `raw_lecture_notes`, `exam`; `studyai_summarize_sources`; `studyai_generate_tenta_questions`.

```bash
# stdio (Cursor MCP)
cd /path/to/StudyAI
uv run studyai-mcp
# eller: uv run python -m studyai.mcp_server
```

Om du får **`ModuleNotFoundError: No module named 'studyai'`** för `studyai-mcp` (vanligt med **editable** `uv sync` på Python 3.14): kör **`uv sync --no-editable`** eller **`make sync-cli`**, eller använd alltid **`uv run python -m studyai.mcp_server`**.

**Cursor** — lägg till i MCP-inställningar (byt sökväg):

```json
{
  "mcpServers": {
    "studyai": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/StudyAI", "python", "-m", "studyai.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "din-nyckel"
      }
    }
  }
}
```

Valfritt: `STUDYAI_PROJECT_ROOT` om du inte använder `--directory`.

**HTTP** (som demo-projektets calculator):

```bash
export STUDYAI_MCP_TRANSPORT=http
export STUDYAI_MCP_HTTP_PORT=8003
uv run studyai-mcp
```

### MCP + agent som väljer verktyg (samma idé som klasskamraten)

Det finns två vanliga sätt att *«välja verktyg»*:

| Sätt | Vad det är |
|------|------------|
| **MCP Inspector** (eller liknande UI) | *Du* väljer verktyg och fyller i JSON manuellt mot en **HTTP**-MCP-server. |
| **LangChain-agent + MCP** | En **LLM** kopplas till MCP-verktygen via `langchain-mcp-adapters`; **modellen** väljer vilket verktyg som ska anropas (precis som en ReAct/agent). |

StudyAI stödjer det andra med **`studyai-mcp-agent`** (`studyai/mcp_agent_client.py`):

```bash
cd /path/to/StudyAI
# stdio: MCP-servern startas automatiskt som subprocess — en terminal räcker
uv run python -m studyai.mcp_agent_client
# eller: uv run studyai-mcp-agent
```

**HTTP** (MCP-servern måste redan köra i annan terminal):

```bash
# Terminal 1
STUDYAI_MCP_TRANSPORT=http STUDYAI_MCP_HTTP_PORT=8003 uv run python -m studyai.mcp_server

# Terminal 2
uv run python -m studyai.mcp_agent_client --http
# valfritt: --url http://127.0.0.1:8003/mcp
```

Miljö: `OPENAI_API_KEY` i `.env` (agenten **och** verktygen som anropar OpenAI behöver den).  
Valfritt: `STUDYAI_MCP_CLIENT_TRANSPORT=http` och `STUDYAI_MCP_URL` istället för `--http` / `--url`.

### Inlämning / demo (uppgift: MCP-server och koppling till agent)

Uppgiften *«Skapa en MCP server och koppla till en agent»* är uppfylld så här i det här repot:

| Krav | Var i projektet |
|------|-----------------|
| **MCP-server** | `studyai/mcp_server/` — FastMCP (`server.py`), start: `uv run python -m studyai.mcp_server` |
| **Koppling till agent** | (1) MCP-verktyg anropar samma kod som CLI: **`studyai_generate_tenta_questions`**, **`studyai_summarize_sources`**. (2) **LangChain-agent** som *väljer* verktyg: `uv run python -m studyai.mcp_agent_client` → `studyai/mcp_agent_client.py` + `langchain-mcp-adapters`. |

**Kort arkitektur (en mening):** antingen anropar en **värd** eller **LLM-agent** MCP (`tools/list`, `tools/call`); serverns verktyg kör tenta-RAG / sammanfattning (OpenAI + filer under `lecture_notes/`, `raw_lecture_notes/`, `exam/`).

**Checklista inför demo / inlämning**

1. **Starta servern** — stdio (`uv run python -m studyai.mcp_server`) om värden använder Cursor/MCP-värd, eller **HTTP** för t.ex. MCP Inspector:
   ```bash
   STUDYAI_MCP_TRANSPORT=http STUDYAI_MCP_HTTP_PORT=8003 uv run python -m studyai.mcp_server
   ```
2. **Visa verktyg** — klienten ska lista verktyg (t.ex. `studyai_list_lecture_notes`, `studyai_generate_tenta_questions`, `studyai_summarize_sources`).
3. **Anropa minst ett agent-verktyg** — t.ex. `studyai_generate_tenta_questions` med `lecture: "1"` och en `prompt` på svenska; visa att en fil skapas under **`exam/`**, eller anropa `studyai_summarize_sources` med giltiga `relative_paths` och ev. `output_relative`.
4. **Alternativ: visa LLM som väljer verktyg** — kör `uv run python -m studyai.mcp_agent_client` och ställ en fråga som kräver verktyg (t.ex. *«Vilka föreläsningar finns?»* eller *«Skapa tre tentafrågor från föreläsning 1»*).
5. **Säkerställ API-nyckel** — `OPENAI_API_KEY` i `.env` (eller i miljön som startar servern) för steg som anropar modellen.

**Tips:** [MCP Inspector](https://github.com/modelcontextprotocol/inspector) (`npx @modelcontextprotocol/inspector`) kan användas mot HTTP-Servern för att lista och **manuellt** välja verktyg utan Cursor.

## Miljövariabler

| Variabel | Beskrivning |
|----------|-------------|
| `OPENAI_API_KEY` | **Krävs** |
| `OPENAI_MODEL` | Chatmodell (standard `gpt-4o-mini`) |
| `OPENAI_EMBEDDING_MODEL` | Embeddings (standard `text-embedding-3-small`) |
| `OPENAI_BASE_URL` | Valfritt (t.ex. proxy / Azure-kompatibel endpoint) |
| `STUDYAI_PROJECT_ROOT` | Valfritt: repo-root om `lecture_notes/` inte hittas från cwd |
| `STUDYAI_MCP_TRANSPORT` | `stdio` (default) eller `http` för MCP-servern |
| `STUDYAI_MCP_HTTP_HOST` / `STUDYAI_MCP_HTTP_PORT` | HTTP-MCP (default `0.0.0.0:8003`) |
| `STUDYAI_MCP_CLIENT_TRANSPORT` | För `mcp_agent_client`: `stdio` (default) eller `http` |
| `STUDYAI_MCP_URL` | HTTP-MCP-URL för agent-klienten (default `http://127.0.0.1:8003/mcp`) |
| `STUDYAI_SUMMARY_*` | Valfritt: chunk-storlek, max chunks, `max_tokens` per steg + `STUDYAI_SUMMARY_MAX_TOKENS_FINAL` |

## Struktur

- `lecture_notes/` — textfiler för tenta-RAG
- `raw_lecture_notes/` — t.ex. PDF:er för sammanfattning
- `exam/` — genererade tentafrågor
- `studyai/tenta_rag.py` — tenta-agent
- `studyai/mcp_server/` — MCP-server (`studyai-mcp`)
- `studyai/mcp_agent_client.py` — LangChain-agent som använder MCP-verktyg (`studyai-mcp-agent`)
- `studyai/summarize_agent.py` — multi-fil PDF/TXT → LECTURE SUMMARY
- `studyai/summary_prompts.py` — mall och extraktionsprompter
- `studyai/util/models.py`, `embeddings.py` — OpenAI
