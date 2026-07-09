# PCSA (Personal Cognitive Sovereignty Agent)

[![CI](https://github.com/jstnorv/PCSA/actions/workflows/ci.yml/badge.svg)](https://github.com/jstnorv/PCSA/actions/workflows/ci.yml)

A local-first Python agent architecture for personal knowledge workflows using plain Markdown, local embeddings, and local LLM inference.

## Philosophy

- Absolute local control
- Zero external cloud dependencies
- Minimal framework bloat
- Plain-text Markdown workflow

## Architecture

- `core/`: Markdown ingestion, embedding generation, and ChromaDB vector store management
- `llm/`: Local Ollama HTTP client orchestration
- `agent/`: State machine, sliding-window memory, and execution loop
- `storage/knowledge_base/`: Raw Markdown files
- `storage/vector_db/`: Persistent local embeddings (ChromaDB)

## Project Layout

```text
PCSA/
├── agent/
├── core/
├── llm/
├── storage/
│   ├── knowledge_base/
│   └── vector_db/
├── config.py
├── main.py
└── requirements.txt
```

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama is running locally (default endpoint: `http://localhost:11434`).
4. Put Markdown notes into `storage/knowledge_base/`.
5. Start the agent:
   ```bash
   python3 main.py
   ```

## Configuration

Runtime settings are managed in `config.py` with environment variable overrides:

- `PCSA_PROJECT_ROOT`
- `PCSA_KNOWLEDGE_BASE_DIR`
- `PCSA_VECTOR_DB_DIR`
- `PCSA_OLLAMA_BASE_URL`
- `PCSA_OLLAMA_CHAT_MODEL`
- `PCSA_OLLAMA_EMBED_MODEL`
- `PCSA_OLLAMA_TIMEOUT_SECONDS`
- `PCSA_MEMORY_WINDOW_SIZE`
- `PCSA_RETRIEVAL_TOP_K`

## Notes

- This repository intentionally avoids LangChain/LlamaIndex abstractions.
- All inference and retrieval are designed for local execution.

## Continuous Integration

The GitHub Actions workflow validates each push and pull request to `main` by:

- Installing dependencies from `requirements.txt`
- Running `python -m compileall .`
- Running a minimal import smoke test for core modules
