# PCSA (Personal Cognitive Sovereignty Agent)

[![CI](https://github.com/jstnorv/PCSA/actions/workflows/ci.yml/badge.svg)](https://github.com/jstnorv/PCSA/actions/workflows/ci.yml)

A local-first Python agent architecture for personal knowledge workflows using plain Markdown, local embeddings, and local LLM inference.

## Philosophy

- Absolute local control
- No mandatory external cloud dependency for baseline operation
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
3. Ensure Ollama is running locally (default endpoint: `http://localhost:11434`) for full local inference.
4. Put Markdown notes into `storage/knowledge_base/`.
5. Build or refresh the local vector index:
   ```bash
   python3 main.py ingest
   ```
6. Start the agent:
   ```bash
   python3 main.py chat
   ```

By default, chat startup does not auto-ingest to reduce first-run latency on constrained machines.
Use `python3 main.py chat --ingest-on-start` when you explicitly want re-indexing at launch.

Run local readiness diagnostics:

```bash
python3 main.py doctor
```

## Configuration

Runtime settings are managed in `config.py` with environment variable overrides:

- `PCSA_PROJECT_ROOT`
- `PCSA_KNOWLEDGE_BASE_DIR`
- `PCSA_VECTOR_DB_DIR`
- `PCSA_SESSION_STATE_PATH`
- `PCSA_SETUP_PROFILE_PATH`
- `PCSA_INGEST_MANIFEST_PATH`
- `PCSA_OLLAMA_BASE_URL`
- `PCSA_OLLAMA_CHAT_MODEL`
- `PCSA_OLLAMA_EMBED_MODEL`
- `PCSA_OLLAMA_TIMEOUT_SECONDS`
- `PCSA_HEALTH_PROBE_TIMEOUT_SECONDS`
- `PCSA_FAILURE_COOLDOWN_SECONDS`
- `PCSA_CHUNK_SIZE_CHARS`
- `PCSA_CHUNK_OVERLAP_CHARS`
- `PCSA_MEMORY_WINDOW_SIZE`
- `PCSA_RETRIEVAL_TOP_K`
- `PCSA_DELEGATION_CONFIDENCE_THRESHOLD`
- `PCSA_DELEGATION_COMPLEXITY_THRESHOLD`
- `PCSA_STRICT_LOCAL_ONLY`
- `PCSA_USER_PRIVACY_SENSITIVITY`
- `PCSA_USER_LATENCY_TOLERANCE_MS`
- `PCSA_USER_LOCAL_MEMORY_BUDGET_GB`
- `PCSA_USER_COMPUTE_BUDGET`
- `PCSA_AUTO_INGEST_ON_CHAT_START`

## Notes

- This repository intentionally avoids LangChain/LlamaIndex abstractions.
- Retrieval and default inference paths are local-first; external compute is optional and consent-gated unless trust override is explicitly enabled.

## Trust Gateway (Delegated Computation Controller)

The runtime now includes a Delegated Computation Controller that enforces a trust-preserving escalation policy.

Formal specification:

- [docs/trust_gateway_policy.md](docs/trust_gateway_policy.md)

Decision flow:

1. Route every request to local retrieval + local inference first.
2. Score the local response and task complexity.
3. If confidence is below threshold or complexity exceeds threshold, produce an escalation proposal.
4. If trust preference is `always_trust_cloud = false`, require explicit user confirmation before any external compute call.
5. If trust preference is `always_trust_cloud = true`, bypass intercept for subsequent escalations.

Escalation proposals include:

- Why local resources are insufficient for this request
- A metadata-only summary of what would be shared
- A direct confirmation prompt

CLI trust commands:

- `/trust status`
- `/trust on`
- `/trust off`

## Local Setup Optimization

At startup, PCSA now performs local setup optimization before runtime boot:

1. Detect local capabilities (CPU, memory, local model availability).
2. Apply user limitations to select bounded local defaults.
3. Persist optimization profile to `storage/setup_profile.json` (configurable).

If `PCSA_STRICT_LOCAL_ONLY=true`, external delegation is disabled at the controller level.

## Universal Floor Resilience Defaults

- Fast startup health probing uses `PCSA_HEALTH_PROBE_TIMEOUT_SECONDS`.
- Failure cooldown (`PCSA_FAILURE_COOLDOWN_SECONDS`) activates degraded mode after local model errors.
- In degraded mode, the agent returns deterministic local fallback summaries instead of hard-failing.
- Ingestion uses incremental file-hash tracking by default to avoid full re-indexing work.

## Continuous Integration

The GitHub Actions workflow validates each push and pull request to `main` by:

- Installing dependencies from `requirements.txt`
- Running `python -m compileall .`
- Running a minimal import smoke test for core modules
- Running Trust Gateway compliance checks

## Smoke Test

Run a local ingestion idempotency smoke test (no Ollama required):

```bash
python3 scripts/smoke_ingest_idempotency.py
```

Run Trust Gateway policy compliance checks:

```bash
python3 scripts/trust_gateway_compliance.py
```
