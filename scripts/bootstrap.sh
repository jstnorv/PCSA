#!/usr/bin/env bash
set -euo pipefail

# Run this script from the project root.
mkdir -p \
	core \
	llm \
	agent \
	storage/knowledge_base \
	storage/vector_db \
	scripts

touch \
	requirements.txt \
	config.py \
	main.py \
	scripts/bootstrap.sh \
	core/__init__.py \
	core/ingestion.py \
	core/embeddings.py \
	core/vector_store.py \
	llm/__init__.py \
	llm/ollama_client.py \
	agent/__init__.py \
	agent/memory.py \
	agent/state_machine.py \
	agent/execution_loop.py \
	storage/knowledge_base/.gitkeep \
	storage/vector_db/.gitkeep
