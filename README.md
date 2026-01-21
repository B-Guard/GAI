# SALVIA - RAG Chatbot for Educational Q&A

**S**low **A**nd **L**ow-**V**alue **I**ntelligent **A**gent

A retrieval-augmented generation (RAG) chatbot designed to help students find information in their lecture notes. SALVIA compares two retrieval approaches: FAISS vector search (fast semantic similarity) and Neo4j GraphRAG (knowledge graph with entity relationships).

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12">
  <img src="https://img.shields.io/badge/LLM-Llama_3.2-green" alt="Llama 3.2">
  <img src="https://img.shields.io/badge/Vector-FAISS-orange" alt="FAISS">
  <img src="https://img.shields.io/badge/Graph-Neo4j-brightgreen" alt="Neo4j">
</p>

---

## Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [FAISS Retriever](#1-faiss-retriever)
  - [Neo4j GraphRAG](#2-neo4j-graphrag)
  - [Web UI (Streamlit)](#3-web-ui-streamlit)
  - [Evaluation](#4-evaluation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance Notes](#-performance-notes)

---

## Features

- **Dual Retrieval Systems**: 
  - **FAISS**: Fast semantic similarity search using sentence embeddings
  - **Neo4j GraphRAG**: Knowledge graph with entity extraction and relationship mapping
- **Source Citations**
- **Streamlit Web UI**
- **Local LLM**: Uses Ollama (Llama 3.2)
- **Evaluation Framework**: Built-in metrics for faithfulness, relevancy, correctness, and context relevance

---

## Prerequisites

### Required Software

| Software | Version | Download |
|----------|---------|----------|
| **Python** | 3.12.x | [python.org](https://www.python.org/downloads/) |
| **Ollama** | 0.14.x+ | [ollama.com/download](https://ollama.com/download) |
| **Docker Desktop** | Latest | [docker.com](https://www.docker.com/products/docker-desktop/) (only for Neo4j) |


---

## Installation

### 1. Clone Repository and Setup Environment

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `sentence-transformers` - For embeddings
- `faiss-cpu` - For vector search
- `neo4j` and `neo4j-graphrag` - For graph database
- `langchain` ecosystem - For document processing
- `streamlit` - For web UI
- `ollama` - For local LLM

### 3. Pull Ollama Model

```bash
# Pull Llama 3.2 (this will take a few minutes)
ollama pull llama3.2:latest

# Verify it's installed
ollama list
```

You should see `llama3.2:latest` in the list.

### 4. Configure Environment Variables

```bash
# Copy example config
cp .env.example .env

# Edit .env with your preferred editor
```

**Recommended `.env` settings:**

```env
# Chunking settings
CHUNK_SIZE=512
CHUNK_OVERLAP=128

# Models
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHAT_MODEL_NAME=llama3.2:latest
EVAL_MODEL_NAME=llama3.2:latest

# Retrieval settings
SEMANTIC_WEIGHT=0.7
KEYWORD_WEIGHT=0.3
TOP_K=5

# Neo4j settings (only needed if using GraphRAG)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass123
NEO4J_VECTOR_INDEX=chunk_embedding_idx

# Neo4j ingestion settings
INGEST_TIMEOUT_SEC=1500

# FAISS settings
FAISS_INDEX_PATH=data/vector_store_faiss.pkl
```

---

## Quick Start

### Option A: FAISS Only (Fastest)

```bash
# 1. Ingest documents into FAISS
python main.py ingest \
  --llm-model llama3.2:latest \
  --retriever faiss \
  --path "data/texts/*.txt"

# 2. Chat via command line
python main.py chat \
  --llm-model llama3.2:latest \
  --retriever faiss

# 3. Or use the web UI
streamlit run app.py
# Then select "faiss" in sidebar and click "Initialize Chat"
```

### Option B: Neo4j GraphRAG (Using Pre-loaded Database Dump)

```bash
# 1. Ensure dump file exists
ls dump/neo4j.dump 

# 2. Start Neo4j with Docker Compose
docker compose up -d

# 3. Load the pre-ingested database
# Copy dump to import directory
cp dump/neo4j.dump neo4jimport/neo4j.dump

# Stop Neo4j
docker stop neo4j-gai

# Load the dump
docker run --rm \
  -v "$(pwd)/neo4jdata:/data" \
  -v "$(pwd)/neo4jlogs:/logs" \
  -v "$(pwd)/neo4jimport:/import" \
  -v "$(pwd)/neo4jplugins:/plugins" \
  neo4j:5.26.19 \
  neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true

# Restart Neo4j
docker start neo4j-gai

# 4. Verify the database loaded correctly
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 \
  "MATCH (c:Chunk) RETURN count(c) AS chunks;"
# Should show a non-zero number of chunks

# 5. Start chatting via command line
python main.py chat \
  --llm-model llama3.2:latest \
  --retriever neo4j

# 6. Or use the web UI
streamlit run app.py
# Select "neo4j" in sidebar and click "Initialize Chat"
```

---

## Usage

### 1. FAISS Retriever

FAISS (Facebook AI Similarity Search) uses semantic embeddings for fast similarity search.

#### Ingesting Documents

```bash
python main.py ingest \
  --llm-model llama3.2:latest \
  --retriever faiss \
  --path "data/texts/*.txt"
```

**What happens:**
1. Reads all `.txt` files matching the pattern
2. Splits them into chunks (512 tokens with 128 overlap by default)
3. Generates embeddings using `sentence-transformers`
4. Stores embeddings in `data/vector_store_faiss.pkl`

**Time estimate**: ~1-2 minutes for 10 documents

**Adding more documents:**
```bash
# Just run ingestion again with new files
python main.py ingest \
  --llm-model llama3.2:latest \
  --retriever faiss \
  --path "path/to/new/docs/*.txt"

# FAISS will append to existing index
```

#### Chatting with FAISS

**Command line:**
```bash
python main.py chat \
  --llm-model llama3.2:latest \
  --retriever faiss \
  --top-k 5

# Type your questions
You: What is attention in transformers?
SALVIA: [streaming answer with [s1], [s2] citations]

# Type 'exit' to quit
```

**Adjusting retrieval:**
```bash
# Retrieve more sources
python main.py chat \
  --llm-model llama3.2:latest \
  --retriever faiss \
  --top-k 10

# Use different embedding model
python main.py chat \
  --llm-model llama3.2:latest \
  --retriever faiss \
  --embedding-model all-mpnet-base-v2
```

---

### 2. Neo4j GraphRAG

Neo4j GraphRAG extracts entities and relationships to build a knowledge graph, then retrieves context using both similarity search and graph neighborhood information.

#### Setting Up Neo4j

**Start Neo4j with Docker:**

```bash
# Start Neo4j container
docker compose up -d

# Check it's running
docker ps --filter "name=neo4j"

# Neo4j Browser will be available at:
# http://localhost:7474
# Username: neo4j
# Password: neo4jpass123
```

**Verify connection:**
```bash
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 \
  "RETURN 1 AS test;"

# Should return: test | 1
```

#### Option A: Restore from Pre-ingested Dump (Recommended)

**This is faster than ingesting from scratch.**

```bash
# 1. Ensure dump exists
ls dump/neo4j.dump

# 2. Copy to import directory
cp dump/neo4j.dump neo4jimport/neo4j.dump

# Windows PowerShell:
# Copy-Item ".\dump\neo4j.dump" ".\neo4jimport\neo4j.dump" -Force

# 3. Stop Neo4j
docker stop neo4j-gai

# 4. Load the dump
docker run --rm \
  -v "$(pwd)/neo4jdata:/data" \
  -v "$(pwd)/neo4jlogs:/logs" \
  -v "$(pwd)/neo4jimport:/import" \
  -v "$(pwd)/neo4jplugins:/plugins" \
  neo4j:5.26.19 \
  neo4j-admin database load neo4j \
    --from-path=/import \
    --overwrite-destination=true

# Windows PowerShell version:
# docker run --rm `
#   -v "${PWD}\neo4jdata:/data" `
#   -v "${PWD}\neo4jlogs:/logs" `
#   -v "${PWD}\neo4jimport:/import" `
#   -v "${PWD}\neo4jplugins:/plugins" `
#   neo4j:5.26.19 `
#   neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true

# 5. Restart Neo4j
docker start neo4j-gai

# 6. Wait for startup (important!)
sleep 30  # Linux/Mac
# Start-Sleep -Seconds 30  # Windows PowerShell

# 7. Verify data loaded
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 \
  "MATCH (c:Chunk) RETURN count(c) AS chunks;"

# Should show ~150-200 chunks

# 8. Check vector index is online
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 \
  "SHOW VECTOR INDEXES YIELD name, state, populationPercent;"

# Should show: chunk_embedding_idx | ONLINE | 100.0
```

#### Option B: Ingest from Scratch (Slow)

**Warning**: GraphRAG ingestion is VERY slow because it:
- Calls Llama 3.2 multiple times per document for entity extraction
- Builds knowledge graph with entities and relationships
- Creates vector embeddings for each chunk

```bash
# Ingest documents
python main.py ingest \
  --llm-model llama3.2:latest \
  --retriever neo4j \
  --path "data/texts/*.txt"

# Progress will be shown with tqdm progress bar
# Each document takes 15-30 minutes
# Timeouts set to 1500 seconds (25 min) per document

# Check ingestion status
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 \
  "MATCH (d:Document) RETURN d.path, d.status;"
```

**Understanding ingestion status:**
- `PENDING`: Currently processing
- `OK`: Successfully ingested
- `FAILED`: Timeout or error occurred
- Missing: Not yet started

**Resuming failed ingestion:**
```bash
# Just run ingestion again
# Already ingested documents (status=OK) will be skipped
python main.py ingest \
  --llm-model llama3.2:latest \
  --retriever neo4j \
  --path "data/texts/*.txt"
```

#### Chatting with Neo4j GraphRAG

```bash
python main.py chat \
  --llm-model llama3.2:latest \
  --retriever neo4j \
  --top-k 5

# Type your questions
You: How does backpropagation work?
SALVIA: [streaming answer with citations]
```

#### Exploring the Knowledge Graph

Access Neo4j Browser at `http://localhost:7474`:

```cypher
// See all node types
CALL db.labels();

// Count nodes by type
MATCH (n:Concept) RETURN count(n);
MATCH (n:Method) RETURN count(n);
MATCH (n:Model) RETURN count(n);

// View sample entities
MATCH (c:Concept) RETURN c.name LIMIT 10;

// See relationships
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25;

// Find specific concept and its relationships
MATCH (c:Concept {name: "attention mechanism"})-[r]-(related)
RETURN c, r, related;
```

---

### 3. Web UI (Streamlit)

The Streamlit UI provides a clean chat interface with clickable citations.

#### Starting the UI

```bash
streamlit run app.py
```

Your browser should automatically open to `http://localhost:8501`

#### Using the UI

1. **Sidebar Configuration**:
   - **Retriever Type**: Choose `faiss` or `neo4j`
   - **LLM Model**: Usually `llama3.2:latest`
   - **Top K Results**: Number of sources to retrieve (1-10)

2. **Initialize Chat**:
   - Click the blue "Initialize Chat" button
   - Wait for "Chat initialized" success message
   - If FAISS: Must have run ingestion first
   - If Neo4j: Docker container must be running with loaded database

3. **Chatting**:
   - Type question in the input box at the bottom
   - Press Enter to send
   - Answer streams in real-time
   - Citations like `[s1]` are clickable badges
   - Click "View Sources" to see full source text

4. **Source Citations**:
   - Click on a `[s1]` badge to jump to that source
   - Sources shown in expandable section below answer
   - First 500 characters of each source displayed

5. **Clear History**:
   - Click "Clear Chat History" button in sidebar
   - Resets conversation (doesn't affect stored documents)

#### UI Troubleshooting

**"FAISS index not found":**
```bash
# Run ingestion first
python main.py ingest --llm-model llama3.2:latest --retriever faiss --path "data/texts/*.txt"
```

**"Could not connect to Neo4j":**
```bash
# Check Neo4j is running
docker ps --filter "name=neo4j"

# Start if not running
docker start neo4j-gai

# Wait 30 seconds
sleep 30
```

**"Chat not initialized":**
- Click the "Initialize Chat" button in the sidebar
- Check console for error messages

---

### 4. Evaluation

Run automated evaluation on both retrievers:

```bash
python main.py tests \
  --llm-model llama3.2:latest \
  --top-k 5
```

**What happens:**
1. Loads 5 test questions from `evaluation/test_questions.py`
2. Runs both FAISS and Neo4j retrievers
3. Generates answers for each question
4. Uses Llama 3.2 as judge to score: Faithfulness, Relevancy, Correctness, Context Relevance.
5. Saves results to `evaluation/results/`

**Results files:**
- `evaluation_results_faiss.json`
- `evaluation_results_graphrag.json`

**Interpreting results:**
- Scores are 0-1 (higher is better)
- 0.7+ is good
- 0.5-0.7 is acceptable
- <0.5 needs improvement

---

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── retriever.py          # FAISS & Neo4j retriever implementations
│   └── chat.py                # RAG chat logic with citation formatting
│
├── evaluation/
│   ├── test_questions.py      # Test dataset configuration
│   ├── evaluate_rag.py        # FAISS evaluation
│   ├── evaluate_graphrag.py   # Neo4j evaluation
│   └── results/               # JSON evaluation outputs
│
├── data/
│   ├── texts/                 # Source documents (.txt, .md)
│   └── vector_store_faiss.pkl # FAISS index (generated)
│
├── neo4jdata/                # Neo4j database files (generated)
├── neo4jimport/              # Place neo4j.dump here for restore
├── neo4jlogs/                # Neo4j logs
├── neo4jplugins/             # Neo4j plugins
│
├── dump/
│   └── neo4j.dump            # Pre-ingested Neo4j database
│
├── main.py                   # CLI entry point
├── app.py                    # Streamlit web UI
├── docker-compose.yml        # Neo4j container configuration
├── schema_gai.json           # Neo4j entity/relationship schema
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── README.md                 # This file
```

---

## Configuration

### Environment Variables (`.env`)

```env
# === Chunking ===
CHUNK_SIZE=512                    # Tokens per chunk
CHUNK_OVERLAP=128                 # Overlap between chunks

# === Models ===
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2    # Sentence transformer
CHAT_MODEL_NAME=llama3.2:latest           # Ollama model for chat
EVAL_MODEL_NAME=llama3.2:latest           # Ollama model for evaluation

# === Retrieval ===
SEMANTIC_WEIGHT=0.7               # Not currently used
KEYWORD_WEIGHT=0.3                # Not currently used
TOP_K=5                           # Default sources to retrieve

# === Neo4j ===
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass123
NEO4J_VECTOR_INDEX=chunk_embedding_idx
INGEST_TIMEOUT_SEC=1500           # Timeout per document (25 min)

# === FAISS ===
FAISS_INDEX_PATH=data/vector_store_faiss.pkl
```

### Command Line Arguments

Override `.env` values:

```bash
python main.py chat \
  --llm-model llama3.2:latest \
  --retriever faiss \
  --top-k 10 \
  --embedding-model all-mpnet-base-v2 \
  --faiss-path custom/path/to/index.pkl
```

### Neo4j Schema (`schema_gai.json`)

Defines entity types and relationships for GraphRAG:

**Node types:**
- `Concept`: ML concepts (e.g., "attention mechanism")
- `Method`: Techniques (e.g., "backpropagation")
- `Model`: ML models (e.g., "GPT-3")
- `Dataset`: Datasets (e.g., "ImageNet")
- `Metric`: Evaluation metrics (e.g., "accuracy")
- `Tool`: Tools/libraries (e.g., "PyTorch")
- `Paper`: Research papers

**Relationship types:**
- `DEFINES`, `USES`, `EVALUATED_BY`, `IMPROVES`, 
- `COMPARES_WITH`, `PART_OF`, `MENTIONS`

---

## Troubleshooting

### General Issues

**Problem**: `ModuleNotFoundError`
```bash
# Ensure virtual environment is activated
# You should see (.venv) in your prompt

# Reinstall requirements
pip install -r requirements.txt
```

**Problem**: `ollama: command not found`
```bash
# Install Ollama from https://ollama.com/download
# After install, verify:
ollama --version

# Pull model:
ollama pull llama3.2:latest
```

### FAISS Issues

**Problem**: `FAISS index is empty`
```bash
# Run ingestion first
python main.py ingest \
  --llm-model llama3.2:latest \
  --retriever faiss \
  --path "data/texts/*.txt"
```

### Neo4j Issues

**Problem**: `Could not connect to Neo4j database`
```bash
# Check Neo4j is running
docker ps --filter "name=neo4j"

# If not running:
docker start neo4j-gai

# Test connection
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 "RETURN 1;"
```

**Problem**: `Authentication failed`
```bash
# Password in .env might not match Neo4j

# Option 1: Use docker-compose password
# In .env:
NEO4J_PASSWORD=neo4jpass123

# Option 2: Reset Neo4j completely
docker stop neo4j-gai
docker rm neo4j-gai
rm -rf neo4jdata/*  # WARNING: Deletes all data
docker compose up -d
```

**Problem**: Vector index not found
```bash
# Check index status
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 \
  "SHOW VECTOR INDEXES;"

# If missing, reload dump or re-run ingestion
```

**Problem**: Ingestion timeout errors
- 25 minutes per document is normal
- Check Ollama is running: `ollama list`
- Check system resources (RAM, CPU)
- Increase timeout in `.env`: `INGEST_TIMEOUT_SEC=3000`

### Streamlit UI Issues

**Problem**: Port already in use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

**Problem**: Citations not clickable
- This is a known limitation - anchor links don't work perfectly in Streamlit
- Sources are still visible in "View Sources" expander

**Problem**: Streaming doesn't work
- Ensure Ollama is running
- Check console for errors
- Try command-line chat first to isolate issue

---

## Performance Notes

### Evaluation Results (from our testing)

| Metric | FAISS | Neo4j GraphRAG |
|--------|-------|----------------|
| Faithfulness | 0.74 | 0.68 |
| Relevancy | 0.82 | 0.78 |
| Correctness | 0.70 | 0.62 |
| Context Relevance | 0.62 | 0.60 |
| **Overall** | **0.72** | **0.67** |

---

---

## License

This project was created for the Generative AI course at TU Wien (WS2025).

---

## Contributors

- Aigerim Doskozha - 01529258
- Avelardo Ramirez - 12435655
- Brais Guardado Collazo - 12233000
- Ekaterina Grigorashchenko - 12432933
- Valeriia Avakian - 12432869

---

