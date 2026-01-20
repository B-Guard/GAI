```markdown
# GraphRAG with Neo4j (Updated) — Windows PowerShell Guide

This guide documents a **fully working setup** for running:

```

test/graphrag_w_neo4j_updated.py

````

using:

- Neo4j 5.x in Docker
- Ollama for local LLM inference
- A restored Neo4j dump (recommended, fastest path)

This README assumes you start with **only the initial project files**.

---

## Verified working versions

These versions are known to work together:

- **Docker Engine**: 27.5.1  
- **Docker Desktop**: 4.38.0  
- **Docker Compose**: v2.32.4  
- **Ollama**: 0.14.2  
- **Neo4j image**: `neo4j:5.26.19`  
- **Ollama model**: `llama3.2:latest`  

Verify tools:

```powershell
docker version
docker compose version
ollama --version
ollama list
````

---

## Python environment

### 1) Activate virtual environment

If `.venv` already exists (common when using PyCharm):

```powershell
.\.venv\Scripts\Activate.ps1
```

> If you try `python -m venv .venv` and get a permission error,
> the environment already exists — just activate it.

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Ollama setup

Pull the model used in `.env`:

```powershell
ollama pull llama3.2:latest
ollama list
```

---

## Environment variables

### 1) Create `.env`

```powershell
Copy-Item .env.example .env
```

### 2) Edit `.env`

Use these **known-good values**:

```env
CHUNK_SIZE=1200
CHUNK_OVERLAP=200

EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHAT_MODEL_NAME=llama3.2:latest
EVAL_MODEL_NAME=llama3.2:latest

SEMANTIC_WEIGHT=0.7
KEYWORD_WEIGHT=0.3

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass123
NEO4J_VECTOR_INDEX=chunk_embedding_idx

DO_INGEST=0
TOP_K=5
```

### 3) Check PowerShell overrides

PowerShell environment variables override `.env`.

Check:

```powershell
echo $env:NEO4J_URI
echo $env:NEO4J_USER
echo $env:NEO4J_PASSWORD
```

Fix if needed:

```powershell
$env:NEO4J_PASSWORD="neo4jpass123"
```

Or remove to rely on `.env`:

```powershell
Remove-Item Env:NEO4J_PASSWORD
```

---

## Start Neo4j with Docker Compose

```powershell
docker compose up -d
docker ps --filter "name=neo4j"
```

Neo4j should be available at:

* [http://localhost:7474](http://localhost:7474)
* bolt://localhost:7687

---

## Restore Neo4j database from dump (recommended)

### 1) Locate the dump

```powershell
Get-ChildItem -Recurse -Filter neo4j.dump | Select-Object FullName
```

Expected paths:

```
.\dump\neo4j.dump
.\neo4jimport\neo4j.dump
```

### 2) Ensure dump is in import directory

```powershell
Copy-Item ".\dump\neo4j.dump" ".\neo4jimport\neo4j.dump" -Force
```

### 3) Stop Neo4j

```powershell
docker stop neo4j-gai
```

### 4) Load the database (Neo4j 5 safe method)

```powershell
docker run --rm `
  -v "${PWD}\neo4jdata:/data" `
  -v "${PWD}\neo4jlogs:/logs" `
  -v "${PWD}\neo4jimport:/import" `
  -v "${PWD}\neo4jplugins:/plugins" `
  neo4j:5.26.19 `
  neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true
```

### 5) Restart Neo4j

```powershell
docker start neo4j-gai
Start-Sleep -Seconds 20
```

---

## Verify Neo4j state

```powershell
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 "RETURN 1;"
```

Check data:

```powershell
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 `
"MATCH (c:Chunk) RETURN count(c) AS chunks;"
```

Check vector index:

```powershell
docker exec neo4j-gai cypher-shell -u neo4j -p neo4jpass123 `
"SHOW VECTOR INDEXES YIELD name, state, populationPercent RETURN name, state, populationPercent;"
```

Expected:

```
chunk_embedding_idx | ONLINE | 100.0
```

---

## Run GraphRAG (updated)

```powershell
python .\test\graphrag_w_neo4j_updated.py
```

You should see:

```
Entering GraphRAG chat. Type 'exit' to quit.
```

Example:

```
You: explain backpropagation
AI: ...
```

---

## Common issues & fixes

### Authentication failure

* Password in `.env` does not match Neo4j internal password
* Fix by resetting password **before first start**, or reload dump again

### Connection refused

* Neo4j not fully started yet
* Wait 15–30 seconds after `docker start`

### Missing vector index

* Dump not loaded
* Index name mismatch (`NEO4J_VECTOR_INDEX`)
* Re-run dump restore

---

## Notes

* Dump restore is **much faster** than ingestion
* `DO_INGEST=0` is recommended once data exists
* Neo4j vector indexes **must be ONLINE** before GraphRAG runs

