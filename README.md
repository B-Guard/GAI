# Generative AI

## Developers installation
Python 3.12, don't forget to create new virtual env. `pip install -r requirements.txt`.

You will also need Ollama installed and running. You will need the following models if you don't have them installed:
```bash
ollama pull llama3.2:latest
```

Copy the `.env.example` file to `.env` and adjust parameters as needed.
```bash
cp .env.example .env
```

## Usage

Ingest documents (only needed once!):
```bash
python rag_hybrid.py ingest "data/*.txt"
```

Start chat:
```bash
python rag_hybrid.py chat
```

## Project Plan

RAG setup:
- Load and process files
- Perform queries and get answers with citation

GraphRAG:
- GraphRAG implementation that works alongside existing infrastructure
- Explore capabilities and shortcomings of the current approach, try to improve. Document all your findings!

Test
- Evaluate the system according to the chosen metrics

UI
- tbd


## Backlog

### RAG setup
Basic RAG setup is done. Ingest files by running `python rag.py ingest "<pattern>"`, then chat with `python rag.py chat`. Pattern can be e.g. `data/*.txt`.

### GraphRAG


### Test


### UI
