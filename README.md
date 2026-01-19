# Generative AI

## Developer's installation
Python 3.12, don't forget to create new virtual env. Then `pip install -r requirements.txt`.

You will also need Ollama installed and running. We have tested our setup with Llama 3.2, but the pipeline allows other models. Pull it with:
```bash
ollama pull llama3.2:latest
```

Copy the `.env.example` file to `.env` and adjust parameters as needed.
```bash
cp .env.example .env
```

Finally, you will need a running Neo4j instance, and of course, don't forget to add your credentials to the `.env` file.
```bash
NEO4J_USER=your_username
NEO4J_PASSWORD=your_password
```

## Usage

We provide two retrievers, backed up by different vector/graph stores: FAISS and Neo4j, respectively.
By default, Neo4j is used. If you want to use FAISS especifically, pass the `--retriever faiss` argument when running the scripts. This also makes the `--faiss-path` compulsory, which indicates where to store/load the FAISS index from.

When running the default Neo4j retriever, you can also overwrite the Neo4j connection parameters via command line arguments: `--neo4j-uri`, `--neo4j-user` and `--neo4j-password`.

Ingest documents (only needed once!):
```bash
python main.py ingest --path "<path pattern to text files>"
```
Don't forget the quotes around the path pattern! To ingest the provided sample files, the recommended pattern is `"data/texts/*.txt"`.

The ingestion of documents if using the Neo4j retriever will take quite some time. This is why we provide a dump of a pre-ingested database inside the `dump` folder. You can restore it in your Neo4j instance to skip the ingestion step.


Start chat:
```bash
python main.py chat
```

Run automatic evaluation (this will take a while):
```bash
python main.py tests
```
Results will be saved in JSON format inside the `evaluation/results` folder.

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
