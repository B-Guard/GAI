import os
import json
import glob
import asyncio
import re
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.llm import OllamaLLM as Neo4jOllamaLLM, VertexAILLM, OpenAILLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
import hashlib
from pathlib import Path


from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX", "chunk_embedding_idx")
TOP_K = int(os.getenv("TOP_K", "5"))

INGEST_TIMEOUT_SEC = int(os.getenv("INGEST_TIMEOUT_SEC", "1500"))

RETRIEVAL_QUERY = """
RETURN coalesce(node.text, '') AS text, score AS score
"""





class Neo4jGraphRAG:
    def __init__(self, llm_model_name: str, embedding_model_name: str, driver):
        self.driver = driver
        assert self._test_connection(), "Could not connect to Neo4j database."
        self.llm = Neo4jOllamaLLM(
            model_name=llm_model_name,
            model_params={"options": {"temperature": 0}, "format": "json"}
        )

        # self.llm = VertexAILLM()
        self.embedder = SentenceTransformerEmbeddings(model=embedding_model_name)
        STRICT_JSON_PROMPT = """
        You are an entity & relationship extraction system.

        Extract entities and relationships from the TEXT using the SCHEMA as guidance.
        Return ONLY a valid JSON object. No markdown. No explanations.

        Required JSON format (exact keys):
        {{
          "nodes": [
            {{"id": "string", "label": "string", "properties": {{"name": "string"}}}}
          ],
          "relationships": [
            {{"type": "string", "start_node_id": "string", "end_node_id": "string", "properties": {{}}}}
          ]
        }}

        Rules:
        - Every node must have a non-empty properties.name.
        - Use one of the schema node labels for label.
        - Use one of the schema relationship labels for type.
        - If nothing relevant is found, return {{"nodes": [], "relationships": []}}.

        SCHEMA:
        {schema}

        TEXT:
        {text}
        """

        import logging
        logging.basicConfig(level=logging.INFO)

        self.kg_builder = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embedder,
            neo4j_database="neo4j",
            schema=self._get_schema(),
            text_splitter=LangChainTextSplitterAdapter(
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
            ),
            from_pdf=False,
            prompt_template=STRICT_JSON_PROMPT,
            on_error="RAISE",
            perform_entity_resolution=False,  # speed/debug first; turn on later
        )

    def _test_connection(self):
        """
        Tests the connection to the Neo4j database.
        """
        import traceback
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                return record["test"] == 1

        except Exception as e:
                traceback.print_exc()

    def _get_schema(self):
        import json
        with open("schema_gai.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _doc_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _already_ingested(self, doc_hash: str) -> bool:
        cypher = """
        MATCH (d:Document {hash: $hash})
        RETURN coalesce(d.status, '') = 'OK' AS ok
        """
        with self.driver.session() as session:
            rec = session.run(cypher, hash=doc_hash).single()
            return bool(rec and rec["ok"])

    async def ingest_documents(self, pattern: str):
        paths = glob.glob(pattern, recursive=True)
        print(f"[INGEST] matched {len(paths)} files")

        for path in atqdm(paths):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            doc_hash = self._doc_hash(text)

            if self._already_ingested(doc_hash):
                print(f"[SKIP] already ingested: {path}")
                continue

            # Create/Update marker as PENDING
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (d:Document {hash: $hash})
                    SET d.path = $path,
                        d.createdAt = coalesce(d.createdAt, datetime()),
                        d.status = 'PENDING',
                        d.lastError = null
                    """,
                    hash=doc_hash,
                    path=str(path),
                )

            print(f"[RUN] {path}")

            try:
                # Prevent infinite hanging on Ollama
                await asyncio.wait_for(
                    self.kg_builder.run_async(
                        file_path=str(path),
                        text=text,
                        document_metadata={"path": str(path)}
                    ),
                timeout=INGEST_TIMEOUT_SEC
                )

                # Mark OK only after successful ingest
                with self.driver.session() as session:
                    session.run(
                        """
                        MATCH (d:Document {hash:$hash})
                        SET d.status='OK', d.ingestedAt=datetime()
                        """,
                        hash=doc_hash,
                    )

                print(f"[OK] ingested: {path}")

            except Exception as e:
                # Mark FAILED so reruns will try again
                with self.driver.session() as session:
                    session.run(
                        """
                        MATCH (d:Document {hash:$hash})
                        SET d.status='FAILED', d.failedAt=datetime(), d.lastError=$err
                        """,
                        hash=doc_hash,
                        err=str(e)[:2000],
                    )

                print(f"[ERROR] {path}: {e}")



async def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    rag = Neo4jGraphRAG(
        llm_model_name=CHAT_MODEL_NAME,
        embedding_model_name=EMBEDDING_MODEL,
        driver=driver,
    )

    # INGEST ONLY IF ENABLED
    DO_INGEST = os.getenv("DO_INGEST", "0") == "1"
    if DO_INGEST:
        await rag.ingest_documents(os.path.join("data", "texts", "*.txt"))
        print("Ingestion finished")
        with driver.session() as s:
            print("[CHECK] docs =", s.run("MATCH (d:Document) RETURN count(d) AS n").single()["n"])
            print("[CHECK] chunks =", s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"])
            print("[CHECK] embedded =",
                  s.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS n").single()["n"])
            print("[CHECK] vector indexes =", s.run(
                "SHOW VECTOR INDEXES YIELD name, state, populationPercent RETURN collect({name:name,state:state,pop:populationPercent}) AS x").single()[
                "x"])


    else:
        print("Skipping ingestion (DO_INGEST != 1)")

    # 2) GRAPH-BASED RETRIEVAL + CHAT
    embedder = SentenceTransformerEmbeddings(model=EMBEDDING_MODEL)

    retriever = VectorCypherRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        embedder=embedder,
        retrieval_query=RETRIEVAL_QUERY,
    )

    llm = Neo4jOllamaLLM(
        model_name=CHAT_MODEL_NAME,
        model_params={"format": "json", "options": {"temperature": 0}}
    )

    rag = GraphRAG(
        retriever=retriever,
        llm=llm,
    )

    print("Entering GraphRAG chat. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            break

        resp = rag.search(
            query_text=q,
            retriever_config={"top_k": TOP_K},
            return_context=True,
            response_fallback="I don't have information about that in your notes."
        )

        print("\nAI:", resp.answer)

        # show retrieved context
        print("\n--- Retrieved context ---")

        dump = resp.model_dump()
        items = dump.get("retriever_result", {}).get("items", [])

        for i, it in enumerate(items, 1):
            content = it.get("content", "")

            m = re.search(r"score=([0-9.]+)", content)
            score = float(m.group(1)) if m else None

            m2 = re.search(r'text="(.*)"\s+score=', content, re.DOTALL)
            text = m2.group(1) if m2 else content

            text = text.replace("\\n", "\n")

            print(f"\n[c{i}] score={score}")
            print(text[:1200])

    driver.close()



if __name__ == "__main__":
    asyncio.run(main())
