import os
import json
import glob
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever, Text2CypherRetriever
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


class Neo4jGraphRAG:
    def __init__(self, llm_model_name: str, embedding_model_name: str, driver):
        self.driver = driver
        assert self._test_connection(), "Could not connect to Neo4j database."
        self.llm = Neo4jOllamaLLM(model_name=llm_model_name)
        # self.llm = VertexAILLM()
        self.embedder = SentenceTransformerEmbeddings(model=embedding_model_name)
        self.kg_builder = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embedder,
            schema=self._get_schema(),
            text_splitter=LangChainTextSplitterAdapter(
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=2000, chunk_overlap=150
                )
            ),
            from_pdf=False,
        )

    def _test_connection(self):
        """
        Tests the connection to the Neo4j database.
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                return record["test"] == 1
        except Exception:
            return False

    def _get_schema(self) -> str:
        return "FREE"

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
                    self.kg_builder.run_async(text=text),
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


class Neo4jVectorStore:
    def __init__(self, driver, index_name):
        self.driver = driver
        self.index_name = index_name

    def retrieve(self, query_vec, top_k=5):
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $qvec)
        YIELD node, score
        RETURN node.text AS text, score
        ORDER BY score DESC
        """
        with self.driver.session() as session:
            rows = session.run(
                cypher,
                index_name=self.index_name,
                k=top_k,
                qvec=query_vec,
            )
            return [(r["text"], r["score"]) for r in rows if r["text"]]

class GraphRAGChat:
    def __init__(self, vector_store, embedder, llm, top_k=5):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k

    def ask(self, question: str):
        qvec = self.embedder.encode(question).tolist()
        results = self.vector_store.retrieve(qvec, self.top_k)

        if not results:
            print("AI: I don't have information about that in your notes.")
            return

        sources = [f"[s{i}] {txt}" for i, (txt, _) in enumerate(results, 1)]
        context = "\n---\n".join(sources)

        prompt = (
            "You are an AI tutor.\n"
            "CRITICAL RULES:\n"
            "1) Answer ONLY using the Sources.\n"
            "2) If the Sources do not contain the answer, say: "
            "'I don't have information about that in your notes.'\n"
            "3) Cite sources with [sX].\n\n"
            f"Sources:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        print("AI: ", end="", flush=True)
        for chunk in self.llm.stream(prompt):
            print(chunk, end="", flush=True)

        print("\n\nSources:")
        for i, (txt, score) in enumerate(results, 1):
            print(f"(score: {score:.4f}) [s{i}] {txt}".replace("\n", " "))


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
    else:
        print("Skipping ingestion (DO_INGEST != 1)")


    # 2) CHAT
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    llm = OllamaLLM(model=CHAT_MODEL_NAME, temperature=0)

    store = Neo4jVectorStore(driver, INDEX_NAME)
    chat = GraphRAGChat(store, embedder, llm, top_k=TOP_K)

    print("Entering GraphRAG chat. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() == "exit":
            break
        chat.ask(q)

    driver.close()



if __name__ == "__main__":
    asyncio.run(main())
