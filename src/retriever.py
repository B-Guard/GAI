import os
import re
import glob
import pickle
import hashlib
import asyncio
import numpy as np
from typing import Any
from abc import abstractmethod
from tqdm.asyncio import tqdm as atqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
import faiss
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = os.path.join("data", "vector_store_faiss.pkl")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX", "chunk_embedding_idx")

INGEST_TIMEOUT_SEC = int(os.getenv("INGEST_TIMEOUT_SEC", "1500"))

RETRIEVAL_QUERY = """
WITH node, score
OPTIONAL MATCH (node)-[r]-(nbr)
WITH node, score, r, nbr,
     coalesce(
        nbr.name,
        nbr.title,
        nbr.path,
        CASE WHEN nbr.text IS NOT NULL THEN left(nbr.text, 160) END,
        CASE WHEN nbr.content IS NOT NULL THEN left(nbr.content, 160) END
     ) AS surface
WHERE surface IS NOT NULL AND trim(surface) <> ""
WITH node, score,
     collect(DISTINCT type(r) + " -> " + surface)[0..12] AS neigh
RETURN
  ("Chunk:\n" + coalesce(node.text, node.content, '') +
   "\n\nGraph neighbors:\n" +
   reduce(s = "", x IN neigh | s + "- " + x + "\n")
  ) AS text,
  score AS score
"""


class VectorStore:
    @abstractmethod
    async def ingest(self, file_pattern: str):
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, Any]]:
        pass


class FaissVectorStore(VectorStore):
    def __init__(self, model_name: str, vector_store_path: str):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.embeddings = None
        self.vector_store_path = vector_store_path
        self.chunk_hashes = set()  # Track hashes of ingested chunks
        # Try to load existing vector store if it exists
        if os.path.exists(self.vector_store_path):
            try:
                self.load(self.vector_store_path)
                print(f"Loaded existing vector store from {self.vector_store_path}")
            except Exception as e:
                print(f"Warning: Could not load vector store: {e}")
                raise e
        else:
            print("Starting a fresh FAISS vector store.")

    async def ingest(self, file_pattern: str):
        files = glob.glob(file_pattern, recursive=True)
        new_texts = []
        new_embeddings = []
        new_hashes = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                ext = os.path.splitext(file)[1].lower()
                if ext == ".md":
                    splitter = MarkdownTextSplitter(
                        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                    )
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        separators=[".", "\n"],
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    if chunk_hash in self.chunk_hashes:
                        continue  # Skip already ingested chunk
                    new_texts.append(chunk)
                    emb = self.model.encode(chunk, convert_to_numpy=True)
                    new_embeddings.append(emb)
                    new_hashes.append(chunk_hash)
        if not new_embeddings:
            print("No new chunks to ingest (all already ingested or no files matched).")
            return
        new_embeddings = np.stack(new_embeddings).astype("float32")
        faiss.normalize_L2(new_embeddings)
        # If index exists, add to it; else, create new
        if (
            self.index is not None
            and self.embeddings is not None
            and len(self.texts) > 0
        ):
            # Concatenate texts and embeddings
            self.texts.extend(new_texts)
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.index.add(new_embeddings)
            self.chunk_hashes.update(new_hashes)
        else:
            dim = new_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(new_embeddings)
            self.texts = new_texts
            self.embeddings = new_embeddings
            self.chunk_hashes = set(new_hashes)
        print(
            f"Ingested {len(new_texts)} new chunks from {len(files)} files into the FAISS vector store."
        )
        self.save(self.vector_store_path)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self.index is None:
            raise ValueError("FAISS index is empty. Ingest data before retrieval!")
        # Encode and normalize the query
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        # Search FAISS index
        scores, indices = self.index.search(q_emb, top_k)
        # Return (text, score) pairs
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                if score > 0.2:
                    results.append((self.texts[idx], score))
        return results

    def save(self, path: str = VECTOR_STORE_PATH):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "index": self.index,
                    "texts": self.texts,
                    "embeddings": self.embeddings,
                    "chunk_hashes": list(self.chunk_hashes),
                },
                f,
            )

    def load(self, path: str = VECTOR_STORE_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.texts = data["texts"]
            self.embeddings = data["embeddings"]
            self.chunk_hashes = set(data.get("chunk_hashes", []))


class Neo4jRetriever(VectorStore):
    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        uri: str,
        user: str,
        password: str,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        assert self._test_connection(), "Could not connect to Neo4j database. Is it running?"

        self.embedder = SentenceTransformerEmbeddings(model=embedding_model_name)
        self.kg_builder = SimpleKGPipeline(
            llm=OllamaLLM(model_name=llm_model_name, model_params={"temperature": 0}),
            driver=self.driver,
            embedder=self.embedder,
            schema=self._get_schema(),
            text_splitter=LangChainTextSplitterAdapter(
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
            ),
            from_pdf=False,
        )
        self.retriever = VectorCypherRetriever(
            driver=self.driver,
            embedder=self.embedder,
            index_name=INDEX_NAME,
            retrieval_query=RETRIEVAL_QUERY,
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

    async def ingest(self, pattern: str):
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
                        document_metadata={"path": str(path)},
                    ),
                    timeout=INGEST_TIMEOUT_SEC,
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

    def retrieve(self, query: str, top_k: int) -> tuple[list[tuple[str, dict]], dict]:
        results = self.retriever.search(
            query_text=query,
            top_k=top_k,
        )
        # print(results.metadata) # This attribute contains the query vecto and the retriever name
        context = []
        for record in results.items:
            m = re.search(r"score=([0-9.]+)", record.content)
            score = float(m.group(1)) if m else None

            m2 = re.search(r'text="(.*)"\s+score=', record.content, re.DOTALL)
            text = m2.group(1) if m2 else record.content
            if score is None or score > 0.2:
                context.append((text.replace("\\n", "\n"), score))
        return context
