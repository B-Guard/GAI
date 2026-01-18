import os
import json
import glob
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever, Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.llm import OllamaLLM, VertexAILLM, OpenAILLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)

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


class Neo4jGraphRAG:
    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        assert self._test_connection(), "Could not connect to Neo4j database."
        self.llm = OllamaLLM(model_name=llm_model_name)
        # self.llm = VertexAILLM()
        self.embedder = SentenceTransformerEmbeddings(model=embedding_model_name)
        self.kg_builder = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embedder,
            schema=self._get_schema(),
            text_splitter=LangChainTextSplitterAdapter(
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100
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

    async def ingest_documents(self, pattern: str):
        """
        Ingests a document into the Neo4j graph database by extracting entities and relationships.
        """
        for path in atqdm(glob.glob(pattern, recursive=True)):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            await self.kg_builder.run_async(text=text)


async def main():
    rag = Neo4jGraphRAG(
        llm_model_name=CHAT_MODEL_NAME,
        embedding_model_name=EMBEDDING_MODEL,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
    )

    await rag.ingest_documents(os.path.join("data", "texts", "Large Language Models explained briefly.txt"))


if __name__ == "__main__":
    asyncio.run(main())
