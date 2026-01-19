import os
import asyncio
import argparse
import subprocess
from dotenv import load_dotenv
from src.chat import RAGChat
from src.retriever import FaissVectorStore, Neo4jRetriever

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "vector_store_faiss")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_available_ollamas():
    """
    Lists available models in Ollama by calling 'ollama list' via subprocess.
    Returns a list of model names.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        # Skip header line, parse model names
        models = [line.split()[0] for line in lines[1:] if line]
        return models
    except Exception as e:
        print(f"Error listing Ollama models: {e}")
        return []


async def main():
    parser = argparse.ArgumentParser(
        description="SALVIA - Slow And Low-Value Intelligent Agent"
    )
    parser.add_argument(
        "mode",
        choices=["ingest", "chat", "test", "help"],
        help="Mode to run the script: ingest documents, chat, or perform automatic tests",
    )
    parser.add_argument("--llm-model", help="Ollama LLM model to use")
    parser.add_argument(
        "--embedding-model", default=EMBEDDING_MODEL_NAME, help="Embedding model to use"
    )
    parser.add_argument(
        "--retriever",
        choices=["neo4j", "faiss"],
        default="neo4j",
        help="Retriever type to use",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top documents to retrieve"
    )
    parser.add_argument("--path", type=str, help="Path to documents for ingestion")
    parser.add_argument(
        "--faiss-path",
        type=str,
        default=FAISS_INDEX_PATH,
        help="Path to store/load Faiss index",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=NEO4J_URI,
        help="Neo4j database URI",
    )
    parser.add_argument(
        "--neo4j-user", type=str, default=NEO4J_USER, help="Neo4j database user"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=NEO4J_PASSWORD,
        help="Neo4j database password",
    )
    args = parser.parse_args()

    if args.retriever == "faiss":
        retriever = FaissVectorStore(
            model_name=args.embedding_model, vector_store_path=args.faiss_path
        )
    else:
        retriever = Neo4jRetriever(
            args.llm_model,
            args.embedding_model,
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
        )

    chat = RAGChat(retriever, args.top_k)

    if args.mode == "ingest":
        if not args.path:
            print("Please provide a path to documents for ingestion using --path")
            return
        print("Ingesting documents...")
        await retriever.ingest(args.path)
    elif args.mode == "chat":
        print("Starting chat...")
        chat.chat_loop()
    elif args.mode == "test":
        print("Performing automatic tests...")
        # TODO test logic
    elif args.mode == "help":
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
