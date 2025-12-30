import os
import glob
import pickle
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaLLM
from langchain.memory import ConversationBufferMemory
import faiss

VECTOR_STORE_PATH = os.path.join("data", "vector_store_faiss.pkl")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHAT_MODEL_NAME = "llama3.2:latest"


def show_info(vector_store_path=VECTOR_STORE_PATH):
    print("--- Vector Store Info ---")
    try:
        with open(vector_store_path, "rb") as f:
            data = pickle.load(f)
        embeddings = data["embeddings"]
        print(f"Number of vectors: {embeddings.shape[0]}")
        print(f"Vector dimension: {embeddings.shape[1]}")
    except Exception as e:
        print(f"Could not load vector store: {e}")
    print("\n--- Available Ollama Models ---")
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags")
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            for m in models:
                print(f"- {m.get('name')}")
        else:
            print("Could not fetch Ollama models (is Ollama running?)")
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")


class FaissVectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.embeddings = None

    def ingest(self, file_pattern: str):
        files = glob.glob(file_pattern)
        texts = []
        embeddings = []
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
                    texts.append(chunk)
                    emb = self.model.encode(chunk, convert_to_numpy=True)
                    embeddings.append(emb)
        if not embeddings:
            print("No files matched.")
            return
        embeddings = np.stack(embeddings).astype("float32")
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.texts = texts
        self.embeddings = embeddings
        print(
            f"Ingested {len(texts)} chunks from {len(files)} files into the FAISS vector store."
        )

    def retrieve(self, query: str, top_k: int = 5):
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
                },
                f,
            )

    def load(self, path: str = VECTOR_STORE_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.texts = data["texts"]
            self.embeddings = data["embeddings"]


class RAGChat:
    def __init__(self, faiss_store: FaissVectorStore, top_k):
        self.faiss_store = faiss_store
        self.top_k = top_k
        self.llm = OllamaLLM(model=CHAT_MODEL_NAME, temperature=0)

    def _process_question(self, question: str):
        # Retrieve top chunks and scores using only FAISS
        results = self.faiss_store.retrieve(question, self.top_k)
        if len(results) == 0:
            print("It seems I am not capable of answering that question.")
            return
        sources = [f"[s{idx}] " + text for idx, (text, score) in enumerate(results, 1)]
        context = "\n---\n".join(sources)
        prompt = (
            "You are a helpful assistant. Use the following context to answer the question.\n"
            "When you make a factual statement supported by a source, append the corresponding [sX] tag at the end of the sentence.\n"
            f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        print("AI: ", end="", flush=True)
        for chunk in self.llm.stream(prompt):
            print(chunk, end="", flush=True)
        print()  # Newline after completion
        print("\nSources:")
        for i, (src, (_, score)) in enumerate(zip(sources, results), 1):
            print(f"(score: {score:.4f}) {src}".replace("\n", " "))

    def chat(self):
        print(
            f"Entering chat mode. Type 'exit' to quit. Showing top {self.top_k} sources."
        )
        while True:
            question = input("\nYou: ")
            if question.strip().lower() == "exit":
                print("Exiting chat mode.")
                break
            self._process_question(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple RAG pipeline: ingest files or chat."
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest files into the vector store."
    )
    ingest_parser.add_argument(
        "pattern", type=str, help="File name or glob pattern to ingest."
    )
    ingest_parser.add_argument(
        "--vector_store",
        type=str,
        default=VECTOR_STORE_PATH,
        help="Path to vector store file.",
    )
    ingest_parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="SentenceTransformer model name.",
    )

    chat_parser = subparsers.add_parser("chat", help="Enter chat mode with RAG.")
    chat_parser.add_argument(
        "--vector_store",
        type=str,
        default=VECTOR_STORE_PATH,
        help="Path to vector store file.",
    )
    chat_parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="SentenceTransformer model name.",
    )
    chat_parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top documents to retrieve for each question.",
    )

    show_parser = subparsers.add_parser(
        "show", help="Show vector store and Ollama model info."
    )
    show_parser.add_argument(
        "--vector_store",
        type=str,
        default=VECTOR_STORE_PATH,
        help="Path to vector store file.",
    )

    args = parser.parse_args()
    if args.command == "ingest":
        store = FaissVectorStore(model_name=args.model)
        store.ingest(args.pattern)
        store.save(args.vector_store)
    elif args.command == "chat":
        store = FaissVectorStore(model_name=args.model)
        store.load(args.vector_store)
        chat = RAGChat(store, top_k=args.top_k)
        chat.chat()
    elif args.command == "show":
        show_info(vector_store_path=args.vector_store)
    else:
        parser.print_help()
