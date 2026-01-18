import os
import glob
import pickle
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_ollama import OllamaLLM
import faiss
from rank_bm25 import BM25Okapi
import re

from dotenv import load_dotenv
load_dotenv()

VECTOR_STORE_PATH = os.path.join("data", "vector_store_hybrid.pkl")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT"))


def show_info(vector_store_path=VECTOR_STORE_PATH):
    print("--- Vector Store Info ---")
    try:
        with open(vector_store_path, "rb") as f:
            data = pickle.load(f)
        embeddings = data["embeddings"]
        texts = data["texts"]
        print(f"Number of vectors: {embeddings.shape[0]}")
        print(f"Vector dimension: {embeddings.shape[1]}")
        print(f"Number of text chunks: {len(texts)}")
        if len(texts) > 0:
            avg_chunk_length = np.mean([len(t) for t in texts])
            print(f"Average chunk length: {avg_chunk_length:.0f} characters")
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


class HybridVectorStore:
    """
    Hybrid retrieval store with:
    - Better chunking (done during ingest via improved separators)
    - Metadata (tracks source filename for each chunk)
    - Improved tokenization for BM25
    - Larger FAISS candidate pool for better recall
    - Reciprocal Rank Fusion (RRF) to combine FAISS + BM25 robustly
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None
        self.bm25 = None

        # Each entry: {"text": str, "source": str}
        self.docs = []
        self.embeddings = None
        self.tokenized_texts = []

    def _tokenize(self, text: str):
        # Better BM25 tokenization: keeps numbers, splits paths/snake_case, filters tiny tokens
        text = text.lower()
        text = re.sub(r"[_/\\\-]+", " ", text)         # split snake_case, paths, hyphens
        tokens = re.findall(r"[a-z0-9]+", text)        # keep numbers
        tokens = [t for t in tokens if len(t) >= 2]    # remove very short junk
        return tokens

    def ingest(self, file_pattern: str):
        files = glob.glob(file_pattern)
        docs = []
        embeddings = []

        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                ext = os.path.splitext(file)[1].lower()

                if ext == ".md":
                    splitter = MarkdownTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                else:
                    # Improved separators => better chunks => better retrieval
                    splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " "],
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )

                chunks = splitter.split_text(text)
                src = os.path.basename(file)

                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    docs.append({"text": chunk, "source": src})
                    emb = self.model.encode(chunk, convert_to_numpy=True)
                    embeddings.append(emb)

        if not embeddings:
            print("No files matched.")
            return

        # Build FAISS index (cosine via normalized inner product)
        embeddings = np.stack(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

        # Build BM25 index
        self.tokenized_texts = [self._tokenize(d["text"]) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_texts)

        self.docs = docs
        self.embeddings = embeddings

        print(f"Ingested {len(docs)} chunks from {len(files)} files into the hybrid vector store.")

    def _rrf_fuse(self, faiss_ranked_indices, bm25_ranked_indices, k: int = 60):
        scores = {}

        for rank, idx in enumerate(faiss_ranked_indices, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

        for rank, idx in enumerate(bm25_ranked_indices, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

        fused_sorted = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return fused_sorted, scores

    def retrieve(self, query: str, top_k: int = 10):
        if self.faiss_index is None or self.bm25 is None or not self.docs:
            return []

        n_docs = len(self.docs)

        # --- FAISS semantic search
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)

        # Larger candidate pool => better recall before fusion/rerank
        k_candidates = min(max(top_k * 10, 50), n_docs)
        faiss_scores, faiss_indices = self.faiss_index.search(q_emb, k_candidates)
        faiss_ranked = [idx for idx in faiss_indices[0] if idx >= 0]

        # --- BM25 keyword search
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = list(np.argsort(bm25_scores)[::-1][:k_candidates])

        fused_indices, fused_scores = self._rrf_fuse(faiss_ranked, bm25_ranked, k=60)

        # normalize fused score to 0..1 for display (optional but nice)
        max_fused = max(fused_scores.values()) if fused_scores else 1.0

        results = []
        for idx in fused_indices[:top_k]:
            doc = self.docs[idx]
            score = fused_scores[idx] / max_fused
            results.append((doc, float(score)))

        return results

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "docs": self.docs,
                    "embeddings": self.embeddings,
                    "tokenized_texts": self.tokenized_texts,
                    "faiss_index": self.faiss_index,
                    "bm25": self.bm25,
                },
                f,
            )

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.docs = data["docs"]
        self.embeddings = data["embeddings"]
        self.tokenized_texts = data["tokenized_texts"]
        self.faiss_index = data["faiss_index"]
        self.bm25 = data["bm25"]



class RAGChat:
    def __init__(self, vector_store: HybridVectorStore, top_k):
        self.vector_store = vector_store
        self.top_k = top_k
        self.llm = OllamaLLM(model=CHAT_MODEL_NAME, temperature=0)

    def _process_question(self, question: str):
        results = self.vector_store.retrieve(question, self.top_k)
        
        if len(results) == 0:
            print("It seems I am not capable of answering that question.")
            return

        sources = []
        for idx, (doc, score) in enumerate(results, 1):
            # Show filename for debugging / trust
            sources.append(f"[s{idx}] (source: {doc['source']})\n{doc['text']}")
        context = "\n---\n".join(sources)

        prompt = (
            "You are an AI tutor helping students understand machine learning concepts.\n"
            "CRITICAL RULES:\n"
            "1. Answer ONLY using information from the sources below\n"
            "2. If sources don't contain the answer, say 'I don't have information about that in your notes'\n"
            "3. When making a factual claim, cite the source with [sX] at the end of the sentence\n"
            "4. Synthesize across sources when relevant but stay grounded in what they say\n"
            "5. Do NOT add information from your training data\n\n"
            f"Sources:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (cite sources with [sX]):"
        )
        
        print("AI: ", end="", flush=True)
        for chunk in self.llm.stream(prompt):
            print(chunk, end="", flush=True)
        print()

        print("\nSources:")
        for i, ((doc, score)) in enumerate(results, 1):
            print(f"(score: {score:.4f}) [s{i}] (source: {doc['source']}) {doc['text']}".replace("\n", " "))

    def chat(self):
        print(f"Entering chat mode. Type 'exit' to quit. Showing top {self.top_k} sources.")
        while True:
            question = input("\nYou: ")
            if question.strip().lower() == "exit":
                print("Exiting chat mode.")
                break
            self._process_question(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Improved RAG pipeline: hybrid search with better grounding."
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
        store = HybridVectorStore(model_name=args.model)
        store.ingest(args.pattern)
        store.save(args.vector_store)
    elif args.command == "chat":
        store = HybridVectorStore(model_name=args.model)
        store.load(args.vector_store)
        chat = RAGChat(store, top_k=args.top_k)
        chat.chat()
    elif args.command == "show":
        show_info(vector_store_path=args.vector_store)
    else:
        parser.print_help()
