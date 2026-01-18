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
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None
        self.bm25 = None
        self.texts = []
        self.embeddings = None
        self.tokenized_texts = []

    def _tokenize(self, text: str):
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

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
                        chunk_size=CHUNK_SIZE, 
                        chunk_overlap=CHUNK_OVERLAP
                    )
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", ". ", " "],
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
        
        # Build FAISS index
        embeddings = np.stack(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        
        # Build BM25 index
        self.tokenized_texts = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_texts)
        
        self.texts = texts
        self.embeddings = embeddings
        
        print(
            f"Ingested {len(texts)} chunks from {len(files)} files into the hybrid vector store."
        )

    def retrieve(self, query: str, top_k: int = 10):
        if self.faiss_index is None or self.bm25 is None:
            return []
        
        # FAISS semantic search
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        
        k_candidates = min(top_k * 3, len(self.texts))
        faiss_scores, faiss_indices = self.faiss_index.search(q_emb, k_candidates)
        
        # Normalize FAISS scores to [0, 1]
        faiss_scores = faiss_scores[0]
        faiss_scores = (faiss_scores + 1) / 2
        
        # BM25 keyword search
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Combine scores
        combined_scores = {}
        for rank, idx in enumerate(faiss_indices[0]):
            if idx < 0:
                continue
            
            text = self.texts[idx]
            semantic_score = faiss_scores[rank]
            keyword_score = bm25_scores[idx]
            combined = (SEMANTIC_WEIGHT * semantic_score + 
                       KEYWORD_WEIGHT * keyword_score)
            
            combined_scores[idx] = (text, combined)
        
        sorted_results = sorted(
            combined_scores.values(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Filter low scores
        results = [(text, score) for text, score in sorted_results if score > 0.3]
        return results[:top_k]

    def save(self, path: str = VECTOR_STORE_PATH):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "faiss_index": self.faiss_index,
                    "bm25": self.bm25,
                    "texts": self.texts,
                    "embeddings": self.embeddings,
                    "tokenized_texts": self.tokenized_texts,
                },
                f,
            )

    def load(self, path: str = VECTOR_STORE_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.faiss_index = data["faiss_index"]
            self.bm25 = data["bm25"]
            self.texts = data["texts"]
            self.embeddings = data["embeddings"]
            self.tokenized_texts = data["tokenized_texts"]


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
        
        sources = [f"[s{idx}] " + text for idx, (text, score) in enumerate(results, 1)]
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
        for i, (src, (_, score)) in enumerate(zip(sources, results), 1):
            print(f"(score: {score:.4f}) {src}".replace("\n", " "))

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
