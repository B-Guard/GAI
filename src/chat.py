import os
from langchain_ollama import OllamaLLM

from src.retriever import VectorStore

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT"))


class RAGChat:
    def __init__(self, vector_store: VectorStore, top_k):
        self.vector_store = vector_store
        self.top_k = top_k
        self.llm = OllamaLLM(model=CHAT_MODEL_NAME, temperature=0)

    def _process_question(self, question: str, stream: bool = True):
        results = self.vector_store.retrieve(question, self.top_k)
        if len(results) == 0:
            print("It seems I am not capable of answering that question.")
            return
        sources = [f"[s{idx}] " + text for idx, (text, score) in enumerate(results, 1)]
        context = "\n---\n".join(sources)
        prompt = (
            "Your name is SALVIA, a helpful AI assistant. Use the following Sources to answer the question.\n"
            "When you make a factual statement supported by a source, append the corresponding [sX] tag at the end of the sentence.\n"
            f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        print("AI: ", end="", flush=True)
        if stream:
            for chunk in self.llm.stream(prompt):
                print(chunk, end="", flush=True)
        else:
            answer = self.llm.invoke(prompt)
            print(answer, end="", flush=True)
        print()  # Newline after completion
        print("\nSources:")
        for i, (src, (_, score)) in enumerate(zip(sources, results), 1):
            print(f"(score: {score:.4f}) {src}".replace("\n", " "))

    def chat_loop(self):
        print(
            f"Entering chat mode. Type 'exit' to quit. Showing top {self.top_k} sources."
        )
        while True:
            question = input("\nYou: ")
            # question = "What stands for GPT?"
            if question.strip().lower() == "exit":
                print("Exiting chat mode.")
                break
            self._process_question(question)

    