import os
from typing import Any, Generator
from time import sleep
from random import choice
from langchain_ollama import OllamaLLM

from src.retriever import VectorStore

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT"))


class RAGChat:
    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = CHAT_MODEL_NAME,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.llm = OllamaLLM(model=model_name, temperature=0)

    def _format_sources(
        self, sources_and_scores: list[tuple[str, tuple[str, Any]]]
    ) -> str:
        # return [
        #     f"(score: {score:.4f}) {src}".replace("\n", " ")
        #     for src, (_, score) in sources_and_scores
        # ]
        return [f"{src}".replace("\n", " ") for src, (_, score) in sources_and_scores]

    def __fake_streaming(
        self, text: str, delay: float = 0.01
    ) -> Generator[tuple[str, None], None, None]:
        """A generator that yields text one character at a time, simulating streaming."""
        for char in text:
            yield char, None
            sleep(delay)

    def _prepare_prompt_and_sources(
        self, question: str
    ) -> tuple[str | None, str | None, list[tuple[str, tuple[str, Any]]]]:
        results = self.vector_store.retrieve(question, self.top_k)
        if len(results) == 0:
            messages = [
                "It seems I'm not good enough to answer that question.",
                "Great question! But you are asking the wrong guy for that, sorry.",
                "I wish I could help you with that, but that level of wisdom is beyond me.",
                "I'm afraid I don't have the answer to that one. Maybe try asking a human?",
            ]
            message = choice(messages)
            return None, message, []
        sources = [f"[s{idx}] " + text for idx, (text, score) in enumerate(results, 1)]
        context = "\n---\n".join(sources)
        prompt = (
            "Your name is SALVIA, a helpful AI assistant. Use the following Sources to answer the question.\n"
            "When you make a factual statement supported by a source, append the corresponding [sX] tag at the end of the sentence.\n"
            f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return prompt, None, list(zip(sources, results))

    def stream_answer(
        self, question: str
    ) -> Generator[tuple[str | None, list[str] | None], None, None]:
        """
        Generator that yields answer chunks for the given question.
        Yields (chunk, None) for answer, then (None, sources) at the end
        """
        prompt, message, sources_and_scores = self._prepare_prompt_and_sources(question)
        if prompt is None:
            yield from self.__fake_streaming(message)
            yield None, []
            return
        for chunk in self.llm.stream(prompt):
            yield chunk, None
        # At the end, yield sources with scores
        sources = self._format_sources(sources_and_scores)
        yield None, sources

    def answer(self, question: str) -> tuple[str, list[str]]:
        """
        Returns the full answer and sources for the given question (non-streaming).
        """
        prompt, message, sources_and_scores = self._prepare_prompt_and_sources(question)
        if prompt is None:
            return message, []
        answer = self.llm.invoke(prompt)
        sources = self._format_sources(sources_and_scores)
        return answer, sources

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
            for chunk, sources in self.stream_answer(question):
                if chunk is not None:
                    print(chunk, end="", flush=True)
                else:
                    print()  # New line after answer
                    if sources:
                        print("Sources:")
                        for source in sources:
                            print(source)
                    break
