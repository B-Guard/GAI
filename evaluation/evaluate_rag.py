import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

from src.chat import RAGChat
from evaluation.test_questions import get_test_dataset, load_ground_truth_from_file

load_dotenv()

VECTOR_STORE_PATH = os.path.join("data", "vector_store_faiss.pkl")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "llama3.2:latest")

# Ensure results directory exists
RESULTS_DIR = os.path.join("evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TEST_RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation_results_faiss.json")


def create_test_dataset():
    """Create test dataset with questions based on shared config."""
    test_data_raw = get_test_dataset()

    test_data = {
        "question": [],
        "ground_truth": [],
        "source_file": [],
    }

    for question, gt_file, src_file in zip(
            test_data_raw["question"],
            test_data_raw["ground_truth_file"],
            test_data_raw["source_file"]
    ):
        test_data["question"].append(question)
        ground_truth = load_ground_truth_from_file(gt_file)
        test_data["ground_truth"].append(ground_truth)
        test_data["source_file"].append(src_file)

    return test_data


def evaluate_with_llm(llm, metric_name, question, answer, contexts, ground_truth):
    """Generic LLM-based evaluation for different metrics."""

    if metric_name == "faithfulness":
        context_text = "\n\n".join(contexts)
        prompt = f"""You are an evaluation judge. Determine if the ANSWER is faithful to the CONTEXTS.

CONTEXTS:
{context_text}

ANSWER:
{answer}

Rate faithfulness (0-10):
- 0 = Makes up information
- 5 = Partially faithful
- 10 = Completely faithful

Format: SCORE: X
EXPLANATION: [brief explanation]"""

    elif metric_name == "relevancy":
        prompt = f"""You are an evaluation judge. Determine if the ANSWER is relevant to the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Rate relevancy (0-10):
- 0 = Completely irrelevant
- 5 = Partially relevant
- 10 = Highly relevant

Format: SCORE: X
EXPLANATION: [brief explanation]"""

    elif metric_name == "correctness":
        prompt = f"""You are an evaluation judge. Determine if the ANSWER is correct compared to GROUND TRUTH.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

ANSWER:
{answer}

Rate correctness (0-10):
- 0 = Completely incorrect
- 5 = Partially correct
- 10 = Completely correct

Format: SCORE: X
EXPLANATION: [brief explanation]"""

    elif metric_name == "context_relevance":
        context_preview = "\n\n".join([f"Context {i + 1}: {ctx[:200]}..." for i, ctx in enumerate(contexts)])
        prompt = f"""You are an evaluation judge. Determine if the CONTEXTS are relevant to the QUESTION.

QUESTION:
{question}

CONTEXTS:
{context_preview}

Rate context relevance (0-10):
- 0 = Completely irrelevant
- 5 = Partially relevant
- 10 = Highly relevant

Format: SCORE: X
EXPLANATION: [brief explanation]"""

    response = llm.invoke(prompt)

    # Extract score
    try:
        score_line = [line for line in response.split('\n') if 'SCORE:' in line][0]
        score = float(score_line.split(':')[1].strip())
        score = min(10, max(0, score)) / 10
    except Exception:
        score = 0.5

    return score, response


def generate_rag_responses(vector_store, questions, top_k=5):
    """Generate RAG responses for evaluation."""
    answers = []
    contexts = []

    rag_chat = RAGChat(vector_store, top_k=top_k)

    for question in questions:
        print(f"\nProcessing: {question}")

        # Retrieve contexts
        results = vector_store.retrieve(question, top_k)

        if len(results) == 0:
            answers.append("No relevant information found.")
            contexts.append([])
            continue

        # Extract context texts
        context_texts = [text for text, score in results]
        contexts.append(context_texts)

        # Generate answer
        context = "\n---\n".join([f"[s{idx}] " + text for idx, text in enumerate(context_texts, 1)])

        prompt = (
            "You are a helpful assistant. Use the following context to answer the question.\n"
            "When you make a factual statement supported by a source, append the corresponding [sX] tag.\n"
            f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        answer = rag_chat.llm.invoke(prompt)
        answers.append(answer)

        print(f"✓ Answer generated ({len(answer)} chars)")

    return answers, contexts


def run_evaluation(chat_interface: RAGChat, embedding_model_name: str):
    """Run Ollama-based evaluation on FAISS RAG system."""
    print("=" * 80)
    print("FAISS RAG Evaluation (rag.py)")
    print("=" * 80)

    # Initialize Ollama
    print(f"\n[0/6] Initializing Ollama model ({EVAL_MODEL_NAME})...")
    try:
        eval_llm = OllamaLLM(model=EVAL_MODEL_NAME, temperature=0)
        eval_llm.invoke("Hello")
        print("✓ Ollama model ready")
    except Exception as e:
        print(f"✗ Error: Could not connect to Ollama: {e}")
        return

    # # Load vector store
    # print("\n[1/6] Loading FAISS vector store...")
    # store = FaissVectorStore(model_name=EMBEDDING_MODEL_NAME, vector_store_path=VECTOR_STORE_PATH)
    # if not store.texts:
    #     print(f"✗ Error: Vector store is empty or not found at {VECTOR_STORE_PATH}")
    #     print("Please run ingestion first: python test/rag.py ingest 'data/texts/*.txt'")
    #     return
    # print(f"✓ Loaded vector store with {len(store.texts)} chunks")

    # Create test dataset
    print("\n[2/6] Creating test dataset...")
    test_data = create_test_dataset()
    print(f"✓ Created {len(test_data['question'])} test cases")

    # Generate RAG responses
    print(f"\n[3/6] Generating RAG responses (top_k={chat_interface.top_k})...")
    answers, contexts = [], []
    for question in tqdm(test_data["question"]):
        answer, context_list = chat_interface.answer(question, stream=False)
        answers.append(answer)
        contexts.append(context_list)
    print(f"✓ Generated {len(answers)} answers")

    # Run evaluations
    print("\n[4/6] Running evaluations...")
    print("This will take several minutes...\n")

    results = {
        "faithfulness": [],
        "relevancy": [],
        "correctness": [],
        "context_relevance": [],
    }

    detailed_results = []

    for i, (question, answer, context_list, ground_truth) in tqdm(enumerate(
            zip(test_data["question"], answers, contexts, test_data["ground_truth"]), 1
    )):
        print(f"\n--- Evaluating Question {i}/{len(test_data['question'])} ---")
        print(f"Q: {question[:80]}...")

        if not context_list:
            print("⚠ No contexts retrieved, skipping")
            continue

        # Evaluate all metrics
        print("  - Evaluating faithfulness...")
        faith_score, faith_exp = evaluate_with_llm(eval_llm, "faithfulness", question, answer, context_list,
                                                   ground_truth)
        results["faithfulness"].append(faith_score)

        print("  - Evaluating relevancy...")
        rel_score, rel_exp = evaluate_with_llm(eval_llm, "relevancy", question, answer, context_list, ground_truth)
        results["relevancy"].append(rel_score)

        print("  - Evaluating correctness...")
        corr_score, corr_exp = evaluate_with_llm(eval_llm, "correctness", question, answer, context_list, ground_truth)
        results["correctness"].append(corr_score)

        print("  - Evaluating context relevance...")
        ctx_score, ctx_exp = evaluate_with_llm(eval_llm, "context_relevance", question, answer, context_list,
                                               ground_truth)
        results["context_relevance"].append(ctx_score)

        print(f"  ✓ Scores: Faith={faith_score:.2f}, Rel={rel_score:.2f}, Corr={corr_score:.2f}, Ctx={ctx_score:.2f}")

        detailed_results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "num_contexts": len(context_list),
            "scores": {
                "faithfulness": float(faith_score),
                "relevancy": float(rel_score),
                "correctness": float(corr_score),
                "context_relevance": float(ctx_score),
            }
        })

    # Calculate averages
    print("\n[5/6] Calculating final scores...")
    avg_scores = {
        "faithfulness": sum(results["faithfulness"]) / len(results["faithfulness"]) if results["faithfulness"] else 0,
        "relevancy": sum(results["relevancy"]) / len(results["relevancy"]) if results["relevancy"] else 0,
        "correctness": sum(results["correctness"]) / len(results["correctness"]) if results["correctness"] else 0,
        "context_relevance": sum(results["context_relevance"]) / len(results["context_relevance"]) if results[
            "context_relevance"] else 0,
    }

    overall_avg = sum(avg_scores.values()) / len(avg_scores)

    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - FAISS RAG")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Score':<10} {'Description'}")
    print("-" * 80)
    print(f"{'Faithfulness':<25} {avg_scores['faithfulness']:.4f}     How grounded in context")
    print(f"{'Answer Relevancy':<25} {avg_scores['relevancy']:.4f}     How relevant to question")
    print(f"{'Correctness':<25} {avg_scores['correctness']:.4f}     Accuracy vs ground truth")
    print(f"{'Context Relevance':<25} {avg_scores['context_relevance']:.4f}     Context quality")

    print("-" * 80)
    print(f"{'AVERAGE SCORE':<25} {overall_avg:.4f}")
    print("=" * 80)

    # Save results
    print("\n[6/6] Saving results...")
    final_results = {
        "model": "FAISS RAG",
        "summary": avg_scores,
        "overall_average": float(overall_avg),
        "config": {
            "top_k": chat_interface.top_k,
            "embedding_model": embedding_model_name,
            "chat_model": chat_interface.llm.model,
            "vector_store_path": VECTOR_STORE_PATH,
        },
        "test_cases": detailed_results,
    }

    with open(TEST_RESULTS_PATH, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"✓ Results saved to: {TEST_RESULTS_PATH}")

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FAISS RAG system")
    parser.add_argument("--top_k", type=int, default=5, help="Number of contexts to retrieve")

    args = parser.parse_args()
    print("Will not run!")
    # run_evaluation(top_k=args.top_k) # Outdated