import os
import json
from dotenv import load_dotenv
import sys
from langchain_ollama import OllamaLLM

# Add test and evaluation directories to path
sys.path.append('test')
sys.path.append('evaluation')
from rag_hybrid import HybridVectorStore, RAGChat
from test_questions import get_test_dataset, load_ground_truth_from_file

load_dotenv()

VECTOR_STORE_PATH = os.path.join("data", "vector_store_hybrid.pkl")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "llama3.2:latest")

# Ensure results directory exists
RESULTS_DIR = os.path.join("evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TEST_RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation_results_hybrid.json")


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


def evaluate_faithfulness(llm, answer, contexts):
    """
    Evaluate if the answer is faithful to the retrieved contexts.
    Returns a score between 0 and 1.
    """
    context_text = "\n\n".join(contexts)

    prompt = f"""You are an evaluation judge. Your task is to determine if the ANSWER is faithful to the CONTEXTS provided.

An answer is faithful if:
1. All claims in the answer are supported by the contexts
2. The answer doesn't add information not present in contexts
3. The answer doesn't contradict the contexts

CONTEXTS:
{context_text}

ANSWER:
{answer}

Rate the faithfulness on a scale of 0-10 where:
- 0 = Completely unfaithful (makes up information or contradicts contexts)
- 5 = Partially faithful (some claims supported, some not)
- 10 = Completely faithful (all claims directly supported by contexts)

Respond with ONLY a number between 0 and 10, followed by a brief explanation.
Format: SCORE: X
EXPLANATION: [brief explanation]
"""

    response = llm.invoke(prompt)

    # Extract score
    try:
        score_line = [line for line in response.split('\n') if 'SCORE:' in line][0]
        score = float(score_line.split(':')[1].strip())
        score = min(10, max(0, score)) / 10  # Normalize to 0-1
    except:
        score = 0.5  # Default if parsing fails

    return score, response


def evaluate_relevancy(llm, question, answer):
    """
    Evaluate if the answer is relevant to the question.
    Returns a score between 0 and 1.
    """
    prompt = f"""You are an evaluation judge. Your task is to determine if the ANSWER is relevant to the QUESTION.

An answer is relevant if:
1. It directly addresses what was asked
2. It provides useful information related to the question
3. It doesn't go off-topic

QUESTION:
{question}

ANSWER:
{answer}

Rate the relevancy on a scale of 0-10 where:
- 0 = Completely irrelevant (doesn't address the question at all)
- 5 = Partially relevant (addresses some aspects)
- 10 = Highly relevant (directly and completely answers the question)

Respond with ONLY a number between 0 and 10, followed by a brief explanation.
Format: SCORE: X
EXPLANATION: [brief explanation]
"""

    response = llm.invoke(prompt)

    # Extract score
    try:
        score_line = [line for line in response.split('\n') if 'SCORE:' in line][0]
        score = float(score_line.split(':')[1].strip())
        score = min(10, max(0, score)) / 10  # Normalize to 0-1
    except:
        score = 0.5  # Default if parsing fails

    return score, response


def evaluate_correctness(llm, question, answer, ground_truth):
    """
    Evaluate if the answer is correct compared to ground truth.
    Returns a score between 0 and 1.
    """
    prompt = f"""You are an evaluation judge. Your task is to determine if the ANSWER is correct compared to the GROUND TRUTH.

The answer doesn't need to be identical, but should convey the same key information and concepts.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

ANSWER:
{answer}

Rate the correctness on a scale of 0-10 where:
- 0 = Completely incorrect (wrong information)
- 5 = Partially correct (some correct concepts, some missing or wrong)
- 10 = Completely correct (captures all key concepts accurately)

Respond with ONLY a number between 0 and 10, followed by a brief explanation.
Format: SCORE: X
EXPLANATION: [brief explanation]
"""

    response = llm.invoke(prompt)

    # Extract score
    try:
        score_line = [line for line in response.split('\n') if 'SCORE:' in line][0]
        score = float(score_line.split(':')[1].strip())
        score = min(10, max(0, score)) / 10  # Normalize to 0-1
    except:
        score = 0.5  # Default if parsing fails

    return score, response


def evaluate_context_relevance(llm, question, contexts):
    """
    Evaluate if the retrieved contexts are relevant to the question.
    Returns a score between 0 and 1.
    """
    context_text = "\n\n".join([f"Context {i + 1}: {ctx[:200]}..." for i, ctx in enumerate(contexts)])

    prompt = f"""You are an evaluation judge. Your task is to determine if the CONTEXTS are relevant to the QUESTION.

QUESTION:
{question}

CONTEXTS:
{context_text}

Rate how relevant the contexts are on a scale of 0-10 where:
- 0 = Completely irrelevant (contexts don't relate to the question)
- 5 = Partially relevant (some contexts useful, some not)
- 10 = Highly relevant (all contexts directly useful for answering)

Respond with ONLY a number between 0 and 10, followed by a brief explanation.
Format: SCORE: X
EXPLANATION: [brief explanation]
"""

    response = llm.invoke(prompt)

    # Extract score
    try:
        score_line = [line for line in response.split('\n') if 'SCORE:' in line][0]
        score = float(score_line.split(':')[1].strip())
        score = min(10, max(0, score)) / 10  # Normalize to 0-1
    except:
        score = 0.5  # Default if parsing fails

    return score, response


def generate_rag_responses(vector_store, questions, top_k=5):
    """
    Generate RAG responses for evaluation.
    Returns answers and retrieved contexts.
    """
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

        # Generate answer using the RAG system
        context = "\n---\n".join([f"[s{idx}] " + text for idx, text in enumerate(context_texts, 1)])

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

        # Get complete answer (not streaming)
        answer = rag_chat.llm.invoke(prompt)
        answers.append(answer)

        print(f"✓ Answer generated ({len(answer)} chars)")

    return answers, contexts


def run_ollama_evaluation(top_k=5):
    """
    Run Ollama-based evaluation on the RAG system.
    """
    print("=" * 80)
    print("Ollama-Based RAG Evaluation")
    print("=" * 80)

    # Initialize Ollama LLM for evaluation
    print(f"\n[0/6] Initializing Ollama model ({CHAT_MODEL_NAME})...")
    try:
        eval_llm = OllamaLLM(model=CHAT_MODEL_NAME, temperature=0)
        # Test the model
        eval_llm.invoke("Hello")
        print("✓ Ollama model ready")
    except Exception as e:
        print(f"✗ Error: Could not connect to Ollama: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Check if Ollama app is open (menu bar icon)")
        print("  2. Or run: ollama serve")
        print(f"  3. Verify model is available: ollama list")
        return

    # Load vector store
    print("\n[1/6] Loading vector store...")
    store = HybridVectorStore(model_name=EMBEDDING_MODEL_NAME)
    try:
        store.load(VECTOR_STORE_PATH)
        print(f"✓ Loaded vector store with {len(store.texts)} chunks")
    except FileNotFoundError:
        print(f"✗ Error: Vector store not found at {VECTOR_STORE_PATH}")
        print("Please run ingestion first: python test/rag_hybrid.py ingest 'data/texts/*.txt'")
        return

    # Create test dataset
    print("\n[2/6] Creating test dataset...")
    test_data = create_test_dataset()
    print(f"✓ Created {len(test_data['question'])} test cases")

    # Generate RAG responses
    print(f"\n[3/6] Generating RAG responses (top_k={top_k})...")
    answers, contexts = generate_rag_responses(store, test_data["question"], top_k)
    print(f"✓ Generated {len(answers)} answers")

    # Run evaluations
    print("\n[4/6] Running Ollama-based evaluations...")
    print("This will take several minutes...\n")

    results = {
        "faithfulness": [],
        "relevancy": [],
        "correctness": [],
        "context_relevance": [],
    }

    detailed_results = []

    for i, (question, answer, context_list, ground_truth) in enumerate(
            zip(test_data["question"], answers, contexts, test_data["ground_truth"]), 1
    ):
        print(f"\n--- Evaluating Question {i}/{len(test_data['question'])} ---")
        print(f"Q: {question[:80]}...")

        if not context_list:
            print("⚠ No contexts retrieved, skipping evaluation")
            continue

        # Evaluate faithfulness
        print("  - Evaluating faithfulness...")
        faith_score, faith_exp = evaluate_faithfulness(eval_llm, answer, context_list)
        results["faithfulness"].append(faith_score)

        # Evaluate relevancy
        print("  - Evaluating relevancy...")
        rel_score, rel_exp = evaluate_relevancy(eval_llm, question, answer)
        results["relevancy"].append(rel_score)

        # Evaluate correctness
        print("  - Evaluating correctness...")
        corr_score, corr_exp = evaluate_correctness(eval_llm, question, answer, ground_truth)
        results["correctness"].append(corr_score)

        # Evaluate context relevance
        print("  - Evaluating context relevance...")
        ctx_score, ctx_exp = evaluate_context_relevance(eval_llm, question, context_list)
        results["context_relevance"].append(ctx_score)

        print(f"  ✓ Scores: Faith={faith_score:.2f}, Rel={rel_score:.2f}, "
              f"Corr={corr_score:.2f}, Ctx={ctx_score:.2f}")

        # Store detailed results
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
            },
            "evaluations": {
                "faithfulness": faith_exp,
                "relevancy": rel_exp,
                "correctness": corr_exp,
                "context_relevance": ctx_exp,
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
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Score':<10} {'Description'}")
    print("-" * 80)
    print(f"{'Faithfulness':<25} {avg_scores['faithfulness']:.4f}     "
          "How grounded answers are in context")
    print(f"{'Answer Relevancy':<25} {avg_scores['relevancy']:.4f}     "
          "How relevant answers are to questions")
    print(f"{'Correctness':<25} {avg_scores['correctness']:.4f}     "
          "How correct answers are vs ground truth")
    print(f"{'Context Relevance':<25} {avg_scores['context_relevance']:.4f}     "
          "How relevant contexts are to questions")

    print("-" * 80)
    print(f"{'AVERAGE SCORE':<25} {overall_avg:.4f}")
    print("=" * 80)

    # Save results
    print(f"\n[6/6] Saving results...")
    final_results = {
        "summary": avg_scores,
        "overall_average": float(overall_avg),
        "config": {
            "top_k": top_k,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chat_model": CHAT_MODEL_NAME,
            "vector_store_path": VECTOR_STORE_PATH,
        },
        "test_cases": detailed_results,
    }

    with open(TEST_RESULTS_PATH, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"✓ Detailed results saved to: {TEST_RESULTS_PATH}")

    return final_results


def print_sample_outputs(store, num_samples=2):
    """
    Print sample Q&A outputs for manual inspection.
    """
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS (for manual inspection)")
    print("=" * 80)

    test_data = create_test_dataset()

    for i in range(min(num_samples, len(test_data["question"]))):
        question = test_data["question"][i]
        ground_truth = test_data["ground_truth"][i]

        print(f"\n{'─' * 80}")
        print(f"Question {i + 1}: {question}")
        print(f"Source: {test_data['source_file'][i]}")
        print(f"{'─' * 80}")

        # Retrieve and display
        results = store.retrieve(question, top_k=3)

        print(f"\nRetrieved Contexts ({len(results)} chunks):")
        for idx, (text, score) in enumerate(results, 1):
            print(f"\n[Context {idx}] (score: {score:.4f})")
            print(f"{text[:200]}..." if len(text) > 200 else text)

        print(f"\n{'─' * 80}")
        print(f"Ground Truth (first 200 chars): {ground_truth[:200]}...")
        print(f"{'─' * 80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system using Ollama (no OpenAI API needed)")
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of contexts to retrieve (default: 5)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Show sample outputs without running full evaluation"
    )

    args = parser.parse_args()

    if args.sample:
        # Just show samples
        print("Loading vector store for sample outputs...")
        store = HybridVectorStore(model_name=EMBEDDING_MODEL_NAME)
        try:
            store.load(VECTOR_STORE_PATH)
            print_sample_outputs(store)
        except FileNotFoundError:
            print(f"Error: Vector store not found at {VECTOR_STORE_PATH}")
    else:
        # Run full evaluation
        run_ollama_evaluation(top_k=args.top_k)