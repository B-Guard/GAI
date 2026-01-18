import os
import json
import subprocess
import sys
from datetime import datetime

# Ensure results directory exists
RESULTS_DIR = os.path.join("evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
COMPARISON_RESULTS_PATH = os.path.join(RESULTS_DIR, "comparison_results.json")


def run_evaluation(script_name, model_name, top_k=5):
    """Run an evaluation script and return the results."""
    print("\n" + "=" * 80)
    print(f"Running {model_name} Evaluation...")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, f"evaluation/{script_name}", "--top_k", str(top_k)],
            capture_output=False,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {model_name} evaluation: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def load_results(filepath):
    """Load evaluation results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Results file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def compare_results(top_k=5):
    """Run all evaluations and compare results."""
    print("=" * 80)
    print("RAG SYSTEMS COMPARISON EVALUATION")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"top_k: {top_k}")

    # Define evaluation scripts
    evaluations = [
        {
            "script": "evaluate_rag.py",
            "name": "FAISS RAG",
            "results_file": os.path.join(RESULTS_DIR, "evaluation_results_faiss.json"),
        },
        {
            "script": "evaluate_rag_hybrid.py",
            "name": "Hybrid RAG (FAISS + BM25)",
            "results_file": os.path.join(RESULTS_DIR, "evaluation_results_hybrid.json"),
        },
        {
            "script": "evaluate_graphrag.py",
            "name": "Neo4j GraphRAG",
            "results_file": os.path.join(RESULTS_DIR, "evaluation_results_graphrag.json"),
        },
    ]

    # Run all evaluations
    print("\n" + "─" * 80)
    print("STEP 1: Running Individual Evaluations")
    print("─" * 80)

    for eval_config in evaluations:
        success = run_evaluation(eval_config["script"], eval_config["name"], top_k)
        if not success:
            print(f"⚠ {eval_config['name']} evaluation failed or was skipped")

    # Load all results
    print("\n" + "─" * 80)
    print("STEP 2: Loading and Comparing Results")
    print("─" * 80)

    all_results = []
    for eval_config in evaluations:
        results = load_results(eval_config["results_file"])
        if results:
            all_results.append({
                "name": eval_config["name"],
                "results": results
            })

    if not all_results:
        print("✗ No results to compare. Make sure at least one evaluation completed successfully.")
        return

    # Display comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<25}", end="")
    for result in all_results:
        print(f"{result['name']:<25}", end="")
    print()
    print("-" * (25 + 25 * len(all_results)))

    # Metrics to compare
    metrics = ["faithfulness", "relevancy", "correctness", "context_relevance"]
    metric_names = {
        "faithfulness": "Faithfulness",
        "relevancy": "Answer Relevancy",
        "correctness": "Correctness",
        "context_relevance": "Context Relevance",
    }

    comparison_data = {}

    for metric in metrics:
        print(f"{metric_names[metric]:<25}", end="")
        comparison_data[metric] = {}

        for result in all_results:
            score = result["results"]["summary"].get(metric, 0)
            comparison_data[metric][result["name"]] = score
            print(f"{score:.4f} ({score * 100:.1f}%){' ':<10}", end="")
        print()

    # Overall average
    print("-" * (25 + 25 * len(all_results)))
    print(f"{'OVERALL AVERAGE':<25}", end="")

    overall_scores = {}
    for result in all_results:
        avg = result["results"].get("overall_average", 0)
        overall_scores[result["name"]] = avg
        print(f"{avg:.4f} ({avg * 100:.1f}%){' ':<10}", end="")
    print()

    print("=" * (25 + 25 * len(all_results)))

    # Determine winner for each metric
    print("\n" + "─" * 80)
    print("BEST PERFORMER BY METRIC")
    print("─" * 80)

    winners = {}
    for metric in metrics:
        scores = comparison_data[metric]
        if scores:
            best_model = max(scores, key=scores.get)
            best_score = scores[best_model]
            winners[metric] = {"model": best_model, "score": best_score}
            print(f"{metric_names[metric]:<25} → {best_model:<30} ({best_score:.4f})")

    # Overall winner
    if overall_scores:
        best_overall = max(overall_scores, key=overall_scores.get)
        best_overall_score = overall_scores[best_overall]
        print("-" * 80)
        print(f"{'OVERALL BEST':<25} → {best_overall:<30} ({best_overall_score:.4f})")
        print("=" * 80)

    # Save comparison results
    print("\n" + "─" * 80)
    print("STEP 3: Saving Comparison Results")
    print("─" * 80)

    comparison_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": top_k,
        },
        "models": [r["name"] for r in all_results],
        "metrics": comparison_data,
        "overall_scores": overall_scores,
        "winners": winners,
        "best_overall": {
            "model": best_overall,
            "score": best_overall_score
        } if overall_scores else None,
    }

    with open(COMPARISON_RESULTS_PATH, 'w') as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"✓ Comparison results saved to: {COMPARISON_RESULTS_PATH}")

    # Performance insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    if len(all_results) >= 2:
        # Compare FAISS vs Hybrid
        faiss_result = next((r for r in all_results if "FAISS RAG" in r["name"]), None)
        hybrid_result = next((r for r in all_results if "Hybrid" in r["name"]), None)

        if faiss_result and hybrid_result:
            faiss_score = overall_scores.get(faiss_result["name"], 0)
            hybrid_score = overall_scores.get(hybrid_result["name"], 0)
            diff = hybrid_score - faiss_score

            print(f"\n📊 Hybrid RAG vs Simple FAISS:")
            if abs(diff) < 0.01:
                print(f"   Similar performance (difference: {abs(diff):.4f})")
            elif diff > 0:
                print(f"   Hybrid is better by {diff:.4f} ({diff * 100:.1f}%)")
                print(f"   → BM25 keyword search adds value")
            else:
                print(f"   FAISS is better by {abs(diff):.4f} ({abs(diff) * 100:.1f}%)")
                print(f"   → Pure semantic search performs better")

        # Compare with GraphRAG
        graph_result = next((r for r in all_results if "GraphRAG" in r["name"]), None)

        if graph_result:
            graph_score = overall_scores.get(graph_result["name"], 0)
            print(f"\n🕸️  GraphRAG Performance:")
            print(f"   Overall score: {graph_score:.4f}")

            if faiss_result:
                faiss_score = overall_scores.get(faiss_result["name"], 0)
                diff = graph_score - faiss_score
                if diff > 0:
                    print(f"   Better than FAISS by {diff:.4f} ({diff * 100:.1f}%)")
                    print(f"   → Graph structure adds value")
                else:
                    print(f"   FAISS performs better by {abs(diff):.4f} ({abs(diff) * 100:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare all RAG systems: FAISS, Hybrid, and GraphRAG"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of contexts to retrieve (default: 5)"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running evaluations, just load and compare existing results"
    )

    args = parser.parse_args()

    if args.skip_eval:
        print("Skipping evaluations, loading existing results...")
        # Just load and compare
        print("=" * 80)
        print("LOADING EXISTING RESULTS")
        print("=" * 80)

        # Load results directly
        results_files = {
            "FAISS RAG": os.path.join(RESULTS_DIR, "evaluation_results_faiss.json"),
            "Hybrid RAG": os.path.join(RESULTS_DIR, "evaluation_results_hybrid.json"),
            "Neo4j GraphRAG": os.path.join(RESULTS_DIR, "evaluation_results_graphrag.json"),
        }

        all_results = []
        for name, filepath in results_files.items():
            results = load_results(filepath)
            if results:
                all_results.append({"name": name, "results": results})

        if all_results:
            # Display comparison (reuse the comparison logic)
            # ... (same comparison display code as above)
            print("Use --top_k to run fresh evaluations")
        else:
            print("No existing results found. Run without --skip-eval to generate results.")
    else:
        compare_results(top_k=args.top_k)