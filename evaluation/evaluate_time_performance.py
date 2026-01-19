import os
import json
import time
import statistics
from dotenv import load_dotenv
import sys

# Add paths
sys.path.append('test')
sys.path.append('evaluation')
from test_questions import get_test_dataset, load_ground_truth_from_file

load_dotenv()

RESULTS_DIR = os.path.join("evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
PERF_RESULTS_PATH = os.path.join(RESULTS_DIR, "performance_results.json")


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


def evaluate_faiss_performance(top_k=5, num_runs=3):
    """Evaluate FAISS RAG performance."""
    from rag import FaissVectorStore, RAGChat
    from langchain_ollama import OllamaLLM

    print("\n" + "="*80)
    print("FAISS RAG Performance Evaluation")
    print("="*80)

    VECTOR_STORE_PATH = os.path.join("data", "vector_store_faiss.pkl")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "llama3.2:latest")

    # Load vector store
    print("\nLoading vector store...")
    load_start = time.time()
    store = FaissVectorStore(model_name=EMBEDDING_MODEL_NAME, vector_store_path=VECTOR_STORE_PATH)
    load_time = time.time() - load_start

    if not store.texts:
        print("✗ Vector store is empty")
        return None

    print(f"✓ Loaded in {load_time:.2f}s ({len(store.texts)} chunks)")

    llm = OllamaLLM(model=CHAT_MODEL_NAME, temperature=0)
    test_data = create_test_dataset()

    results = {
        "model": "FAISS RAG",
        "load_time": load_time,
        "num_chunks": len(store.texts),
        "top_k": top_k,
        "num_runs": num_runs,
        "queries": []
    }

    for question in test_data["question"]:
        query_results = {
            "question": question,
            "retrieval_times": [],
            "generation_times": [],
            "total_times": [],
            "answer_lengths": [],
            "num_contexts": 0
        }

        print(f"\nTesting: {question[:60]}...")

        for run in range(num_runs):
            # Measure retrieval
            retrieval_start = time.time()
            retrieved = store.retrieve(question, top_k)
            retrieval_time = time.time() - retrieval_start

            if not retrieved:
                continue

            query_results["num_contexts"] = len(retrieved)

            # Prepare context
            context_texts = [text for text, score in retrieved]
            context = "\n---\n".join([f"[s{idx}] " + text for idx, text in enumerate(context_texts, 1)])

            prompt = (
                "Use the following context to answer the question.\n"
                "When you make a factual statement supported by a source, append the corresponding [sX] tag.\n"
                f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:"
            )

            # Measure generation
            generation_start = time.time()
            answer = llm.invoke(prompt)
            generation_time = time.time() - generation_start

            total_time = retrieval_time + generation_time

            query_results["retrieval_times"].append(retrieval_time)
            query_results["generation_times"].append(generation_time)
            query_results["total_times"].append(total_time)
            query_results["answer_lengths"].append(len(answer))

            print(f"  Run {run+1}: Retrieval={retrieval_time:.3f}s, Generation={generation_time:.3f}s, Total={total_time:.3f}s")

        # Calculate averages
        if query_results["total_times"]:
            query_results["avg_retrieval_time"] = statistics.mean(query_results["retrieval_times"])
            query_results["avg_generation_time"] = statistics.mean(query_results["generation_times"])
            query_results["avg_total_time"] = statistics.mean(query_results["total_times"])
            query_results["avg_answer_length"] = statistics.mean(query_results["answer_lengths"])

            results["queries"].append(query_results)

    # Calculate overall metrics
    all_retrieval_times = [q["avg_retrieval_time"] for q in results["queries"]]
    all_generation_times = [q["avg_generation_time"] for q in results["queries"]]
    all_total_times = [q["avg_total_time"] for q in results["queries"]]

    results["summary"] = {
        "avg_retrieval_time": statistics.mean(all_retrieval_times),
        "avg_generation_time": statistics.mean(all_generation_times),
        "avg_total_time": statistics.mean(all_total_times),
        "throughput_qps": 1.0 / statistics.mean(all_total_times) if all_total_times else 0,
    }

    return results


def evaluate_hybrid_performance(top_k=5, num_runs=3):
    """Evaluate Hybrid RAG performance."""
    from rag_hybrid import HybridVectorStore, RAGChat
    from langchain_ollama import OllamaLLM

    print("\n" + "="*80)
    print("Hybrid RAG Performance Evaluation")
    print("="*80)

    VECTOR_STORE_PATH = os.path.join("data", "vector_store_hybrid.pkl")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "llama3.2:latest")

    # Load vector store
    print("\nLoading vector store...")
    load_start = time.time()
    store = HybridVectorStore(model_name=EMBEDDING_MODEL_NAME)
    store.load(VECTOR_STORE_PATH)
    load_time = time.time() - load_start

    if not store.texts:
        print("✗ Vector store is empty")
        return None

    print(f"✓ Loaded in {load_time:.2f}s ({len(store.texts)} chunks)")

    llm = OllamaLLM(model=CHAT_MODEL_NAME, temperature=0)
    test_data = create_test_dataset()

    results = {
        "model": "Hybrid RAG",
        "load_time": load_time,
        "num_chunks": len(store.texts),
        "top_k": top_k,
        "num_runs": num_runs,
        "queries": []
    }

    for question in test_data["question"]:
        query_results = {
            "question": question,
            "retrieval_times": [],
            "generation_times": [],
            "total_times": [],
            "answer_lengths": [],
            "num_contexts": 0
        }

        print(f"\nTesting: {question[:60]}...")

        for run in range(num_runs):
            # Measure retrieval
            retrieval_start = time.time()
            retrieved = store.retrieve(question, top_k)
            retrieval_time = time.time() - retrieval_start

            if not retrieved:
                continue

            query_results["num_contexts"] = len(retrieved)

            # Prepare context
            context_texts = [text for text, score in retrieved]
            context = "\n---\n".join([f"[s{idx}] " + text for idx, text in enumerate(context_texts, 1)])

            prompt = (
                "You are a tutor helping students understand machine learning concepts.\n"
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

            # Measure generation
            generation_start = time.time()
            answer = llm.invoke(prompt)
            generation_time = time.time() - generation_start

            total_time = retrieval_time + generation_time

            query_results["retrieval_times"].append(retrieval_time)
            query_results["generation_times"].append(generation_time)
            query_results["total_times"].append(total_time)
            query_results["answer_lengths"].append(len(answer))

            print(f"  Run {run+1}: Retrieval={retrieval_time:.3f}s, Generation={generation_time:.3f}s, Total={total_time:.3f}s")

        # Calculate averages
        if query_results["total_times"]:
            query_results["avg_retrieval_time"] = statistics.mean(query_results["retrieval_times"])
            query_results["avg_generation_time"] = statistics.mean(query_results["generation_times"])
            query_results["avg_total_time"] = statistics.mean(query_results["total_times"])
            query_results["avg_answer_length"] = statistics.mean(query_results["answer_lengths"])

            results["queries"].append(query_results)

    # Calculate overall metrics
    all_retrieval_times = [q["avg_retrieval_time"] for q in results["queries"]]
    all_generation_times = [q["avg_generation_time"] for q in results["queries"]]
    all_total_times = [q["avg_total_time"] for q in results["queries"]]

    results["summary"] = {
        "avg_retrieval_time": statistics.mean(all_retrieval_times),
        "avg_generation_time": statistics.mean(all_generation_times),
        "avg_total_time": statistics.mean(all_total_times),
        "throughput_qps": 1.0 / statistics.mean(all_total_times) if all_total_times else 0,
    }

    return results


def evaluate_graphrag_performance(top_k=5, num_runs=3):
    """Evaluate GraphRAG performance."""
    try:
        from neo4j import GraphDatabase
        from sentence_transformers import SentenceTransformer
        from langchain_ollama import OllamaLLM

        # Import from test directory explicitly
        import importlib.util
        import os as os_import

        # Load the graphrag module from test directory
        spec = importlib.util.spec_from_file_location(
            "graphrag_w_neo4j",
            os_import.path.join("test", "graphrag_w_neo4j.py")
        )
        graphrag_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graphrag_module)

        Neo4jVectorStore = graphrag_module.Neo4jVectorStore

    except Exception as e:
        print(f"✗ Failed to import GraphRAG dependencies: {e}")
        print(f"  Make sure test/graphrag_w_neo4j.py exists")
        return None

    print("\n" + "="*80)
    print("Neo4j GraphRAG Performance Evaluation")
    print("="*80)

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX", "chunk_embedding_idx")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "llama3.2:latest")

    # Connect to Neo4j
    print("\nConnecting to Neo4j...")
    connection_start = time.time()
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            if result.single()["test"] != 1:
                raise Exception("Connection test failed")
        connection_time = time.time() - connection_start
        print(f"✓ Connected in {connection_time:.2f}s")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector_store = Neo4jVectorStore(driver, INDEX_NAME)
    llm = OllamaLLM(model=CHAT_MODEL_NAME, temperature=0)

    test_data = create_test_dataset()

    results = {
        "model": "Neo4j GraphRAG",
        "connection_time": connection_time,
        "top_k": top_k,
        "num_runs": num_runs,
        "queries": []
    }

    for question in test_data["question"]:
        query_results = {
            "question": question,
            "encoding_times": [],
            "retrieval_times": [],
            "generation_times": [],
            "total_times": [],
            "answer_lengths": [],
            "num_contexts": 0
        }

        print(f"\nTesting: {question[:60]}...")

        for run in range(num_runs):
            # Measure encoding
            encoding_start = time.time()
            qvec = embedder.encode(question).tolist()
            encoding_time = time.time() - encoding_start

            # Measure retrieval
            retrieval_start = time.time()
            retrieved = vector_store.retrieve(qvec, top_k)
            retrieval_time = time.time() - retrieval_start

            if not retrieved:
                continue

            query_results["num_contexts"] = len(retrieved)

            # Prepare context
            context_texts = [text for text, score in retrieved]
            sources = [f"[s{i}] {txt}" for i, (txt, _) in enumerate(retrieved, 1)]
            context = "\n---\n".join(sources)

            prompt = (
                "You are an AI tutor.\n"
                "CRITICAL RULES:\n"
                "1) Answer ONLY using the Sources.\n"
                "2) If the Sources do not contain the answer, say: "
                "'I don't have information about that in your notes.'\n"
                "3) Cite sources with [sX].\n\n"
                f"Sources:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            # Measure generation
            generation_start = time.time()
            answer = llm.invoke(prompt)
            generation_time = time.time() - generation_start

            total_time = encoding_time + retrieval_time + generation_time

            query_results["encoding_times"].append(encoding_time)
            query_results["retrieval_times"].append(retrieval_time)
            query_results["generation_times"].append(generation_time)
            query_results["total_times"].append(total_time)
            query_results["answer_lengths"].append(len(answer))

            print(f"  Run {run+1}: Encode={encoding_time:.3f}s, Retrieval={retrieval_time:.3f}s, "
                  f"Generation={generation_time:.3f}s, Total={total_time:.3f}s")

        # Calculate averages
        if query_results["total_times"]:
            query_results["avg_encoding_time"] = statistics.mean(query_results["encoding_times"])
            query_results["avg_retrieval_time"] = statistics.mean(query_results["retrieval_times"])
            query_results["avg_generation_time"] = statistics.mean(query_results["generation_times"])
            query_results["avg_total_time"] = statistics.mean(query_results["total_times"])
            query_results["avg_answer_length"] = statistics.mean(query_results["answer_lengths"])

            results["queries"].append(query_results)

    # Calculate overall metrics
    all_encoding_times = [q["avg_encoding_time"] for q in results["queries"]]
    all_retrieval_times = [q["avg_retrieval_time"] for q in results["queries"]]
    all_generation_times = [q["avg_generation_time"] for q in results["queries"]]
    all_total_times = [q["avg_total_time"] for q in results["queries"]]

    results["summary"] = {
        "avg_encoding_time": statistics.mean(all_encoding_times),
        "avg_retrieval_time": statistics.mean(all_retrieval_times),
        "avg_generation_time": statistics.mean(all_generation_times),
        "avg_total_time": statistics.mean(all_total_times),
        "throughput_qps": 1.0 / statistics.mean(all_total_times) if all_total_times else 0,
    }

    driver.close()
    return results


def display_performance_comparison(all_results):
    """Display performance comparison table."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)

    # Filter out None results
    valid_results = [r for r in all_results if r is not None]

    if not valid_results:
        print("No results to display")
        return

    # Header
    print(f"\n{'Metric':<30}", end="")
    for result in valid_results:
        print(f"{result['model']:<25}", end="")
    print()
    print("-" * (30 + 25 * len(valid_results)))

    # Load time
    print(f"{'Load/Connect Time (s)':<30}", end="")
    for result in valid_results:
        load_time = result.get('load_time') or result.get('connection_time', 0)
        print(f"{load_time:<25.3f}", end="")
    print()

    # Average retrieval time
    print(f"{'Avg Retrieval Time (s)':<30}", end="")
    for result in valid_results:
        print(f"{result['summary']['avg_retrieval_time']:<25.3f}", end="")
    print()

    # Average generation time
    print(f"{'Avg Generation Time (s)':<30}", end="")
    for result in valid_results:
        print(f"{result['summary']['avg_generation_time']:<25.3f}", end="")
    print()

    # Average total time
    print(f"{'Avg Total Time (s)':<30}", end="")
    for result in valid_results:
        print(f"{result['summary']['avg_total_time']:<25.3f}", end="")
    print()

    # Throughput
    print(f"{'Throughput (queries/sec)':<30}", end="")
    for result in valid_results:
        print(f"{result['summary']['throughput_qps']:<25.2f}", end="")
    print()

    print("="*80)

    # Winner analysis
    print("\nFASTEST BY METRIC:")
    print("-"*80)

    retrieval_times = {r['model']: r['summary']['avg_retrieval_time'] for r in valid_results}
    fastest_retrieval = min(retrieval_times, key=retrieval_times.get)
    print(f"Retrieval:   {fastest_retrieval:<30} ({retrieval_times[fastest_retrieval]:.3f}s)")

    generation_times = {r['model']: r['summary']['avg_generation_time'] for r in valid_results}
    fastest_generation = min(generation_times, key=generation_times.get)
    print(f"Generation:  {fastest_generation:<30} ({generation_times[fastest_generation]:.3f}s)")

    total_times = {r['model']: r['summary']['avg_total_time'] for r in valid_results}
    fastest_total = min(total_times, key=total_times.get)
    print(f"Overall:     {fastest_total:<30} ({total_times[fastest_total]:.3f}s)")

    print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument("--top_k", type=int, default=5, help="Number of contexts to retrieve")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per query for averaging")
    parser.add_argument("--systems", nargs="+", choices=["faiss", "hybrid", "graphrag", "all"],
                        default=["all"], help="Which systems to evaluate")

    args = parser.parse_args()

    print("="*80)
    print("RAG PERFORMANCE EVALUATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  top_k: {args.top_k}")
    print(f"  runs per query: {args.runs}")
    print(f"  systems: {args.systems}")

    all_results = []

    # Determine which systems to test
    systems_to_test = set(args.systems)
    if "all" in systems_to_test:
        systems_to_test = {"faiss", "hybrid", "graphrag"}

    # Run evaluations
    if "faiss" in systems_to_test:
        try:
            faiss_results = evaluate_faiss_performance(args.top_k, args.runs)
            if faiss_results:
                all_results.append(faiss_results)
        except Exception as e:
            print(f"\n✗ FAISS evaluation failed: {e}")

    if "hybrid" in systems_to_test:
        try:
            hybrid_results = evaluate_hybrid_performance(args.top_k, args.runs)
            if hybrid_results:
                all_results.append(hybrid_results)
        except Exception as e:
            print(f"\n✗ Hybrid evaluation failed: {e}")

    if "graphrag" in systems_to_test:
        try:
            graphrag_results = evaluate_graphrag_performance(args.top_k, args.runs)
            if graphrag_results:
                all_results.append(graphrag_results)
        except Exception as e:
            print(f"\n✗ GraphRAG evaluation failed: {e}")

    # Display comparison
    if all_results:
        display_performance_comparison(all_results)

        # Save results
        with open(PERF_RESULTS_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Detailed results saved to: {PERF_RESULTS_PATH}")
    else:
        print("\n✗ No results to compare")


if __name__ == "__main__":
    main()