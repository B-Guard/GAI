"""
Test questions configuration for RAG evaluation.
This file contains the mapping between test questions and their corresponding ground truth files.
"""

import os

# Base directory for data files
DATA_DIR = os.path.join("data", "texts")

# Test questions mapped to their ground truth source files
TEST_QUESTIONS = {
    "What is attention in transformers?":
        "Attention in transformers, step-by-step.txt",

    "Explain how gradient descent works in neural networks":
        "Gradient descent, how neural networks learn.txt",

    "How does backpropagation work?":
        "Backpropagation, intuitively.txt",

    "How do LLMs work?":
        "Large Language Models explained briefly.txt",

    "What is proximal policy optimization?":
        "Proximal Policy Optimization (PPO) for LLMs Explained Intuitively.txt",
}


def get_test_dataset():
    """
    Returns test dataset with questions and ground truth file paths.

    Returns:
        dict: Dictionary with 'question', 'ground_truth_file', and 'source_file' lists
    """
    test_data = {
        "question": [],
        "ground_truth_file": [],
        "source_file": [],
    }

    for question, filename in TEST_QUESTIONS.items():
        test_data["question"].append(question)
        test_data["ground_truth_file"].append(os.path.join(DATA_DIR, filename))
        test_data["source_file"].append(filename)

    return test_data


def load_ground_truth_from_file(filepath):
    """
    Load ground truth text from a file.
    Args:
        filepath (str): Path to the ground truth file
    Returns:
        str: Ground truth text (first 2 paragraphs, max 500 chars)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Take first 2-3 paragraphs as ground truth
            paragraphs = content.split('\n\n')
            ground_truth = '\n'.join(paragraphs[:2])
            if len(ground_truth) > 500:
                ground_truth = ground_truth[:500] + "..."
            return ground_truth
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return "Ground truth not available"
