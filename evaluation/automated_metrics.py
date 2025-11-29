from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

class AutomatedEvaluator:
    """Automated evaluation metrics for QA."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def semantic_similarity(self, answer, ground_truth):
        """Calculate semantic similarity between answer and ground truth."""
        embeddings = self.model.encode([answer, ground_truth])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def answer_length_ratio(self, answer, ground_truth):
        """Compare answer length to ground truth."""
        ans_len = len(answer.split())
        gt_len = len(ground_truth.split())
        return min(ans_len, gt_len) / max(ans_len, gt_len)
    
    def keyword_overlap(self, answer, ground_truth):
        """Calculate keyword overlap."""
        ans_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at'}
        ans_words -= stop_words
        gt_words -= stop_words
        
        if not gt_words:
            return 0.0
        
        overlap = len(ans_words & gt_words) / len(gt_words)
        return overlap
    
    def evaluate(self, answer, ground_truth):
        """Run all automated metrics."""
        return {
            'semantic_similarity': self.semantic_similarity(answer, ground_truth),
            'length_ratio': self.answer_length_ratio(answer, ground_truth),
            'keyword_overlap': self.keyword_overlap(answer, ground_truth)
        }


if __name__ == "__main__":
    evaluator = AutomatedEvaluator()
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible paths for the JSON file
    possible_paths = [
        'full_results.json',  # Current directory
        os.path.join(script_dir, 'full_results.json'),  # Script directory
        os.path.join(script_dir, '..', 'full_results.json'),  # Parent directory
        os.path.join(script_dir, '..', 'data', 'full_results.json'),  # data folder
        os.path.join(script_dir, '..', 'results', 'full_results.json'),  # results folder
    ]
    
    json_path = None
    for path in possible_paths:
        if os.path.exists(path):
            json_path = path
            break
    
    if json_path is None:
        print("Error: Could not find 'full_results.json' file.")
        print(f"Script is running from: {script_dir}")
        print("\nPlease provide the full path to the JSON file:")
        json_path = input("Path: ").strip().strip('"').strip("'")
    
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Successfully loaded data from: {json_path}")
    print(f"Total questions: {len(data['results'])}\n")
    
    # Iterate through each question
    for result in data['results']:
        question_id = result['question_id']
        question = result['question']
        ground_truth_answer = result['ground_truth']
        good_answer = result['rag_answer']
        bad_answer = result['baseline_answer']
        
        print(f"\n{'='*80}")
        print(f"Question {question_id}: {question}")
        print(f"{'='*80}")
        
        print("\nGood answer (RAG) metrics:")
        rag_metrics = evaluator.evaluate(good_answer, ground_truth_answer)
        print(f"  Semantic Similarity: {rag_metrics['semantic_similarity']:.4f}")
        print(f"  Length Ratio: {rag_metrics['length_ratio']:.4f}")
        print(f"  Keyword Overlap: {rag_metrics['keyword_overlap']:.4f}")
        
        print("\nBad answer (Baseline) metrics:")
        baseline_metrics = evaluator.evaluate(bad_answer, ground_truth_answer)
        print(f"  Semantic Similarity: {baseline_metrics['semantic_similarity']:.4f}")
        print(f"  Length Ratio: {baseline_metrics['length_ratio']:.4f}")
        print(f"  Keyword Overlap: {baseline_metrics['keyword_overlap']:.4f}")
        
        print(f"\n{'='*80}\n")
    
    # Calculate and print overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    rag_similarities = []
    baseline_similarities = []
    rag_length_ratios = []
    baseline_length_ratios = []
    rag_keyword_overlaps = []
    baseline_keyword_overlaps = []
    
    for result in data['results']:
        ground_truth_answer = result['ground_truth']
        good_answer = result['rag_answer']
        bad_answer = result['baseline_answer']
        
        rag_metrics = evaluator.evaluate(good_answer, ground_truth_answer)
        baseline_metrics = evaluator.evaluate(bad_answer, ground_truth_answer)
        
        rag_similarities.append(rag_metrics['semantic_similarity'])
        baseline_similarities.append(baseline_metrics['semantic_similarity'])
        rag_length_ratios.append(rag_metrics['length_ratio'])
        baseline_length_ratios.append(baseline_metrics['length_ratio'])
        rag_keyword_overlaps.append(rag_metrics['keyword_overlap'])
        baseline_keyword_overlaps.append(baseline_metrics['keyword_overlap'])
    
    print(f"\nRAG Average Metrics:")
    print(f"  Semantic Similarity: {np.mean(rag_similarities):.4f} (±{np.std(rag_similarities):.4f})")
    print(f"  Length Ratio: {np.mean(rag_length_ratios):.4f} (±{np.std(rag_length_ratios):.4f})")
    print(f"  Keyword Overlap: {np.mean(rag_keyword_overlaps):.4f} (±{np.std(rag_keyword_overlaps):.4f})")
    
    print(f"\nBaseline Average Metrics:")
    print(f"  Semantic Similarity: {np.mean(baseline_similarities):.4f} (±{np.std(baseline_similarities):.4f})")
    print(f"  Length Ratio: {np.mean(baseline_length_ratios):.4f} (±{np.std(baseline_length_ratios):.4f})")
    print(f"  Keyword Overlap: {np.mean(baseline_keyword_overlaps):.4f} (±{np.std(baseline_keyword_overlaps):.4f})")
    
    print(f"\nRAG wins (higher semantic similarity): {sum(1 for r, b in zip(rag_similarities, baseline_similarities) if r > b)} / {len(rag_similarities)}")