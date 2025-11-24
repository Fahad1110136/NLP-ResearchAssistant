# run_evaluation.py
"""
Evaluation runner - integrates with retrieval and generation
Person 1 will provide the actual QA functions later
"""

import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


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


class EvaluationRunner:
    """Runs complete evaluation on QA systems."""
    
    def __init__(self):
        self.auto_evaluator = AutomatedEvaluator()
        self.results = []
    
    def evaluate_single_question(self, qa_pair, model_answer, is_rag=True):
        """Evaluate one question-answer pair."""
        
        result = {
            'question_id': qa_pair['id'],
            'question': qa_pair['question'],
            'ground_truth': qa_pair['answer'],
            'model_answer': model_answer,
            'expected_paper': qa_pair['paper_file'],
            'expected_pages': qa_pair['page_numbers'],
            'is_rag': is_rag
        }
        
        # Automated metrics
        auto_metrics = self.auto_evaluator.evaluate(
            model_answer, 
            qa_pair['answer']
        )
        result['automated_metrics'] = auto_metrics
        
        # Placeholder for manual scores (filled later)
        result['manual_scores'] = {
            'factuality': None,
            'completeness': None,
            'faithfulness': None,
            'safety': None
        }
        
        return result
    
    def save_results(self, filename='evaluation/results.json'):
        """Save evaluation results."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved {len(self.results)} results to {filename}")
    
    def load_results(self, filename='evaluation/results.json'):
        """Load evaluation results."""
        with open(filename, 'r') as f:
            self.results = json.load(f)
        print(f"✓ Loaded {len(self.results)} results")

if __name__ == "__main__":
    runner = EvaluationRunner()
    
    # Load dev set
    with open('evaluation/qa_dev.json', 'r') as f:
        dev_set = json.load(f)
    
    print(f"Loaded {len(dev_set)} dev questions")
    
    # Evaluate each question with placeholder answers
    for qa_pair in dev_set:
        model_answer = "This is a placeholder answer."  # Replace with actual model answer later
        result = runner.evaluate_single_question(qa_pair, model_answer)
        runner.results.append(result)
    
    # Save results
    runner.save_results('evaluation/results.json')
    
    print("Ready for integration with QA systems!")
    