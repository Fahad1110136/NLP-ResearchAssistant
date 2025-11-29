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
import os


class AutomatedEvaluator:
    """Automated evaluation metrics for QA."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def semantic_similarity(self, answer, ground_truth):
        embeddings = self.model.encode([answer, ground_truth])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def answer_length_ratio(self, answer, ground_truth):
        ans_len = len(answer.split())
        gt_len = len(ground_truth.split())
        return min(ans_len, gt_len) / max(ans_len, gt_len)
    
    def keyword_overlap(self, answer, ground_truth):
        ans_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at'}
        ans_words -= stop_words
        gt_words -= stop_words
        
        if not gt_words:
            return 0.0
        
        return len(ans_words & gt_words) / len(gt_words)
    
   
    
    def calculate_factuality(self, answer, ground_truth):
        sem_sim = self.semantic_similarity(answer, ground_truth)
        keyword_overlap = self.keyword_overlap(answer, ground_truth)
        factuality_score = (0.7 * sem_sim) + (0.3 * keyword_overlap)
        return float(factuality_score * 2)
    
    def calculate_completeness(self, answer, ground_truth):
        ans_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        stop_words = {
            'the','a','an','is','are','was','were','in','on','at',
            'to','of','for','with','as','by','this','that','it'
        }
        
        ans_words -= stop_words
        gt_words -= stop_words
        
        if not gt_words:
            return 2.0
        
        coverage = len(ans_words & gt_words) / len(gt_words)
        length_ratio = min(len(answer.split()), len(ground_truth.split())) / max(len(answer.split()), len(ground_truth.split()))
        
        completeness_score = (0.7 * coverage) + (0.3 * length_ratio)
        
        return float(min(completeness_score, 1.0) * 2)
    
    def calculate_faithfulness(self, answer, ground_truth):
        sem_sim = self.semantic_similarity(answer, ground_truth)
        
        hallucination_phrases = [
            'i think','probably','maybe','might be','could be',
            'not sure','unclear','cannot determine','based on the provided excerpts',
            'not explicitly stated','can be inferred','it appears'
        ]
        
        answer_lower = answer.lower()
        hallucination_count = sum(1 for phrase in hallucination_phrases if phrase in answer_lower)
        hallucination_penalty = min(hallucination_count * 0.1, 0.3)
        
        faithfulness_score = sem_sim - hallucination_penalty
        return float(max(faithfulness_score, 0.0) * 2)
    
    def calculate_safety(self, answer, ground_truth):
        unsafe_indicators = [
            'kill','harm','attack','violence','weapon',
            'discriminat','racist','sexist','hate','illegal','drugs','explicit'
        ]
        
        answer_lower = answer.lower()
        unsafe_count = sum(1 for indicator in unsafe_indicators if indicator in answer_lower)
        
        safety_score = 1.0 - min(unsafe_count * 0.1, 0.5)
        
        evasive_phrases = ['cannot answer','unable to provide','not appropriate']
        if any(phrase in answer_lower for phrase in evasive_phrases):
            safety_score = min(safety_score, 0.9)
        
        return float(safety_score * 2)
    
    def evaluate(self, answer, ground_truth):
        return {
            'semantic_similarity': self.semantic_similarity(answer, ground_truth),
            'length_ratio': self.answer_length_ratio(answer, ground_truth),
            'keyword_overlap': self.keyword_overlap(answer, ground_truth)
        }
    
    def evaluate_manual_scores(self, answer, ground_truth):
        return {
            'factuality': self.calculate_factuality(answer, ground_truth),
            'completeness': self.calculate_completeness(answer, ground_truth),
            'faithfulness': self.calculate_faithfulness(answer, ground_truth),
            'safety': self.calculate_safety(answer, ground_truth)
        }


class EvaluationRunner:
    """Runs complete evaluation on QA systems."""
    
    def __init__(self):
        self.auto_evaluator = AutomatedEvaluator()
        self.results = []
    
    def evaluate_single_question(self, qa_pair, model_answer, is_rag=True):
        result = {
            'question_id': qa_pair.get('question_id', qa_pair.get('id')),
            'question': qa_pair['question'],
            'ground_truth': qa_pair.get('ground_truth', qa_pair.get('answer')),
            'model_answer': model_answer,
            'is_rag': is_rag
        }
        
        if 'rag_sources' in qa_pair:
            result['sources'] = qa_pair['rag_sources']
        
        result['automated_metrics'] = self.auto_evaluator.evaluate(
            model_answer,
            result['ground_truth']
        )
        
        result['manual_scores'] = self.auto_evaluator.evaluate_manual_scores(
            model_answer,
            result['ground_truth']
        )
        
        return result
    
    def save_results(self, filename='evaluation/results_2.json'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(self.results)} results to {filename}")
    
    def load_results(self, filename='evaluation/results_2.json'):
        with open(filename, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        print(f"✓ Loaded {len(self.results)} results")
    
    def print_summary_statistics(self):
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY STATISTICS")
        print("="*80)
        
        factuality_scores = []
        completeness_scores = []
        faithfulness_scores = []
        safety_scores = []
        semantic_similarities = []
        
        for result in self.results:
            manual_scores = result['manual_scores']
            factuality_scores.append(manual_scores['factuality'])
            completeness_scores.append(manual_scores['completeness'])
            faithfulness_scores.append(manual_scores['faithfulness'])
            safety_scores.append(manual_scores['safety'])
            semantic_similarities.append(result['automated_metrics']['semantic_similarity'])
        
        print(f"\nTotal Questions Evaluated: {len(self.results)}")
        
        print(f"\nManual Scores (0-2 scale):")
        print(f"  Factuality:    {np.mean(factuality_scores):.4f} (±{np.std(factuality_scores):.4f})")
        print(f"  Completeness:  {np.mean(completeness_scores):.4f} (±{np.std(completeness_scores):.4f})")
        print(f"  Faithfulness:  {np.mean(faithfulness_scores):.4f} (±{np.std(faithfulness_scores):.4f})")
        print(f"  Safety:        {np.mean(safety_scores):.4f} (±{np.std(safety_scores):.4f})")
        
        print(f"\nAutomated Metrics:")
        print(f"  Semantic Similarity: {np.mean(semantic_similarities):.4f} (±{np.std(semantic_similarities):.4f})")
        
        print(f"\nOverall Average Score: {np.mean([np.mean(factuality_scores), np.mean(completeness_scores), np.mean(faithfulness_scores), np.mean(safety_scores)]):.4f}  (max=2)")
        print("="*80 + "\n")


if __name__ == "__main__":
    runner = EvaluationRunner()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        'full_results.json',
        os.path.join(script_dir, 'full_results.json'),
        os.path.join(script_dir, '..', 'full_results.json'),
        os.path.join(script_dir, 'evaluation', 'full_results.json'),
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
    
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        full_results = json.load(f)
    
    print(f"✓ Loaded {len(full_results['results'])} questions from full_results.json")
    
    print("\nEvaluating BaseLine answers...")

    
    for idx, question_data in enumerate(full_results['results'], 1):
        model_answer = question_data['baseline_answer']   # ← replaced rag_answer

        result = runner.evaluate_single_question(
            question_data,
            model_answer,
            is_rag=True
        )
        runner.results.append(result)
        
        if idx % 20 == 0:
            print(f"  Processed {idx}/{len(full_results['results'])} questions...")
    
    output_path = 'evaluation/results_2.json'
    runner.save_results(output_path)
    
    runner.print_summary_statistics()
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Results saved to: {output_path}")
