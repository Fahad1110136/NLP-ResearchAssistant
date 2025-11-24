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


# Test
if __name__ == "__main__":
    evaluator = AutomatedEvaluator()
    
    gt = "The Transformer uses self-attention mechanisms to process sequences in parallel."
    good_answer = "Transformers employ self-attention to handle sequences simultaneously."
    bad_answer = "The model uses RNN layers for sequential processing."
    
    print("Good answer metrics:")
    print(evaluator.evaluate(good_answer, gt))
    
    print("\nBad answer metrics:")
    print(evaluator.evaluate(bad_answer, gt))
