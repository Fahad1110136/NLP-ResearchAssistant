"""
Baseline Question Answering System (HF API Version)
Direct generation without retrieval using HF Inference API
"""

import sys
sys.path.append('src')

from hf_inference import HFInference
from typing import Dict, List
import time


class BaselineQuestionAnswering:
    """Baseline QA system without retrieval, using HF Inference API."""
    
    def __init__(self, hf_token: str, 
                 model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize Baseline QA system.
        
        Args:
            hf_token: Hugging Face API token
            model: Model name (should match RAG for fair comparison)
        """
        print("Initializing Baseline QA System...")
        
        # Initialize HF Inference API
        self.hf = HFInference(hf_token, model)
        
        print("✓ Baseline QA System ready")
    
    def build_prompt(self, question: str) -> str:
        """
        Build prompt without any retrieved context.
        
        Args:
            question: User question
            
        Returns:
            Formatted prompt
        """
        # Simple prompt - no context, just question
        prompt = f"""You are a helpful AI assistant. Answer the following question about natural language processing and machine learning research. Be specific and technical.

Question: {question}

Answer:"""
        
        return prompt
    
    def answer_question(self, question: str, max_tokens: int = 200) -> Dict:
        """
        Answer a question using baseline (no retrieval).
        
        Args:
            question: Question to answer
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with answer and metadata
        """
        print(f"\nQuestion: {question}")
        start_time = time.time()
        
        # Build prompt (no retrieval)
        prompt = self.build_prompt(question)
        
        # Generate answer via API
        print("Generating answer via HF API (no retrieval)...")
        answer = self.hf.generate(prompt, max_tokens=max_tokens, temperature=0.3)
        
        if answer is None:
            return {
                'question': question,
                'answer': "Failed to generate answer.",
                'error': 'generation_failed',
                'time_taken': time.time() - start_time
            }
        
        time_taken = time.time() - start_time
        print(f"✓ Generated answer ({len(answer)} chars in {time_taken:.1f}s)")
        
        return {
            'question': question,
            'answer': answer,
            'sources': [],  # No sources in baseline
            'prompt_length': len(prompt),
            'model': self.hf.model,
            'retrieval_used': False,
            'time_taken': time_taken
        }
    
    def answer_batch(self, questions: List[str]) -> List[Dict]:
        """Answer multiple questions."""
        results = []
        total_time = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*70}")
            print(f"Processing question {i}/{len(questions)}")
            print('='*70)
            
            result = self.answer_question(question)
            results.append(result)
            total_time += result.get('time_taken', 0)
            
            # Show progress
            avg_time = total_time / i
            remaining = (len(questions) - i) * avg_time
            print(f"\nProgress: {i}/{len(questions)} ({i/len(questions)*100:.1f}%)")
            print(f"Estimated time remaining: {remaining/60:.1f} minutes")
        
        print(f"\n✓ Completed {len(questions)} questions in {total_time/60:.1f} minutes")
        
        return results


# Test
if __name__ == "__main__":
    import sys
    
    # Get token from command line or replace here
    if len(sys.argv) > 1:
        HF_TOKEN = sys.argv[1]
    else:
        HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # Replace with your token
        if HF_TOKEN == "hf_YOUR_TOKEN_HERE":
            print("Please provide your HF token:")
            print("  python src/baseline_qa.py YOUR_TOKEN")
            print("\nOr edit line 97 and replace HF_TOKEN = 'hf_YOUR_TOKEN_HERE'")
            sys.exit(1)
    
    print("Testing Baseline QA System with HF Inference API...")
    
    # Initialize system
    baseline = BaselineQuestionAnswering(HF_TOKEN)
    
    # Test questions (same as RAG for comparison)
    test_questions = [
        "What are the three main components of the scaled dot-product attention mechanism?",
        "What are the two pre-training tasks used to train BERT?",
        "What is few-shot learning in GPT-3?"
    ]
    
    # Answer questions
    results = baseline.answer_batch(test_questions)
    
    # Print results
    print("\n" + "="*70)
    print("BASELINE RESULTS")
    print("="*70)
    
    for result in results:
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: None (baseline has no retrieval)")
        print(f"Time: {result['time_taken']:.1f}s")