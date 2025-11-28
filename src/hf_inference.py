"""
Hugging Face Inference API Wrapper
Uses HF serverless inference for better quality models
"""

from huggingface_hub import InferenceClient
import time
from typing import Optional


class HFInference:
    """Wrapper for Hugging Face Inference API."""
    
    def __init__(self, token: str, model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize HF Inference client.
        
        Args:
            token: Your HF API token
            model: Model to use for inference
        """
        print(f"Initializing Hugging Face Inference API...")
        print(f"Model: {model}")
        
        self.client = InferenceClient(token=token, base_url="https://router.huggingface.co")
        self.model = model
        self.request_count = 0
        self.last_request_time = 0
        
        print("✓ HF Inference API ready")
    
    def generate(self, prompt: str, max_tokens: int = 200, 
                 temperature: float = 0.7, retry_attempts: int = 3) -> Optional[str]:
        """
        Generate text using HF Inference API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            retry_attempts: Number of retries on failure
            
        Returns:
            Generated text or None on failure
        """
        for attempt in range(retry_attempts):
            try:
                # Rate limiting: wait 1 second between requests
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < 1.0:
                    time.sleep(1.0 - time_since_last)
                
                # Use chat_completion for instruction models
                response = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                self.last_request_time = time.time()
                self.request_count += 1
                
                # Extract generated text from chat response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()
                elif isinstance(response, dict) and 'choices' in response:
                    return response['choices'][0]['message']['content'].strip()
                else:
                    return str(response).strip()
            
            except Exception as e:
                error_msg = str(e)
                print(f"API error (attempt {attempt+1}/{retry_attempts}): {error_msg}")
                
                # Handle specific errors
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = 60
                    print(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                elif "loading" in error_msg.lower() or "503" in error_msg:
                    wait_time = 20
                    print(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    if attempt < retry_attempts - 1:
                        time.sleep(5)
                        continue
                    return None
        
        return None
    
    def get_stats(self):
        """Get API usage statistics."""
        return {
            'model': self.model,
            'requests_made': self.request_count
        }


# Test
if __name__ == "__main__":
    import sys
    
    # Get token from command line or replace here
    if len(sys.argv) > 1:
        TOKEN = sys.argv[1]
    else:
        TOKEN = "hf_YOUR_TOKEN_HERE"  # Replace with your token
        if TOKEN == "hf_YOUR_TOKEN_HERE":
            print("Please provide your HF token:")
            print("  python src/hf_inference.py YOUR_TOKEN")
            print("\nOr edit the file and replace TOKEN = 'hf_YOUR_TOKEN_HERE'")
            sys.exit(1)
    
    # Initialize
    hf = HFInference(TOKEN)
    
    # Test prompts
    test_prompts = [
        """Context: The attention function computes the dot product of the query with all keys, divides by the square root of dimension, applies a softmax function to obtain weights, and uses these weights to compute a weighted sum of the values.

Q: What are the three main components of scaled dot-product attention?
A:""",
        """Context: BERT uses two unsupervised pre-training tasks: Masked Language Model (MLM) where 15% of tokens are masked and predicted, and Next Sentence Prediction (NSP) where the model predicts if sentence B follows sentence A.

Q: What are the two pre-training tasks used to train BERT?
A:""",
    ]
    
    print("\n" + "="*70)
    print("TESTING GENERATION")
    print("="*70)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Prompt: {prompt[:100]}...")
        
        result = hf.generate(prompt, max_tokens=100, temperature=0.3)
        
        if result:
            print(f"✓ Result: {result}")
        else:
            print(f"✗ Failed to generate")
    
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(hf.get_stats())