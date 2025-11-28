"""
Local LLM Wrapper using Transformers
Runs models locally without API calls
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from typing import Optional


class LocalLLM:
    """Wrapper for running LLMs locally with transformers."""
    
    # In src/local_llm.py, line 14
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = None):
        """
        Initialize local LLM.
        
        Args:
            model_name: Hugging Face model name
            device: 'cuda' for GPU, 'cpu' for CPU, None for auto-detect
        """
        print(f"Loading model: {model_name}")
        print("This may take a few minutes on first run (downloading ~1GB)...")
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Qwen uses AutoModelForCausalLM, not Seq2Seq
        try:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.is_causal = True
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_causal = False
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✓ Model loaded on {self.device}")
        print(f"  Model: {model_name}")
        print(f"  Parameters: ~{sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M")
        
        self.generation_count = 0
    
    def generate(self, prompt: str, max_tokens: int = 200, 
                 temperature: float = 0.7, num_beams: int = 4) -> Optional[str]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            num_beams: Beam search width (higher = better quality, slower)
            
        Returns:
            Generated text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with no_grad for efficiency
            with torch.no_grad():
                # For causal models (like Qwen), handle differently
                if self.is_causal:
                    if temperature == 0.0:
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=0.95,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                else:
                    # For seq2seq models (like T5)
                    if temperature == 0.0:
                        outputs = self.model.generate(
                            **inputs,
                            max_length=max_tokens,
                            num_beams=num_beams,
                            early_stopping=True
                        )
                    else:
                        outputs = self.model.generate(
                            **inputs,
                            max_length=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=0.95,
                            num_beams=1
                        )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # For causal models, remove the prompt from output
            if self.is_causal:
                # Remove prompt from generated text
                generated_text = generated_text[len(prompt):].strip()
            
            self.generation_count += 1
            
            return generated_text.strip()
        
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return None
    
    def get_stats(self):
        """Get model statistics."""
        return {
            'model': self.model_name,
            'device': self.device,
            'generations': self.generation_count,
            'parameters': f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M"
        }


# Test
if __name__ == "__main__":
    print("Testing Local LLM...")
    
    # Initialize with Qwen
    llm = LocalLLM("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Test questions
    test_prompts = [
        "What is the capital of France?",
        "Explain what attention mechanism is in transformers.",
        "What are the main components of BERT?"
    ]
    
    print("\n" + "="*60)
    print("TESTING GENERATION")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        result = llm.generate(prompt, max_tokens=100, temperature=0.3)
        print(f"Result: {result}")
    
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(llm.get_stats())