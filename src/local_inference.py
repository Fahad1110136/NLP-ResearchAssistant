"""
Local Model Inference for Llama-3.2-3B-Instruct
Requires HuggingFace login for gated model access
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalInference:
    """Run Llama-3.2-3B locally - requires HF login first"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize Llama model.
        
        Before running:
        1. Request access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
        2. Run: huggingface-cli login
        3. Paste your HF token
        
        Args:
            model_name: "meta-llama/Llama-3.2-3B-Instruct" (6.4GB)
        """
        print(f"Loading model: {model_name}")
        print("(First time will download ~6.4GB)")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model (may take a few minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.model.to(self.device)
        
        print(f"✓ Llama loaded on {self.device}")
    
    def generate(self, prompt, max_tokens=200, temperature=0.3):
        """Generate text from prompt."""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Check prompt length
            input_tokens = len(self.tokenizer.encode(formatted_prompt))
            if input_tokens > 4096:
                print(f"Warning: Prompt too long ({input_tokens} tokens), truncating...")
            
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with error handling
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            print(f"  Generated {len(outputs[0])} total tokens, input was {inputs['input_ids'].shape[1]} tokens")
            
            # Decode only new tokens
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            print(f"  Extracted text: {len(generated_text)} chars")
            
            return generated_text.strip()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM! Try reducing max_length or batch size")
                # Clear cache and retry with shorter context
                torch.cuda.empty_cache()
                return self.generate(prompt[:1000], max_tokens=max_tokens//2, temperature=temperature)
            raise
        
        except Exception as e:
            print(f"Generation error: {e}")
            return ""


if __name__ == "__main__":
    print("Testing Llama-3.2-3B local inference...")
    print("\nMake sure you've run: huggingface-cli login")
    
    # Initialize
    llm = LocalInference("meta-llama/Llama-3.2-3B-Instruct")
    
    # Test
    prompt = "What is the capital of France?"
    answer = llm.generate(prompt)
    
    print(f"\nQ: {prompt}")
    print(f"A: {answer}")
    
    print("\n✓ Llama working locally!")