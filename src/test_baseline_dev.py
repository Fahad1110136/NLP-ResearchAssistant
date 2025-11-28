"""
Test Baseline system on dev set using HuggingFace Inference API
Uses Meta Llama 3.2 3B Instruct via API
"""

import json
import sys
import os
sys.path.append('src')

# Use HF Inference API for baseline
from huggingface_hub import InferenceClient
import time

def answer_with_hf_api(question, hf_token, model="meta-llama/Llama-3.2-3B-Instruct"):
    """Answer question using HF Inference API."""
    client = InferenceClient(token=hf_token, base_url="https://router.huggingface.co")
    
    # Build simple prompt (no retrieval)
    prompt = f"""You are a helpful AI assistant. Answer the following question about natural language processing and machine learning research. Be specific and technical.

Question: {question}

Answer:"""
    
    try:
        # Rate limiting
        time.sleep(1)
        
        # Generate using chat_completion
        response = client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )

        return response.choices[0].message["content"].strip()

    
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None


# Get HF token
if len(sys.argv) > 1:
    HF_TOKEN = sys.argv[1]
else:
    HF_TOKEN = input("Enter your Hugging Face API token: ").strip()

# Load dev set
with open('evaluation/qa_dev.json', 'r') as f:
    dev_set = json.load(f)

print(f"Loaded {len(dev_set)} dev questions")
print(f"Using HuggingFace API with meta-llama/Llama-3.2-3B-Instruct")
print(f"Estimated time: ~{len(dev_set) * 0.1} minutes with API")
print()

# Answer all questions
results = []
for i, qa in enumerate(dev_set, 1):
    print(f"\n{'='*70}")
    print(f"Processing question {i}/{len(dev_set)}")
    print('='*70)
    print(f"Question: {qa['question']}")
    
    start_time = time.time()
    
    # Generate answer via API
    answer = answer_with_hf_api(qa['question'], HF_TOKEN)
    
    time_taken = time.time() - start_time
    
    if answer:
        print(f"✓ Generated answer ({len(answer)} chars in {time_taken:.1f}s)")
    else:
        print(f"✗ Failed to generate answer")
        answer = "Failed to generate answer."
    
    # Build result
    result = {
        'qa_id': qa['id'],
        'question': qa['question'],
        'answer': answer,
        'ground_truth': qa['answer'],
        'expected_paper': qa['paper_file'],
        'expected_pages': qa['page_numbers'],
        'sources': [],  # No sources in baseline
        'model': 'meta-llama/Llama-3.2-3B-Instruct',
        'retrieval_used': False,
        'time_taken': time_taken
    }
    
    results.append(result)
    
    print(f"Answer preview: {answer[:100]}...")

# Save results
os.makedirs('evaluation', exist_ok=True)
with open('evaluation/baseline_dev_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved {len(results)} results to evaluation/baseline_dev_results.json")

# Print summary
print("\n" + "="*70)
print("BASELINE DEV SET RESULTS (HF API)")
print("="*70)

total_time = sum(r.get('time_taken', 0) for r in results)
print(f"\nTotal time: {total_time/60:.1f} minutes")
print(f"Average time per question: {total_time/len(results):.1f} seconds")

for r in results:
    print(f"\n{'='*70}")
    print(f"Q{r['qa_id']}: {r['question']}")
    print(f"A: {r['answer']}")
    print(f"Expected: {r['expected_paper']}, pages {r['expected_pages']}")
    print(f"Retrieved: None (baseline)")

print("\n" + "="*70)
print("DONE!")
print("="*70)

# Print summary
print("\n" + "="*70)
print("BASELINE DEV SET RESULTS")
print("="*70)

total_time = sum(r.get('time_taken', 0) for r in results)
print(f"\nTotal time: {total_time/60:.1f} minutes")
print(f"Average time per question: {total_time/len(results):.1f} seconds")

for r in results:
    print(f"\n{'='*70}")
    print(f"Q{r['qa_id']}: {r['question']}")
    print(f"A: {r['answer']}")
    print(f"Expected: {r['expected_paper']}, pages {r['expected_pages']}")
    print(f"Retrieved: None (baseline)")

print("\n" + "="*70)
print("DONE!")
print("="*70)