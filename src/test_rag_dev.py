"""
Test RAG system on dev set using HF Inference API
"""

import json
import sys
import os
sys.path.append('src')

from rag_qa import RAGQuestionAnswering

# Get HF token
if len(sys.argv) > 1:
    HF_TOKEN = sys.argv[1]
else:
    HF_TOKEN = input("Enter your Hugging Face API token: ").strip()

# Load dev set
with open('evaluation/qa_dev.json', 'r') as f:
    dev_set = json.load(f)

print(f"Loaded {len(dev_set)} dev questions")
print(f"Estimated time: ~{len(dev_set) * 0.1} minutes with API")
print()

# Initialize RAG
rag = RAGQuestionAnswering(HF_TOKEN)

# Answer all questions
results = []
for qa in dev_set:
    result = rag.answer_question(qa['question'])
    
    # Add ground truth for comparison
    result['qa_id'] = qa['id']
    result['ground_truth'] = qa['answer']
    result['expected_paper'] = qa['paper_file']
    result['expected_pages'] = qa['page_numbers']
    
    results.append(result)

# Save results
os.makedirs('evaluation', exist_ok=True)
with open('evaluation/rag_dev_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved {len(results)} results to evaluation/rag_dev_results.json")

# Print summary
print("\n" + "="*70)
print("RAG DEV SET RESULTS")
print("="*70)

total_time = sum(r.get('time_taken', 0) for r in results)
print(f"\nTotal time: {total_time/60:.1f} minutes")
print(f"Average time per question: {total_time/len(results):.1f} seconds")

for r in results:
    print(f"\n{'='*70}")
    print(f"Q{r['qa_id']}: {r['question']}")
    print(f"A: {r['answer']}")
    print(f"Expected: {r['expected_paper']}, pages {r['expected_pages']}")
    print(f"Retrieved: {[f"{s['paper_name']}, p.{s['page']}" for s in r['sources'][:2]]}")
    
    # Check if correct source was retrieved
    correct_source = False
    for source in r['sources'][:5]:
        if (source['paper_file'] == r['expected_paper'] and 
            source['page'] in r['expected_pages']):
            correct_source = True
            break
    
    if correct_source:
        print("✓ Correct source retrieved")
    else:
        print("✗ Correct source not in top 5")

print("\n" + "="*70)
print("DONE!")
print("="*70)