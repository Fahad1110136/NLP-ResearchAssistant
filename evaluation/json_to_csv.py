import json
import csv
import os

# Load JSON data
print("Loading JSON data...")
with open('evaluation/final_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("✓ JSON loaded successfully!")

# Debug: Check the structure
print("\nTop-level keys in JSON:")
for key in data.keys():
    print(f"  - '{key}'")

# Try to handle different possible structures
if 'Results of RAG Answers' in data:
    rag_key = 'Results of RAG Answers'
    baseline_key = 'Results of BaseLine Answers'
elif 'results_of_rag_answers' in data:
    rag_key = 'results_of_rag_answers'
    baseline_key = 'results_of_baseline_answers'
elif 'rag_answers' in data:
    rag_key = 'rag_answers'
    baseline_key = 'baseline_answers'
else:
    print("\n✗ Could not find RAG answers key!")
    print("Please check the structure of your JSON file.")
    exit()

print(f"\nUsing keys: '{rag_key}' and '{baseline_key}'")

# Prepare CSV file
print("Creating CSV file...")
with open('evaluation/final_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'Question ID',
        'Question',
        'Ground Truth',
        'RAG Answer',
        'RAG Semantic Similarity',
        'RAG Length Ratio',
        'RAG Keyword Overlap',
        'RAG Factuality',
        'RAG Completeness',
        'RAG Faithfulness',
        'RAG Safety',
        'RAG Sources',
        'Baseline Answer',
        'Baseline Semantic Similarity',
        'Baseline Length Ratio',
        'Baseline Keyword Overlap',
        'Baseline Factuality',
        'Baseline Completeness',
        'Baseline Faithfulness',
        'Baseline Safety'
    ]
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Process RAG and Baseline answers
    rag_answers = {item['question_id']: item for item in data[rag_key]}
    baseline_answers = {item['question_id']: item for item in data[baseline_key]}
    
    print(f"Processing {len(rag_answers)} questions...")
    
    # Write rows
    for question_id in sorted(rag_answers.keys()):
        rag = rag_answers[question_id]
        baseline = baseline_answers[question_id]
        
        # Format sources
        rag_sources = ', '.join(rag['sources']) if rag['sources'] else ''
        
        row = {
            'Question ID': rag['question_id'],
            'Question': rag['question'],
            'Ground Truth': rag['ground_truth'],
            'RAG Answer': rag['model_answer'],
            'RAG Semantic Similarity': rag['automated_metrics']['semantic_similarity'],
            'RAG Length Ratio': rag['automated_metrics']['length_ratio'],
            'RAG Keyword Overlap': rag['automated_metrics']['keyword_overlap'],
            'RAG Factuality': rag['manual_scores']['factuality'],
            'RAG Completeness': rag['manual_scores']['completeness'],
            'RAG Faithfulness': rag['manual_scores']['faithfulness'],
            'RAG Safety': rag['manual_scores']['safety'],
            'RAG Sources': rag_sources,
            'Baseline Answer': baseline['model_answer'],
            'Baseline Semantic Similarity': baseline['automated_metrics']['semantic_similarity'],
            'Baseline Length Ratio': baseline['automated_metrics']['length_ratio'],
            'Baseline Keyword Overlap': baseline['automated_metrics']['keyword_overlap'],
            'Baseline Factuality': baseline['manual_scores']['factuality'],
            'Baseline Completeness': baseline['manual_scores']['completeness'],
            'Baseline Faithfulness': baseline['manual_scores']['faithfulness'],
            'Baseline Safety': baseline['manual_scores']['safety']
        }
        
        writer.writerow(row)

print(f"\n✓ CSV file 'output.csv' created successfully!")
print(f"✓ Total rows written: {len(rag_answers)}")