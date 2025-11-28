"""
Diagnostic: Check Retriever Output Format
==========================================
This script helps debug why matching is failing.
"""

from retriever import DocumentRetriever
import json

# Load retriever and test data
retriever = DocumentRetriever()

with open('data/qa_pairs/qa_dataset.json', 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)

# Test with first question
first_qa = qa_pairs[0]
question = first_qa['question']

print("="*80)
print("DIAGNOSTIC: Checking Retriever Output Format")
print("="*80)

print(f"\nTest Question: {question}")
print(f"\nExpected from QA dataset:")
print(f"  paper_file: {first_qa['paper_file']}")
print(f"  page_numbers: {first_qa['page_numbers']}")

# Retrieve
print(f"\nRetrieving top 5 chunks...")
chunks = retriever.retrieve(question, top_k=5)

print(f"\nActual chunk structure returned:")
print("-"*80)

for i, chunk in enumerate(chunks[:3], 1):  # Show first 3
    print(f"\nChunk {i} keys: {chunk.keys()}")
    print(f"Chunk {i} content:")
    for key, value in chunk.items():
        if key == 'text':
            print(f"  {key}: {str(value)[:100]}...")
        else:
            print(f"  {key}: {value}")

print("\n" + "="*80)
print("COMPARISON:")
print("="*80)

# Try different possible field names
possible_paper_fields = ['paper_name', 'paper_file', 'source', 'document', 'filename', 'paper']
possible_page_fields = ['page', 'page_number', 'page_num']

print("\nLooking for paper identifier field...")
for field in possible_paper_fields:
    if field in chunks[0]:
        print(f"  ✅ Found: '{field}' = {chunks[0][field]}")
    else:
        print(f"  ❌ Not found: '{field}'")

print("\nLooking for page number field...")
for field in possible_page_fields:
    if field in chunks[0]:
        print(f"  ✅ Found: '{field}' = {chunks[0][field]}")
    else:
        print(f"  ❌ Not found: '{field}'")

print("\n" + "="*80)
print("MATCHING TEST:")
print("="*80)

chunk = chunks[0]
expected = first_qa['paper_file']

# Test different matching strategies
print(f"\nExpected paper: '{expected}'")

if 'paper_name' in chunk:
    print(f"Chunk paper_name: '{chunk['paper_name']}'")
    print(f"Exact match: {chunk['paper_name'] == expected}")
    print(f"Case-insensitive match: {chunk['paper_name'].lower() == expected.lower()}")
    
    # Check if it's a path vs filename issue
    import os
    expected_basename = os.path.basename(expected)
    chunk_basename = os.path.basename(chunk['paper_name'])
    print(f"\nBasename comparison:")
    print(f"  Expected basename: '{expected_basename}'")
    print(f"  Chunk basename: '{chunk_basename}'")
    print(f"  Match: {expected_basename == chunk_basename}")

if 'paper_file' in chunk:
    print(f"Chunk paper_file: '{chunk['paper_file']}'")
    print(f"Exact match: {chunk['paper_file'] == expected}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)

# Give recommendation based on findings
if 'paper_name' in chunks[0]:
    paper_field = 'paper_name'
elif 'paper_file' in chunks[0]:
    paper_field = 'paper_file'
else:
    paper_field = list(chunks[0].keys())[0]  # Just use first available

if 'page' in chunks[0]:
    page_field = 'page'
elif 'page_number' in chunks[0]:
    page_field = 'page_number'
else:
    page_field = None

print(f"\nUse these fields in analyze_retrieval_coverage:")
print(f"  Paper field: chunk['{paper_field}']")
if page_field:
    print(f"  Page field: chunk['{page_field}']")
else:
    print(f"  Page field: NOT FOUND (may need to fix retriever)")

print("\nMatching strategy:")
if 'paper_name' in chunks[0] and 'paper_file' in first_qa:
    chunk_name = chunks[0]['paper_name']
    expected_name = first_qa['paper_file']
    
    if chunk_name == expected_name:
        print("  ✅ Direct comparison works!")
    else:
        import os
        if os.path.basename(chunk_name) == os.path.basename(expected_name):
            print("  ⚠️  Use basename comparison: os.path.basename()")
        elif chunk_name.lower() == expected_name.lower():
            print("  ⚠️  Use case-insensitive comparison: .lower()")
        else:
            print(f"  ❌ Names don't match even with basename/case-insensitive")
            print(f"     You may need to normalize paper names in your data")