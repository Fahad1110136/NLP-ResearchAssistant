# evaluation/split_dataset.py
import json
import random

# Load full dataset
with open('data/qa_pairs/qa_dataset.json', 'r') as f:
    qa_pairs = json.load(f)

# Shuffle and split
random.seed(42)
random.shuffle(qa_pairs)

dev_set = qa_pairs[:10]
test_set = qa_pairs[10:]

# Save splits
with open('evaluation/qa_dev.json', 'w') as f:
    json.dump(dev_set, f, indent=2)

with open('evaluation/qa_test.json', 'w') as f:
    json.dump(test_set, f, indent=2)

print(f"✓ Dev set: {len(dev_set)} questions")
print(f"✓ Test set: {len(test_set)} questions")
