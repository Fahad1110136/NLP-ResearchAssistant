import json
import os

BASE_DIR = os.path.dirname(__file__)  # folder where merge.py is located

# --- Load results.json ---
with open(os.path.join(BASE_DIR, "results.json"), "r", encoding="utf-8") as f:
    rag_results = json.load(f)

# --- Load results_2.json ---
with open(os.path.join(BASE_DIR, "results_2.json"), "r", encoding="utf-8") as f:
    baseline_results = json.load(f)

# --- Prepare final merged structure ---
final_output = {
    "Results of RAG Answers": rag_results,
    "Results of BaseLine Answers": baseline_results
}

# --- Save final_results.json ---
with open(os.path.join(BASE_DIR, "final_results.json"), "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print("final_results.json created successfully!")
