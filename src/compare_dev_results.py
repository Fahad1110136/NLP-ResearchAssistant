"""
Compare RAG vs Baseline on dev set
Shows side-by-side comparison and calculates metrics
"""

import json
import sys
import os

# Add evaluation folder to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evaluation_path = os.path.join(project_root, "evaluation")
sys.path.append(evaluation_path)

# Try to import automated evaluator if it exists
try:
    from automated_metrics import AutomatedEvaluator
    evaluator = AutomatedEvaluator()
    has_evaluator = True
except ImportError as e:
    has_evaluator = False
    print("Note: automated_metrics.py not found, skipping semantic similarity")
    print(f"(Import error: {e})")

# Load results
try:
    with open('evaluation/rag_dev_results.json', 'r') as f:
        rag_results = json.load(f)
    print(f"✓ Loaded {len(rag_results)} RAG results")
except FileNotFoundError:
    print("❌ Error: evaluation/rag_dev_results.json not found!")
    print("Run: python test_rag_dev.py first")
    sys.exit(1)

try:
    with open('evaluation/baseline_dev_results.json', 'r') as f:
        baseline_results = json.load(f)
    print(f"✓ Loaded {len(baseline_results)} Baseline results")
except FileNotFoundError:
    print("❌ Error: evaluation/baseline_dev_results.json not found!")
    print("Run: python test_baseline_dev.py first")
    sys.exit(1)

print("\n" + "="*80)
print("RAG vs BASELINE COMPARISON - DEV SET")
print("="*80)

# Calculate metrics
rag_scores = []
baseline_scores = []

for rag, baseline in zip(rag_results, baseline_results):
    assert rag['qa_id'] == baseline['qa_id'], f"Mismatched questions! RAG: {rag['qa_id']}, Baseline: {baseline['qa_id']}"
    
    # Calculate automated metrics if available
    if has_evaluator:
        rag_metrics = evaluator.evaluate(rag['answer'], rag['ground_truth'])
        baseline_metrics = evaluator.evaluate(baseline['answer'], baseline['ground_truth'])
        
        rag_scores.append(rag_metrics['semantic_similarity'])
        baseline_scores.append(baseline_metrics['semantic_similarity'])
    
    # Print comparison
    print(f"\n{'='*80}")
    print(f"Question {rag['qa_id']}: {rag['question']}")
    print('='*80)
    
    print(f"\n📖 Ground Truth:")
    print(f"  {rag['ground_truth'][:200]}...")
    
    print(f"\n🔍 RAG Answer:")
    print(f"  {rag['answer'][:200]}...")
    if 'sources' in rag and rag['sources']:
        sources = [f"{s.get('paper_name', 'unknown')}, p.{s.get('page', '?')}" for s in rag['sources'][:3]]
        print(f"  📚 Sources: {sources}")
    else:
        print(f"  📚 Sources: None retrieved")
    
    if has_evaluator:
        print(f"  📊 Semantic Similarity: {rag_metrics['semantic_similarity']:.3f}")
    
    print(f"\n⚡ Baseline Answer:")
    print(f"  {baseline['answer'][:200]}...")
    print(f"  📚 Sources: None (no retrieval)")
    
    if has_evaluator:
        print(f"  📊 Semantic Similarity: {baseline_metrics['semantic_similarity']:.3f}")
    
    # Manual quality indicators
    rag_has_sources = 'sources' in rag and len(rag['sources']) > 0
    rag_longer = len(rag['answer']) > len(baseline['answer'])
    
    print(f"\n💡 Quick Assessment:")
    print(f"  RAG has sources: {'✅ Yes' if rag_has_sources else '❌ No'}")
    print(f"  RAG more detailed: {'✅ Yes' if rag_longer else '❌ No'}")

# Summary statistics
print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

if has_evaluator and rag_scores:
    rag_avg = sum(rag_scores) / len(rag_scores)
    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    
    print(f"\n📈 Average Semantic Similarity:")
    print(f"  RAG:      {rag_avg:.3f}")
    print(f"  Baseline: {baseline_avg:.3f}")
    print(f"  Difference: {rag_avg - baseline_avg:+.3f}")
    
    if rag_avg > baseline_avg:
        improvement = ((rag_avg - baseline_avg) / baseline_avg) * 100
        print(f"\n✅ RAG is {improvement:.1f}% better than Baseline!")
    else:
        print(f"\n⚠️  Baseline performing better - check RAG implementation")

# Time comparison
rag_time = sum(r.get('time_taken', 0) for r in rag_results)
baseline_time = sum(r.get('time_taken', 0) for r in baseline_results)

print(f"\n⏱️  Average Response Time:")
print(f"  RAG:      {rag_time/len(rag_results):.2f}s per question")
print(f"  Baseline: {baseline_time/len(baseline_results):.2f}s per question")

# Source attribution
rag_with_sources = sum(1 for r in rag_results if r.get('sources') and len(r['sources']) > 0)
print(f"\n📚 Source Attribution:")
print(f"  RAG:      {rag_with_sources}/{len(rag_results)} questions ({rag_with_sources/len(rag_results)*100:.1f}%)")
print(f"  Baseline: 0/{len(baseline_results)} questions (0.0%)")

# Answer length comparison
rag_avg_len = sum(len(r['answer']) for r in rag_results) / len(rag_results)
baseline_avg_len = sum(len(r['answer']) for r in baseline_results) / len(baseline_results)

print(f"\n📝 Average Answer Length:")
print(f"  RAG:      {rag_avg_len:.0f} characters")
print(f"  Baseline: {baseline_avg_len:.0f} characters")

# Save comparison summary
comparison = {
    'rag_avg_similarity': rag_avg if has_evaluator and rag_scores else None,
    'baseline_avg_similarity': baseline_avg if has_evaluator and baseline_scores else None,
    'improvement': rag_avg - baseline_avg if has_evaluator and rag_scores else None,
    'rag_with_sources': rag_with_sources,
    'rag_avg_time': rag_time / len(rag_results),
    'baseline_avg_time': baseline_time / len(baseline_results),
    'num_questions': len(rag_results)
}

with open('evaluation/dev_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\n✓ Saved comparison summary to evaluation/dev_comparison.json")

# Final verdict
print("\n" + "="*80)
print("🎯 FINAL VERDICT")
print("="*80)

if has_evaluator and rag_scores and rag_avg > baseline_avg:
    print(f"✅ RAG system is clearly superior!")
    print(f"   - {improvement:.1f}% better semantic similarity")
    print(f"   - {rag_with_sources/len(rag_results)*100:.0f}% questions have source citations")
    print(f"   - Ready to proceed to full 100-question evaluation")
elif rag_with_sources > len(rag_results) * 0.7:
    print(f"✅ RAG system shows advantage in source attribution")
    print(f"   - {rag_with_sources/len(rag_results)*100:.0f}% questions have sources")
    print(f"   - Provides transparency that baseline lacks")
    print(f"   - Proceed with full evaluation")
else:
    print(f"⚠️  Results inconclusive - review individual answers")
    print(f"   - Check if retrieval is finding relevant passages")
    print(f"   - Verify both systems use same model")

print("="*80)