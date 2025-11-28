"""
Comprehensive RAG Retrieval Quality Debugger
============================================
This script helps you diagnose and fix retrieval issues in your RAG system.

Run this to understand:
1. Are the right papers being retrieved?
2. Are chunks too small/large?
3. Is semantic search matching the wrong things?
4. What's the quality of your top-k results?
"""

from retriever import DocumentRetriever
import json
from collections import Counter
from typing import List, Dict

class RetrievalDebugger:
    def __init__(self, qa_dataset_path='data/qa_pairs/qa_dataset.json'):
        self.retriever = DocumentRetriever()
        with open(qa_dataset_path, 'r', encoding='utf-8') as f:
            self.qa_pairs = json.load(f)
    
    def analyze_single_question(self, question: str, expected_paper: str = None, 
                               expected_page: int = None, top_k: int = 5):
        """Deep dive into retrieval for a single question."""
        print("="*80)
        print(f"QUESTION: {question}")
        print("="*80)
        
        if expected_paper:
            print(f"Expected Source: {expected_paper}, page {expected_page}")
        print()
        
        # Retrieve
        chunks = self.retriever.retrieve(question, top_k=top_k)
        
        # Analyze results
        print(f"TOP {top_k} RETRIEVED CHUNKS:")
        print("-"*80)
        
        found_expected = False
        for i, chunk in enumerate(chunks, 1):
            is_correct = (expected_paper and 
                         chunk['paper_file'] == expected_paper and 
                         chunk['page'] == expected_page)
            
            marker = "✅ CORRECT!" if is_correct else ""
            if is_correct:
                found_expected = True
            
            print(f"\n[{i}] {chunk['paper_file']}, page {chunk['page']} "
                  f"(score: {chunk.get('relevance_score', 0):.3f}) {marker}")
            print(f"Text preview: {chunk['text'][:300]}...")
            print(f"Text length: {len(chunk['text'].split())} words")
        
        print("\n" + "="*80)
        if expected_paper:
            if found_expected:
                print("✅ SUCCESS: Correct source found in top-k!")
            else:
                print("❌ FAILURE: Correct source NOT in top-k!")
                print("\nDEBUGGING TIPS:")
                print("- Try increasing top_k (e.g., from 5 to 10)")
                print("- Check if chunks are too small (should be 300-500 words)")
                print("- Consider adding query reformulation")
        print("="*80)
        print()
        
        return chunks, found_expected
    
    def analyze_retrieval_coverage(self, top_k: int = 5):
        """Analyze how often correct sources are retrieved across all QA pairs."""
        print("\n" + "="*80)
        print(f"RETRIEVAL COVERAGE ANALYSIS (top_k={top_k})")
        print("="*80)
        
        total = len(self.qa_pairs)
        hits = 0
        paper_stats = Counter()
        position_stats = Counter()
        
        for qa in self.qa_pairs:
            question = qa['question']
            expected_paper = qa['paper_file']
            expected_page = qa['page_numbers'][0] if qa['page_numbers'] else None
            
            chunks = self.retriever.retrieve(question, top_k=top_k)
            
            # Check if correct source is in results
            for i, chunk in enumerate(chunks, 1):
                if (chunk['paper_file'] == expected_paper and 
                    (expected_page is None or chunk['page'] == expected_page)):
                    hits += 1
                    position_stats[i] += 1
                    break
            else:
                # Track which papers we're retrieving instead
                wrong_papers = [c['paper_file'] for c in chunks]
                paper_stats.update(wrong_papers)
        
        # Results
        recall = (hits / total) * 100
        print(f"\n📊 RESULTS:")
        print(f"Total questions: {total}")
        print(f"Correct source found: {hits} ({recall:.1f}%)")
        print(f"Correct source NOT found: {total - hits} ({100-recall:.1f}%)")
        
        if recall < 70:
            print("\n⚠️  WARNING: Low retrieval recall! This will hurt your RAG performance.")
            print("   Your LLM can't answer correctly if it doesn't see the right context.")
        
        print(f"\n📍 Position of Correct Chunks (when found):")
        for pos in sorted(position_stats.keys()):
            count = position_stats[pos]
            print(f"  Position {pos}: {count} times ({count/hits*100:.1f}%)")
        
        if position_stats:
            avg_position = sum(pos * count for pos, count in position_stats.items()) / hits
            print(f"  Average position: {avg_position:.1f}")
        
        print(f"\n🔍 Most Common Wrong Papers Retrieved:")
        for paper, count in paper_stats.most_common(5):
            print(f"  {paper}: {count} times")
        
        return recall
    
    def analyze_chunk_quality(self, sample_size: int = 10):
        """Analyze the quality of chunks in your dataset."""
        print("\n" + "="*80)
        print("CHUNK QUALITY ANALYSIS")
        print("="*80)
        
        from chunker import load_chunks  # Adjust import based on your structure
        chunks = load_chunks()
        
        if not chunks:
            print("⚠️  Could not load chunks. Check your chunker.py")
            return
        
        # Sample random chunks
        import random
        sample = random.sample(chunks, min(sample_size, len(chunks)))
        
        word_counts = [len(chunk['text'].split()) for chunk in sample]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        
        print(f"\n📏 Chunk Size Statistics (sample of {len(sample)}):")
        print(f"  Average: {avg_words:.0f} words")
        print(f"  Min: {min_words} words")
        print(f"  Max: {max_words} words")
        
        if avg_words < 200:
            print("\n⚠️  WARNING: Chunks are too small!")
            print("   Small chunks may split important context across boundaries.")
            print("   RECOMMENDATION: Increase chunk_size to 400-500 words")
        elif avg_words > 600:
            print("\n⚠️  WARNING: Chunks are too large!")
            print("   Large chunks may dilute relevant content with noise.")
            print("   RECOMMENDATION: Decrease chunk_size to 300-400 words")
        else:
            print("\n✅ Chunk size looks good!")
        
        # Check for garbage chunks
        print(f"\n🗑️  Checking for low-quality chunks:")
        garbage_count = 0
        for chunk in sample:
            text = chunk['text'].strip()
            # Heuristics for garbage
            if len(text.split()) < 50:
                garbage_count += 1
            elif text.count('\n') > len(text) / 20:  # Too many line breaks
                garbage_count += 1
        
        garbage_pct = (garbage_count / len(sample)) * 100
        print(f"  Low-quality chunks: {garbage_count}/{len(sample)} ({garbage_pct:.1f}%)")
        
        if garbage_pct > 10:
            print("  ⚠️  WARNING: High number of low-quality chunks!")
            print("  RECOMMENDATION: Review your chunking strategy")
    
    def compare_top_k_values(self, question: str, k_values: List[int] = [3, 5, 10, 15]):
        """Test how retrieval quality changes with different top_k values."""
        print("\n" + "="*80)
        print(f"TOP-K COMPARISON FOR: {question}")
        print("="*80)
        
        for k in k_values:
            chunks = self.retriever.retrieve(question, top_k=k)
            papers = set(c['paper_file'] for c in chunks)
            print(f"\ntop_k={k}:")
            print(f"  Unique papers retrieved: {len(papers)}")
            print(f"  Papers: {', '.join(list(papers)[:3])}{'...' if len(papers) > 3 else ''}")
    
    def run_full_diagnostic(self):
        """Run all diagnostic tests."""
        print("\n" + "🔬" + "="*78 + "🔬")
        print("  FULL RAG RETRIEVAL DIAGNOSTIC")
        print("🔬" + "="*78 + "🔬\n")
        
        # 1. Chunk quality
        try:
            self.analyze_chunk_quality()
        except Exception as e:
            print(f"⚠️  Could not analyze chunk quality: {e}")
        
        # 2. Retrieval coverage
        recall_5 = self.analyze_retrieval_coverage(top_k=5)
        recall_10 = self.analyze_retrieval_coverage(top_k=10)
        
        # 3. Sample question deep dive
        print("\n" + "="*80)
        print("DETAILED EXAMPLE: First Question")
        print("="*80)
        first_qa = self.qa_pairs[0]
        self.analyze_single_question(
            first_qa['question'],
            first_qa['paper_file'],
            first_qa['page_numbers'][0] if first_qa['page_numbers'] else None,
            top_k=10
        )
        
        # 4. Recommendations
        print("\n" + "="*80)
        print("📋 RECOMMENDATIONS")
        print("="*80)
        
        if recall_5 < 60:
            print("\n🚨 CRITICAL: Very low retrieval recall!")
            print("   Actions:")
            print("   1. Increase top_k from 5 to 10 (quick win)")
            print("   2. Rebuild chunks with larger chunk_size (400-500 words)")
            print("   3. Consider hybrid search (semantic + keyword)")
        elif recall_5 < 75:
            print("\n⚠️  MODERATE: Retrieval needs improvement")
            print("   Actions:")
            print("   1. Increase top_k from 5 to 10")
            print("   2. Review chunk quality")
        else:
            print("\n✅ GOOD: Retrieval quality is acceptable")
            print("   If RAG still underperforms, focus on:")
            print("   - Prompt engineering")
            print("   - LLM choice/parameters")
        
        improvement = recall_10 - recall_5
        if improvement > 10:
            print(f"\n💡 QUICK WIN: Increasing top_k from 5 to 10 improves recall by {improvement:.1f}%")
            print("   Do this IMMEDIATELY!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    debugger = RetrievalDebugger()
    
    if len(sys.argv) == 1:
        # Run full diagnostic
        debugger.run_full_diagnostic()
    
    elif sys.argv[1] == "question":
        # Analyze specific question
        if len(sys.argv) < 3:
            print("Usage: python debug_retrieval.py question 'Your question here'")
            sys.exit(1)
        question = sys.argv[2]
        debugger.analyze_single_question(question, top_k=10)
    
    elif sys.argv[1] == "coverage":
        # Just run coverage analysis
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        debugger.analyze_retrieval_coverage(top_k=top_k)
    
    elif sys.argv[1] == "compare":
        # Compare different top_k values
        if len(sys.argv) < 3:
            print("Usage: python debug_retrieval.py compare 'Your question here'")
            sys.exit(1)
        question = sys.argv[2]
        debugger.compare_top_k_values(question)
    
    else:
        print("Unknown command. Available commands:")
        print("  python debug_retrieval.py                    # Full diagnostic")
        print("  python debug_retrieval.py question 'Q...'    # Analyze specific question")
        print("  python debug_retrieval.py coverage [top_k]   # Retrieval coverage")
        print("  python debug_retrieval.py compare 'Q...'     # Compare top_k values")