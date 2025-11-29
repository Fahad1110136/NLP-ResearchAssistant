"""
Full evaluation with LOCAL models - NO API LIMITS!
"""

import json
import sys
import time
from pathlib import Path
sys.path.append('src')

# Import LOCAL versions (no API needed!)
from sentence_transformers import SentenceTransformer
import numpy as np

# Copy-paste the RAGQuestionAnsweringLocal and BaselineQuestionAnsweringLocal classes here
# OR import them if you saved them to separate files

from retriever import DocumentRetriever
from hybrid_retriever import HybridRetriever
from typing import Dict, List


class RAGQuestionAnsweringLocal:
    """RAG with LOCAL Llama-3.2-3B-Instruct"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", top_k=15):
        print("  Setting up retrieval system...")
        
        base_retriever = DocumentRetriever()
        self.retriever = HybridRetriever(base_retriever, alpha=0.7)
        
        self.top_k = top_k
        self.last_sources = []
        self.llm = None  # Will be set externally to share model
        
        print("  ✓ RAG retrieval ready")
    
    def build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Build prompt with retrieved context.
        
        Args:
            question: User question
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Formatted prompt
        """
        # Build context from top 3 chunks (retrieve 15, use 3 in prompt)
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            # Truncate to 600 chars for larger chunks
            text = chunk['text']
            if len(text) > 600:
                text = text[:600] + "..."
            
            context_parts.append(f"Excerpt {i}: {text}")
        
        context = "\n\n".join(context_parts)
        
        # More explicit prompt to avoid confusion
        prompt = f"""You are a research assistant answering questions about NLP papers.

Below are relevant excerpts from research papers. Use them to answer the question.

IMPORTANT INSTRUCTIONS:
1. Synthesize information across multiple excerpts
2. If the exact answer isn't stated, infer from related context
3. For "what" questions about components/parts, look for lists or descriptions
4. Be confident - don't say "not mentioned" unless truly no relevant info exists

EXCERPTS:
{context}

QUESTION: {question}

ANSWER (be direct and confident):"""
        
        print(f"  Prompt length: {len(prompt)} chars, ~{len(prompt)//4} tokens")
        
        return prompt
    
    def answer_question(self, question: str, max_tokens=300) -> Dict:
        start_time = time.time()

        # Reformulate query
        queries = [question]
        
        # Simplified
        simplified = question.lower()
        for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'are', 'is', 'does', 'do']:
            simplified = simplified.replace(word, '')
        simplified = ' '.join(simplified.split())
        if simplified and simplified != question.lower():
            queries.append(simplified)
        
        # Keywords
        import re
        words = re.findall(r'\b[a-z]{4,}\b', question.lower())
        stopwords = {'what', 'how', 'why', 'when', 'where', 'which', 'who', 'are', 'the', 'is', 'does', 'do', 'this', 'that', 'with', 'from'}
        keywords = ' '.join([w for w in words if w not in stopwords])
        if keywords and keywords not in queries:
            queries.append(keywords)
        
        # Retrieve from all queries
        all_chunks = []
        seen_chunk_ids = set()
        
        for q in queries:
            retrieved = self.retriever.retrieve(q, top_k=10)
            for chunk in retrieved:
                cid = chunk.get('chunk_id')
                if cid is not None:
                    if cid not in seen_chunk_ids:
                        all_chunks.append(chunk)
                        seen_chunk_ids.add(cid)
                else:
                    # Fallback: use paper+page as ID
                    key = (chunk['paper_file'], chunk['page'])
                    if key not in seen_chunk_ids:
                        all_chunks.append(chunk)
                        seen_chunk_ids.add(key)
        
        # Sort by score and take top 15
        all_chunks.sort(key=lambda x: x.get('hybrid_score', x.get('relevance_score', 0)), reverse=True)
        chunks = all_chunks[:15]  # REMOVED the duplicate line!
        
        if not chunks:
            return {
                'question': question,
                'answer': "No relevant information found.",
                'sources': [],
                'time_taken': time.time() - start_time
            }
        
        prompt = self.build_prompt(question, chunks)
        answer = self.llm.generate(prompt, max_tokens=max_tokens, temperature=0.3)  # Added temperature
        
        sources = []
        for chunk in chunks[:15]:
            sources.append({
                'paper_name': chunk['paper_name'],
                'paper_file': chunk['paper_file'],
                'page': chunk['page'],
                'relevance_score': chunk.get('hybrid_score', 0.0)
            })
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'time_taken': time.time() - start_time
        }


class BaselineQuestionAnsweringLocal:
    """Baseline with LOCAL Llama-3.2-3B-Instruct"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        print("  Setting up baseline...")
        self.llm = None  # Will be set externally to share model
        print("  ✓ Baseline ready")
    
    def answer_question(self, question: str, max_tokens=200) -> Dict:
        start_time = time.time()
        
        prompt = f"""Answer this question about NLP research:

Question: {question}

Answer:"""
        
        answer = self.llm.generate(prompt, max_tokens=max_tokens)
        
        return {
            'question': question,
            'answer': answer,
            'time_taken': time.time() - start_time
        }


def load_progress(filepath='evaluation/progress.json'):
    if Path(filepath).exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed': 0, 'results': []}

def save_progress(data, filepath='evaluation/progress.json'):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


print("="*80)
print("FULL EVALUATION - LOCAL MODELS (NO API LIMITS!)")
print("="*80)

# Load QA dataset
with open('data/qa_pairs/qa_dataset.json', 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)

total_questions = len(qa_pairs)
print(f"\nTotal questions: {total_questions}")

# Ask how many
response = input(f"\nEvaluate all {total_questions}? (y/n, or number): ").strip()
if response.lower() == 'n':
    num = int(input("How many? "))
    qa_pairs = qa_pairs[:num]
    total_questions = len(qa_pairs)
elif response.isdigit():
    qa_pairs = qa_pairs[:int(response)]
    total_questions = len(qa_pairs)

print(f"Will evaluate {total_questions} questions")

# Load progress
progress = load_progress()
completed = progress['completed']
results = progress['results']

if completed > 0:
    print(f"\nFound progress: {completed} completed")
    if input("Continue? (y/n): ").lower() != 'y':
        completed = 0
        results = []

# Initialize LOCAL systems (this takes 5 min first time)
print("\nInitializing LOCAL Llama model (shared between RAG and Baseline)...")
from local_inference import LocalInference
shared_llm = LocalInference("meta-llama/Llama-3.2-3B-Instruct")

print("\nInitializing LOCAL RAG...")
rag = RAGQuestionAnsweringLocal()
rag.llm = shared_llm  # Use shared model

print("\nInitializing LOCAL Baseline...")
baseline = BaselineQuestionAnsweringLocal()
baseline.llm = shared_llm  # Use shared model

print("\nLoading similarity model...")
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✓ All systems ready\n")

# Process questions
start_time = time.time()

for i in range(completed, total_questions):
    qa = qa_pairs[i]
    question = qa['question']
    ground_truth = qa['answer']
    
    print(f"[{i+1}/{total_questions}] {question[:60]}...")
    
    try:
        # RAG
        print("  RAG...", end=" ", flush=True)
        rag_result = rag.answer_question(question)
        rag_answer = rag_result['answer']
        rag_sources = rag_result.get('sources', [])
        print("✓")
        
        # Baseline
        print("  Baseline...", end=" ", flush=True)
        baseline_result = baseline.answer_question(question)
        baseline_answer = baseline_result['answer']
        print("✓")
        
        # Similarities
        print("  Similarity...", end=" ", flush=True)
        
        rag_emb = similarity_model.encode(rag_answer)
        gt_emb = similarity_model.encode(ground_truth)
        rag_sim = float(np.dot(rag_emb, gt_emb) / 
                       (np.linalg.norm(rag_emb) * np.linalg.norm(gt_emb)))
        
        base_emb = similarity_model.encode(baseline_answer)
        base_sim = float(np.dot(base_emb, gt_emb) / 
                        (np.linalg.norm(base_emb) * np.linalg.norm(gt_emb)))
        
        print("✓")
        
        # Store
        result = {
            'question_id': i + 1,
            'question': question,
            'ground_truth': ground_truth,
            'rag_answer': rag_answer,
            'rag_sources': [f"{s['paper_name']}, p.{s['page']}" for s in rag_sources[:3]],
            'rag_similarity': rag_sim,
            'baseline_answer': baseline_answer,
            'baseline_similarity': base_sim,
            'difference': rag_sim - base_sim,
            'rag_wins': rag_sim > base_sim
        }
        
        results.append(result)
        
        # Save every 10
        if (i + 1) % 10 == 0:
            progress = {'completed': i + 1, 'results': results}
            save_progress(progress)
            
            rag_wins = sum(1 for r in results if r['rag_wins'])
            avg_rag = np.mean([r['rag_similarity'] for r in results])
            avg_base = np.mean([r['baseline_similarity'] for r in results])
            improvement = ((avg_rag - avg_base) / avg_base) * 100
            
            elapsed = time.time() - start_time
            per_q = elapsed / (i + 1 - completed)
            remaining = (total_questions - i - 1) * per_q / 60
            
            print(f"\n  Progress: {i+1}/{total_questions} ({(i+1)/total_questions*100:.1f}%)")
            print(f"  RAG: {avg_rag:.3f}, Baseline: {avg_base:.3f}, Improvement: {improvement:+.1f}%")
            print(f"  Win rate: {rag_wins}/{i+1} ({rag_wins/(i+1)*100:.1f}%)")
            print(f"  Time: {elapsed/60:.1f}min, Est remaining: {remaining:.1f}min\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved.")
        save_progress({'completed': i, 'results': results})
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: {e}")
        result = {
            'question_id': i + 1,
            'question': question,
            'error': str(e),
            'rag_similarity': 0.0,
            'baseline_similarity': 0.0,
            'difference': 0.0,
            'rag_wins': False
        }
        results.append(result)
        save_progress({'completed': i + 1, 'results': results})
        
        if input("Continue? (y/n): ").lower() != 'y':
            break

# Final stats
save_progress({'completed': len(results), 'results': results})

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)

valid = [r for r in results if 'error' not in r]

if valid:
    rag_scores = [r['rag_similarity'] for r in valid]
    base_scores = [r['baseline_similarity'] for r in valid]
    
    avg_rag = np.mean(rag_scores)
    avg_base = np.mean(base_scores)
    improvement = ((avg_rag - avg_base) / avg_base) * 100
    rag_wins = sum(1 for r in valid if r['rag_wins'])
    
    print(f"\n📊 RESULTS:")
    print(f"  Questions: {len(valid)}")
    print(f"  RAG: {avg_rag:.3f}")
    print(f"  Baseline: {avg_base:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  RAG wins: {rag_wins}/{len(valid)} ({rag_wins/len(valid)*100:.1f}%)")
    
    # Save
    final = {
        'summary': {
            'total_questions': len(valid),
            'rag_avg_similarity': float(avg_rag),
            'baseline_avg_similarity': float(avg_base),
            'improvement_percent': float(improvement),
            'rag_wins': int(rag_wins),
            'rag_win_rate': float(rag_wins/len(valid)*100)
        },
        'results': results
    }
    
    with open('evaluation/full_results.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved to: evaluation/full_results.json")
    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} min")

print("\n✅ Done!")