"""
RAG Question Answering System (HF API Version)
Combines retrieval with Hugging Face Inference API
"""

import sys
sys.path.append('src')

from retriever import DocumentRetriever
from hybrid_retriever import HybridRetriever
from hf_inference import HFInference
from typing import Dict, List
import time


class RAGQuestionAnswering:
    """RAG-based question answering system with HF Inference API."""
    
    def __init__(self, hf_token: str, 
                 model: str = "meta-llama/Llama-3.2-3B-Instruct",
                 top_k: int = 5):
        """
        Initialize RAG QA system.
        
        Args:
            hf_token: Hugging Face API token
            model: Model name on HF
            top_k: Number of chunks to retrieve
        """
        print("Initializing RAG QA System...")
        
        # Initialize retriever
        print("\nLoading retrieval system...")
        base_retriever = DocumentRetriever()
        self.retriever = HybridRetriever(base_retriever, alpha=0.7)
        
        # Initialize HF Inference API
        print("\nConnecting to Hugging Face API...")
        self.hf = HFInference(hf_token, model)
        
        self.top_k = top_k
        
        print("\n✓ RAG QA System ready")
    
    def build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Build prompt with retrieved context.
        
        Args:
            question: User question
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Formatted prompt
        """
        # Build context from retrieved chunks (use top 3 for better quality)
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
        
        return prompt
    
    def answer_question(self, question: str, max_tokens: int = 200) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: Question to answer
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with answer, sources, and metadata
        """
        print(f"\nQuestion: {question}")
        start_time = time.time()
        
        # Step 1: Retrieve relevant chunks
        print(f"Retrieving top {self.top_k} relevant passages...")
        chunks = self.retriever.retrieve(question, top_k=15)
        
        if not chunks:
            return {
                'question': question,
                'answer': "No relevant information found.",
                'sources': [],
                'error': 'retrieval_failed',
                'time_taken': time.time() - start_time
            }
        
        print(f"✓ Retrieved {len(chunks)} passages")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"  [{i}] {chunk['paper_name']}, page {chunk['page']} (score: {chunk.get('relevance_score', 0):.3f})")
        
        # Step 2: Build prompt
        prompt = self.build_prompt(question, chunks)
        
        # Step 3: Generate answer via API
        print("Generating answer via HF API...")
        answer = self.hf.generate(prompt, max_tokens=max_tokens, temperature=0.3)
        
        if answer is None:
            return {
                'question': question,
                'answer': "Failed to generate answer.",
                'sources': chunks,
                'error': 'generation_failed',
                'time_taken': time.time() - start_time
            }
        
        time_taken = time.time() - start_time
        print(f"✓ Generated answer ({len(answer)} chars in {time_taken:.1f}s)")
        
        # Step 4: Format sources
        sources = []
        for chunk in chunks:
            sources.append({
                'paper_name': chunk['paper_name'],
                'paper_file': chunk['paper_file'],
                'page': chunk['page'],
                'relevance_score': chunk.get('relevance_score', 0.0)
            })
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'retrieved_chunks': chunks,
            'prompt_length': len(prompt),
            'model': self.hf.model,
            'time_taken': time_taken
        }
    
    def answer_batch(self, questions: List[str]) -> List[Dict]:
        """Answer multiple questions."""
        results = []
        total_time = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*70}")
            print(f"Processing question {i}/{len(questions)}")
            print('='*70)
            
            result = self.answer_question(question)
            results.append(result)
            total_time += result.get('time_taken', 0)
            
            # Show progress
            avg_time = total_time / i
            remaining = (len(questions) - i) * avg_time
            print(f"\nProgress: {i}/{len(questions)} ({i/len(questions)*100:.1f}%)")
            print(f"Estimated time remaining: {remaining/60:.1f} minutes")
        
        print(f"\n✓ Completed {len(questions)} questions in {total_time/60:.1f} minutes")
        
        return results
    
    def reformulate_query(self, question: str) -> List[str]:
        """
        Generate multiple query variations to improve retrieval.

        Returns list of queries to search with.
        """
        queries = [question]  # Original
        
        # Add simplified version (remove question words)
        simplified = question.lower()
        for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who']:
            simplified = simplified.replace(word, '')
        simplified = ' '.join(simplified.split())  # Clean whitespace
        if simplified != question:
            queries.append(simplified)
        
        # Add keyword extraction (nouns/important terms only)
        import re
        words = re.findall(r'\\b[a-z]+\\b', question.lower())
        stopwords = {'what', 'are', 'the', 'is', 'how', 'does', 'do', 'a', 'an'}
        keywords = ' '.join([w for w in words if w not in stopwords and len(w) > 3])
        if keywords and keywords not in queries:
            queries.append(keywords)
        
        return queries


    def answer(self, question: str) -> str:
        """Answer question using RAG with multi-query retrieval."""
        
        # Get multiple query variations
        queries = self.reformulate_query(question)
        
        # Retrieve with each query and combine results
        all_chunks = []
        seen_ids = set()
        
        for query in queries:
            chunks = self.retriever.retrieve(query, top_k=10)
            for chunk in chunks:
                if chunk['chunk_id'] not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk['chunk_id'])
        
        # Take top 15 unique chunks by score
        all_chunks.sort(key=lambda x: x.get('hybrid_score', x.get('relevance_score', 0)), 
                        reverse=True)
        final_chunks = all_chunks[:15]
        
        # Continue with generation...
        context = self._format_context(final_chunks)
        answer = self._generate(context, question)
        return answer


# Test
if __name__ == "__main__":
    import sys
    
    # Get token from command line or replace here
    if len(sys.argv) > 1:
        HF_TOKEN = sys.argv[1]
    else:
        HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # Replace with your token
        if HF_TOKEN == "hf_YOUR_TOKEN_HERE":
            print("Please provide your HF token:")
            print("  python src/rag_qa.py YOUR_TOKEN")
            print("\nOr edit line 138 and replace HF_TOKEN = 'hf_YOUR_TOKEN_HERE'")
            sys.exit(1)
    
    print("Testing RAG QA System with HF Inference API...")
    
    # Initialize system
    rag = RAGQuestionAnswering(HF_TOKEN)
    
    # Test questions
    test_questions = [
        "What are the three main components of the scaled dot-product attention mechanism?",
        "What are the two pre-training tasks used to train BERT?",
        "What is few-shot learning in GPT-3?"
    ]
    
    # Answer questions
    results = rag.answer_batch(test_questions)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for result in results:
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {[f"{s['paper_name']}, p.{s['page']}" for s in result['sources'][:3]]}")
        print(f"Time: {result['time_taken']:.1f}s")