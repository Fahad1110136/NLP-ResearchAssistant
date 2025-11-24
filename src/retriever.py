"""
Retrieval System
High-level interface for document retrieval
"""

from embeddings import EmbeddingManager
from typing import List, Dict, Tuple


class DocumentRetriever:
    """High-level retrieval system for academic papers."""
    
    def __init__(self, index_path="data/processed/faiss_index.bin",
                 chunks_path="data/processed/chunks.json",
                 model_name='all-MiniLM-L6-v2'):
        """
        Initialize the retriever.
        
        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks JSON
            model_name: Embedding model name
        """
        self.embedding_manager = EmbeddingManager(model_name=model_name)
        self.embedding_manager.load(index_path=index_path, chunks_path=chunks_path)
        print("✓ Retriever initialized and ready")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of chunk dicts with scores
        """
        results = self.embedding_manager.search(query, top_k=top_k)
        
        # Add scores to chunk dicts
        chunks_with_scores = []
        for chunk, score in results:
            chunk_copy = chunk.copy()
            chunk_copy['relevance_score'] = score
            chunks_with_scores.append(chunk_copy)
        
        return chunks_with_scores
    
    def retrieve_with_context(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve chunks and format them for generation.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Dict with context and metadata
        """
        chunks = self.retrieve(query, top_k)
        
        # Build context string
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] (from {chunk['paper_name']}, page {chunk['page']})\n"
                f"{chunk['text']}\n"
            )
            
            sources.append({
                'source_id': i,
                'paper_name': chunk['paper_name'],
                'paper_file': chunk['paper_file'],
                'page': chunk['page'],
                'relevance_score': chunk['relevance_score']
            })
        
        context = "\n".join(context_parts)
        
        return {
            'query': query,
            'context': context,
            'sources': sources,
            'chunks': chunks
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the indexed corpus."""
        chunks = self.embedding_manager.chunks
        
        papers = set(c['paper_name'] for c in chunks)
        total_words = sum(c['word_count'] for c in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_papers': len(papers),
            'total_words': total_words,
            'avg_words_per_chunk': total_words / len(chunks) if chunks else 0
        }


def test_retrieval():
    """Test the retrieval system with sample queries."""
    print("Initializing retriever...")
    retriever = DocumentRetriever()
    
    # Get statistics
    stats = retriever.get_statistics()
    print("\n" + "="*60)
    print("CORPUS STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test queries
    test_queries = [
        "What is attention mechanism?",
        "How does BERT work?",
        "What is few-shot learning?",
        "Explain masked language modeling"
    ]
    
    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60)
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Retrieve with context
        result = retriever.retrieve_with_context(query, top_k=3)
        
        print(f"\nRetrieved {len(result['chunks'])} chunks:")
        for source in result['sources']:
            print(f"  [{source['source_id']}] {source['paper_name']}, "
                  f"page {source['page']} (score: {source['relevance_score']:.3f})")


if __name__ == "__main__":
    test_retrieval()