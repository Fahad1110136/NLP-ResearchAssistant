"""
Embeddings and Vector Index Creation
Creates semantic embeddings and builds FAISS index for retrieval
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingManager:
    """Manages embeddings and FAISS index for document retrieval."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = None
        
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def create_embeddings(self, chunks, batch_size=32):
        """
        Create embeddings for all chunks.
        
        Args:
            chunks: List of chunk dicts
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✓ Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of embeddings
        """
        print("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index (using L2 distance on normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        
        print(f"✓ Built FAISS index with {self.index.ntotal} vectors")
    
    def save(self, embeddings, chunks, 
             index_path="data/processed/faiss_index.bin",
             embeddings_path="data/processed/embeddings.npy",
             chunks_path="data/processed/chunks.json"):
        """Save index, embeddings, and chunks to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Store chunks in the object for later use
        self.chunks = chunks
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"✓ Saved FAISS index to {index_path}")
        
        # Save embeddings
        np.save(embeddings_path, embeddings)
        print(f"✓ Saved embeddings to {embeddings_path}")
        
        # Save chunks (already saved by chunker, but save again for consistency)
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved chunks to {chunks_path}")
    
    def load(self, index_path="data/processed/faiss_index.bin",
             chunks_path="data/processed/chunks.json"):
        """Load index and chunks from disk."""
        
        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded FAISS index: {self.index.ntotal} vectors")
        
        # Load chunks
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"✓ Loaded {len(self.chunks)} chunks")
    
    def search(self, query, top_k=5):
        """
        Search for most relevant chunks.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Convert L2 distance to similarity score (0 to 1)
                similarity = 1 / (1 + distance)
                results.append((chunk, float(similarity)))
        
        return results


def build_index_pipeline(chunks_path="data/processed/chunks.json",
                        model_name='all-MiniLM-L6-v2'):
    """
    Complete pipeline to build and save the index.
    
    Args:
        chunks_path: Path to chunks JSON file
        model_name: Sentence transformer model name
    """
    # Load chunks
    print("Loading chunks...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Initialize embedding manager
    manager = EmbeddingManager(model_name=model_name)
    
    # Create embeddings
    embeddings = manager.create_embeddings(chunks)
    
    # Build index
    manager.build_faiss_index(embeddings)
    
    # Save everything
    manager.save(embeddings, chunks)
    
    print("\n" + "="*60)
    print("INDEX BUILDING COMPLETE")
    print("="*60)
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Embedding dimension: {manager.embedding_dim}")
    print("Ready for retrieval!")
    
    return manager


if __name__ == "__main__":
    # Build the complete index
    manager = build_index_pipeline()
    
    # Test with a sample query
    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60)
    
    test_query = "What is attention mechanism in transformers?"
    print(f"\nQuery: {test_query}")
    
    results = manager.search(test_query, top_k=3)
    
    print(f"\nTop 3 results:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Paper: {chunk['paper_name']}, Page: {chunk['page']}")
        print(f"   Text preview: {chunk['text'][:150]}...")