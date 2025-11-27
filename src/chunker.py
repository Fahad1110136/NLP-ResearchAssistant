"""
Text Chunking with Metadata Preservation
Splits long texts into smaller chunks while maintaining context
"""

import json
import os
from typing import List, Dict


def chunk_text(text, chunk_size=300, overlap=50):
    """
    Split text into overlapping chunks by words.
    
    Args:
        text: Text to chunk
        chunk_size: Target number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        # Move start position with overlap
        start = end - overlap
        
        # Break if remaining words are too few
        if end >= len(words):
            break
    
    return chunks


def create_chunks_from_papers(papers_data, chunk_size=300, overlap=50):
    """
    Create chunks from all papers with metadata.
    
    Args:
        papers_data: Dict of papers from pdf_processor
        chunk_size: Words per chunk
        overlap: Overlapping words
        
    Returns:
        List of chunk dicts with metadata
    """
    all_chunks = []
    chunk_id = 0
    
    for paper_name, paper_info in papers_data.items():
        for page_data in paper_info['pages']:
            page_num = page_data['page']
            page_text = page_data['text']
            
            # Create chunks for this page
            text_chunks_list = chunk_text(page_text, chunk_size, overlap)
            
            # Add metadata to each chunk
            for chunk_idx, chunk_content in enumerate(text_chunks_list):
                all_chunks.append({
                    'chunk_id': chunk_id,
                    'paper_name': paper_name,
                    'paper_file': paper_info['filename'],
                    'page': page_num,
                    'chunk_index': chunk_idx,
                    'text': chunk_content,
                    'word_count': len(chunk_content.split())
                })
                chunk_id += 1
    
    return all_chunks


def save_chunks(chunks, output_path="data/processed/chunks.json"):
    """Save chunks to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(chunks)} chunks to {output_path}")


def load_chunks(chunks_path="data/processed/chunks.json"):
    """Load chunks from JSON file."""
    if not os.path.exists(chunks_path):
        print(f"Chunks file not found at {chunks_path}")
        return []
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_chunk_statistics(chunks):
    """Print statistics about the chunks."""
    if not chunks:
        print("No chunks to analyze")
        return
    
    total_chunks = len(chunks)
    papers = set(c['paper_name'] for c in chunks)
    avg_words = sum(c['word_count'] for c in chunks) / total_chunks
    
    print("\n" + "="*60)
    print("CHUNKING STATISTICS")
    print("="*60)
    print(f"Total chunks: {total_chunks}")
    print(f"Papers: {len(papers)}")
    print(f"Average words per chunk: {avg_words:.1f}")
    
    # Chunks per paper
    print("\nChunks per paper:")
    from collections import Counter
    paper_counts = Counter(c['paper_name'] for c in chunks)
    for paper, count in sorted(paper_counts.items()):
        print(f"  {paper}: {count} chunks")


if __name__ == "__main__":
    # Load processed papers
    from pdf_processor import load_processed_papers
    
    print("Loading processed papers...")
    papers = load_processed_papers()
    
    if not papers:
        print("No papers found. Run pdf_processor.py first!")
        exit(1)
    
    print(f"Loaded {len(papers)} papers")
    
    # Create chunks
    print("\nCreating chunks...")
    chunks = create_chunks_from_papers(papers, chunk_size=500, overlap=100)
    
    # Save chunks
    save_chunks(chunks)
    
    # Print statistics
    get_chunk_statistics(chunks)