"""
PDF Text Extraction with Page Tracking - Improved Version
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file page by page using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dicts: [{"page": 1, "text": "..."}, ...]
    """
    pages_data = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text with better spacing
            text = page.get_text("text")  # or try "blocks" for better structure
            
            if text and text.strip():
                pages_data.append({
                    "page": page_num + 1,
                    "text": text.strip()
                })
        
        doc.close()
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return []
    
    return pages_data


def process_all_papers(papers_dir="data/papers", output_dir="data/processed"):
    """
    Process all PDFs in the papers directory.
    
    Args:
        papers_dir: Directory containing PDF files
        output_dir: Directory to save processed data
        
    Returns:
        Dict mapping paper names to their page data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = list(Path(papers_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {papers_dir}")
        return {}
    
    print(f"Found {len(pdf_files)} PDF files")
    print("Extracting text from PDFs...")
    
    all_papers = {}
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        paper_name = pdf_path.stem
        
        pages_data = extract_text_from_pdf(pdf_path)
        
        if pages_data:
            all_papers[paper_name] = {
                "filename": pdf_path.name,
                "total_pages": len(pages_data),
                "pages": pages_data
            }
            print(f"✓ {paper_name}: {len(pages_data)} pages")
        else:
            print(f"✗ {paper_name}: Failed to extract text")
    
    output_path = os.path.join(output_dir, "papers_text.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved processed data to {output_path}")
    print(f"Total papers processed: {len(all_papers)}")
    
    return all_papers

def load_processed_papers(processed_path="data/processed/papers_text.json"):
    """Load previously processed papers from JSON."""
    if not os.path.exists(processed_path):
        print(f"Processed data not found at {processed_path}")
        return {}
    
    with open(processed_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    papers = process_all_papers()
    
    if papers:
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        total_pages = sum(p['total_pages'] for p in papers.values())
        print(f"Papers processed: {len(papers)}")
        print(f"Total pages: {total_pages}")
        print("\nPapers:")
        for name, data in papers.items():
            print(f"  - {name}: {data['total_pages']} pages")