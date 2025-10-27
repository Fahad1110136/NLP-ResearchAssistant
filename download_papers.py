"""
Paper Download Script
Downloads all 15 papers for the NLP project
"""

import arxiv
import os
import time

def download_papers():
    """Download all papers from arXiv"""
    
    # All 15 papers with arXiv IDs and filenames
    papers = [
        # Your papers (1-8)
        ("1706.03762", "attention_is_all_you_need.pdf"),
        ("1810.04805", "bert.pdf"),
        ("2005.14165", "gpt3.pdf"),
        ("1907.11692", "roberta.pdf"),
        ("1910.10683", "t5.pdf"),
        ("2106.09685", "lora.pdf"),
        ("2201.11903", "chain_of_thought.pdf"),
        ("2212.08073", "constitutional_ai.pdf"),
        
        # Teammate's papers (9-15)
        ("2005.11401", "rag.pdf"),
        ("2302.13971", "llama.pdf"),
        ("2203.02155", "instructgpt.pdf"),
        ("2001.08361", "scaling_laws.pdf"),
        ("2211.05100", "bloom.pdf"),
        ("2204.14198", "flamingo.pdf"),
        ("2212.10560", "self_instruct.pdf"),
    ]
    
    output_dir = "data/papers"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("DOWNLOADING PAPERS FROM ARXIV")
    print("="*60)
    print(f"Output directory: {output_dir}\n")
    
    successful = 0
    failed = []
    
    for idx, (arxiv_id, filename) in enumerate(papers, 1):
        try:
            print(f"[{idx}/15] Downloading: {filename}...", end=" ")
            
            # Search for the paper by ID
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            # Download PDF
            filepath = os.path.join(output_dir, filename)
            paper.download_pdf(dirpath=output_dir, filename=filename)
            
            print(f"✓ SUCCESS")
            successful += 1
            
            # Be nice to arXiv API - add delay
            time.sleep(3)
            
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            failed.append((filename, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successful: {successful}/15")
    print(f"Failed: {len(failed)}/15")
    
    if failed:
        print("\nFailed downloads:")
        for filename, error in failed:
            print(f"  ✗ {filename}: {error}")
        print("\nTry downloading failed papers manually from:")
        print("https://arxiv.org/")
    else:
        print("\n✓ All papers downloaded successfully!")
    
    print("="*60)


if __name__ == "__main__":
    print("\nThis will download 15 papers (~150-200 MB total)")
    print("Estimated time: 3-5 minutes\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        download_papers()
    else:
        print("Download cancelled.")