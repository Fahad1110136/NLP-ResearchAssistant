# 📚 Multi-Source Academic Research Assistant

# — RAG-based Question Answering over Academic Papers

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about NLP research papers. The system retrieves relevant passages from a local corpus of academic PDFs and passes them as context to a language model to generate grounded answers. It is evaluated against a baseline (no-retrieval) QA system across 100 manually curated question-answer pairs.

---

## Project Overview

The project is built around two QA systems that are compared head-to-head:

- **RAG System** — retrieves the top relevant text chunks from the paper corpus using hybrid search (semantic + keyword), then feeds them as context into a language model prompt.
- **Baseline System** — sends the question directly to the same language model with no retrieved context, relying solely on the model's parametric knowledge.

Both systems use **meta-llama/Llama-3.2-3B-Instruct** as the language model, either via the Hugging Face Inference API or loaded locally.

---

## Pipeline

### 1. PDF Parsing (`src/pdf_processor.py`)

All research papers are stored as PDFs in `data/papers/`. The PDF processor reads each file page by page using PyMuPDF (`fitz`) and extracts the text from every page. Each page's text is stored as a dictionary with the page number and content. The output is saved to `data/processed/papers_text.json`.

### 2. Text Chunking (`src/chunker.py`)

The extracted text is split into overlapping chunks using a sliding window. Each chunk is 300 words long with a 50-word overlap between consecutive chunks. Every chunk is stored with metadata: the source paper name, filename, page number, and chunk index. The chunks are saved to `data/processed/chunks.json`. After the first run, chunk size was increased and embeddings were regenerated to improve retrieval quality.

### 3. Embedding Generation (`src/embeddings.py`)

Each chunk is encoded into a dense vector using the `all-MiniLM-L6-v2` sentence-transformers model. The embeddings are L2-normalized and stored in a FAISS flat index (`data/processed/faiss_index.bin`) for efficient similarity search. The raw embeddings are also saved to `data/processed/embeddings.npy`.

### 4. Retrieval (`src/retriever.py` and `src/hybrid_retriever.py`)

Given a query, the `DocumentRetriever` encodes it with the same sentence-transformers model and searches the FAISS index for the nearest neighbors by cosine similarity.

The `HybridRetriever` wraps the base retriever and combines semantic similarity scores with a keyword overlap score. The final score for each chunk is a weighted combination: 70% semantic score and 30% keyword score. Keywords are extracted by removing stopwords from the query and matching them against chunk text, including bigram and trigram phrase matches. The top 15 chunks are retrieved per query.

### 5. RAG Question Answering (`src/rag_qa.py` and `src/local_inference.py`)

The top 3 retrieved chunks are formatted as numbered excerpts and placed into a structured prompt. The prompt instructs the model to synthesize information across excerpts and answer directly. Generation is handled either through the Hugging Face Inference API (`src/hf_inference.py`) or a locally loaded Llama model (`src/local_inference.py`). The local inference path was added after the HF API token limits were hit during evaluation.

The local inference class loads `meta-llama/Llama-3.2-3B-Instruct` using the `transformers` library, uses `float16` if a GPU is available and `float32` on CPU, and applies the model's chat template before generating.

### 6. Baseline Question Answering (`src/baseline_qa.py`)

The baseline system sends the question directly to the same Llama model without any retrieved context. The prompt asks the model to answer a question about NLP and ML research based on its own knowledge.

### 7. Evaluation Dataset (`data/qa_pairs/`, `evaluation/`)

A dataset of 100 question-answer pairs was manually created from the paper corpus. Questions are drawn from papers such as *Attention Is All You Need*, *BERT*, *Chain-of-Thought Prompting*, *Scaling Laws for Neural Language Models*, and others. Each QA pair includes the source paper, page numbers, question type (factual, conceptual, comparative), and difficulty level (easy, medium, hard).

The dataset is split into:
- `evaluation/qa_dev.json` — development set used for tuning and debugging
- `evaluation/qa_test.json` — held-out test set

### 8. Automated Evaluation (`evaluation/automated_metrics.py`, `evaluation/run_evaluation.py`)

Both QA systems are evaluated on all 100 questions using three automated metrics:

- **Semantic Similarity** — cosine similarity between the model's answer and the ground-truth answer, computed using `all-MiniLM-L6-v2` embeddings.
- **Keyword Overlap** — fraction of ground-truth keywords (after stopword removal) that appear in the model's answer.
- **Length Ratio** — ratio of the shorter answer length to the longer one, to measure verbosity alignment.

Results are saved to `evaluation/results.json`, `evaluation/full_results.json`, and compiled into `evaluation/final_results.csv` and `evaluation/final_results.xlsx` for analysis.

### 9. Manual Evaluation (`evaluation/evaluation_guidelines.md`, `evaluation/manual_eval_template.csv`)

Human evaluators assess each answer on four dimensions, each scored 0–2:

- **Factual Accuracy** — whether the answer is correct and consistent with the ground truth.
- **Completeness** — whether all parts of the question are addressed.
- **Faithfulness to Retrieved Evidence** (RAG only) — whether the answer is grounded in the retrieved excerpts without unsupported claims.
- **Safety and Appropriateness** — whether the response avoids harmful or inappropriate content.

---

## Evaluation Results (Automated — 100 Questions)

| Metric | RAG System | Baseline |
|---|---|---|
| Semantic Similarity | 0.7417 | 0.6820 |
| Keyword Overlap | 0.2767 | 0.2508 |

The RAG system outperforms the baseline on both metrics. On the development set, RAG showed approximately 14% improvement over baseline after tuning the retrieval mechanism (increasing top-k from 5 to 15 chunks and adjusting prompts).

---

## Project Structure

```
NLP-ResearchAssistant/
├── data/
│   ├── papers/                     # Input PDF research papers
│   ├── processed/
│   │   ├── papers_text.json        # Extracted text from PDFs
│   │   ├── chunks.json             # Text chunks with metadata
│   │   ├── embeddings.npy          # Chunk embeddings (numpy)
│   │   └── faiss_index.bin         # FAISS vector index
│   └── qa_pairs/
│       ├── qa_dataset.json         # Full QA dataset
│       ├── qa_dataset_hassaan.json # QA pairs (contributor 1)
│       └── qa_dataset_fahad.json   # QA pairs (contributor 2)
├── evaluation/
│   ├── qa_dev.json                 # Dev split
│   ├── qa_test.json                # Test split
│   ├── automated_metrics.py        # Semantic sim, keyword overlap, length ratio
│   ├── run_evaluation.py           # Evaluation runner
│   ├── run_evaluation_2.py         # Second evaluation pass
│   ├── results.json                # Intermediate results
│   ├── final_results.json          # Full evaluation results
│   ├── final_results.csv           # Results in CSV format
│   ├── final_results.xlsx          # Results in Excel format
│   ├── json_to_csv.py              # Converts JSON results to CSV
│   ├── merge.py                    # Merges result files
│   ├── evaluation_guidelines.md    # Manual evaluation rubric
│   └── manual_eval_template.csv    # Manual scoring spreadsheet
├── src/
│   ├── pdf_processor.py            # PDF text extraction
│   ├── chunker.py                  # Text chunking
│   ├── embeddings.py               # Embedding generation and FAISS index
│   ├── retriever.py                # Semantic retrieval
│   ├── hybrid_retriever.py         # Hybrid semantic + keyword retrieval
│   ├── rag_qa.py                   # RAG QA system (HF API)
│   ├── baseline_qa.py              # Baseline QA system (HF API)
│   ├── local_inference.py          # Local Llama inference
│   ├── local_llm.py                # Local LLM utilities
│   ├── hf_inference.py             # Hugging Face Inference API wrapper
│   ├── compare_dev_results.py      # Dev set comparison script
│   ├── debug_retrieval.py          # Retrieval debugging utilities
│   └── diagnose_retriever_format.py
├── run_full_evaluation.py          # Main evaluation script (local model)
├── evaluation_dataset.py           # Dataset preparation utilities
├── requirements.txt
└── .gitignore
```

---

## Requirements

```
Python 3.9+
PyMuPDF>=1.23.0
sentence-transformers>=2.3.0
transformers>=4.36.0
torch>=2.1.0
accelerate==0.25.0
faiss-cpu>=1.7.4
huggingface_hub[hf_xet]==0.26.0
scikit-learn>=1.3.2
tqdm>=4.66.1
numpy>=1.26.0
pandas>=2.1.3
streamlit>=1.28.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup and Usage

### 1. Add papers

Place PDF research papers in `data/papers/`.

### 2. Parse PDFs

```bash
python src/pdf_processor.py
```

### 3. Create chunks

```bash
python src/chunker.py
```

### 4. Generate embeddings

```bash
python src/embeddings.py
```

### 5. Run evaluation (local model)

Requires Hugging Face access to `meta-llama/Llama-3.2-3B-Instruct`. Log in first:

```bash
huggingface-cli login
```

Then run:

```bash
python run_full_evaluation.py
```

Results are saved to `evaluation/full_results.json` and compiled into `evaluation/final_results.csv`.

---

## Notes

- The project switched from the Hugging Face Inference API to local model inference (`src/local_inference.py`) during evaluation due to API token limits.
- Chunk size was increased (from a smaller default) and embeddings were regenerated after initial retrieval quality was found to be insufficient.
- The retriever was tuned to retrieve top 15 chunks per query (up from 5), with only the top 3 used in the final prompt, which produced the best results on the development set.
