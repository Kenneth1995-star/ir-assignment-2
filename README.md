1. Project Overview

This repository contains the code for Information Retrieval Assignment 2, focusing on the implementation and benchmarking of various indexing techniques for high-dimensional similarity search. The core goal is to compare the speed (query time) and accuracy (Recall@K) of a custom Product Quantization (PQ) index, a custom Locality-Sensitive Hashing (LSH) index, and a state-of-the-art index from the FAISS library.

The pipeline processes document texts, converts them into dense vectors, builds the three different index structures, and finally runs a detailed comparative benchmark.

2. Setup and Installation

A. Clone the Repository

Bash

git clone https://github.com/Kenneth1995-star/ir-assignment-2.git
cd ir-assignment-2

B. Dependencies

This project requires Python 3.x and the following libraries. It is highly recommended to use a virtual environment.
Bash

# Install required Python packages (from requirements.txt)
pip install pandas numpy scikit-learn gensim datasketch joblib matplotlib

# FAISS installation
# NOTE: FAISS is required. For the CPU version:
# conda install -c conda-forge faiss-cpu
# OR if using pip (ensure you have the necessary system dependencies):
pip install faiss-cpu

C. Data

Place your raw data file (e.g., wikipedia-movies.csv or a similar dataset) into the data/ folder. The provided scripts assume the file is named data/all_movies.csv and the text column is 'plot'.

3. Running the Project (Pipeline)

The entire pipeline, from data preparation to final benchmarking and plotting, is orchestrated by the run_all.py script.

A. Default Run

This command runs the full pipeline with default parameters: LSI embeddings (200 dimensions), 10,000 documents, and benchmarks Recall@10.
Bash

python run_all.py \
    --csv data/all_movies.csv \
    --text_col plot \
    --max_docs 10000 \
    --lsi_dims 200 \
    --num_perm 128 \
    --k 10

B. Parameter Sweep (For Detailed Analysis)

The param_sweep.py script is used to systematically test different combinations of LSH parameters (e.g., number of permutations num_perm and similarity threshold) to analyze the speed-accuracy trade-off.
Bash

python param_sweep.py

4. Pipeline Elaboration

The run_all.py script executes the following steps:
Step	Script	Description	Output Location
1. Preprocess	src/preprocess.py	Loads the raw CSV, performs tokenization, stop-word removal, and saves token lists and joined text.	results/processed.npz
2. Embed	src/embed.py	Generates 200D LSI vectors from the preprocessed text (using TF-IDF + Truncated SVD). MinHash sketches are also generated internally for LSH.	results/emb_lsi.npy
3. PQ Index	src/quantize.py	Builds the custom Product Quantization index (M=8 sub-vectors, Ks​=256 centroids).	indices/index_pq.joblib
4. LSH Index	src/lsh_index.py	Builds the MinHashLSH index using the pre-generated sketches (e.g., L=128 permutations, t=0.8).	indices/index_lsh.joblib
5. FAISS Index	src/faiss_hnsw_index.py	Builds the FAISS index. (Note: The default run uses the brute-force IndexFlatIP as the Ground Truth).	indices/index_faiss.joblib
6. Benchmark	src/benchmark.py	Measures the query time and Recall@K for PQ, LSH, and FAISS against the Brute-Force/GT neighbors.	results/metrics.json
7. Plot	(Internal Logic)	Generates a scatter plot comparing the Recall vs. Query Time for all methods.	figures/comparison.png

5. Directory Structure

ir-assignment-2/
├── src/
│   ├── preprocess.py       # Data cleaning and tokenization
│   ├── embed.py            # LSI, Word2Vec, MinHash generation
│   ├── quantize.py         # Custom Product Quantization implementation
│   ├── lsh_index.py        # MinHash LSH implementation
│   ├── faiss_hnsw_index.py # FAISS integration (HNSW or Brute-Force GT)
│   ├── benchmark.py        # Evaluation of all indexing methods
│   ├── param_sweep.py      # Script to sweep parameters for analysis
│   └── run_all.py          # Main execution pipeline
├── data/
│   └── all_movies.csv      # Placeholder for the dataset
├── results/
│   ├── processed.npz       # Preprocessed data (tokens, joined text)
│   ├── emb_lsi.npy         # Dense LSI vectors
│   └── metrics.json        # Final benchmark results
├── indices/
│   ├── index_pq.joblib     # Product Quantization index files
│   ├── index_lsh.joblib    # LSH index structure
│   └── index_faiss.joblib  # FAISS index file
└── README.md# ir-assignment-2