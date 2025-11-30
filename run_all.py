import os, subprocess, argparse
from pathlib import Path
import json

# HERE, I AM setting up the main directory structure
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
FIG = ROOT / "figures"
RES = ROOT / "results"
IND = ROOT / "indices"


for d in [FIG, RES, IND]:
    os.makedirs(d, exist_ok=True)

# HERE, I AM defining a utility function to run shell commands and check for errors
def run(cmd):
    print("RUN:", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        # HERE, I AM raising an error if the command fails
        raise RuntimeError(f"Command failed: {cmd}")

if __name__ == "__main__":
    # HERE, I AM setting up the command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/all_movies.csv")
    parser.add_argument("--text_col", default="plot")
    parser.add_argument("--max_docs", type=int, default=10000)
    parser.add_argument("--lsi_dims", type=int, default=200)
    parser.add_argument("--num_perm", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--shingle_n", type=int, default=5)
    args = parser.parse_args()

    # HERE, I AM defining file paths based on arguments
    processed_npz = DATA / "processed.npz"
    emb_lsi = DATA / f"emb_lsi_{args.lsi_dims}.npy"
    pq_joblib = IND / f"pq_M8_Ks256.joblib"
    sketches_file = IND / f"sketches_{args.num_perm}.joblib"
    lsh_job = IND / f"minhash_lsh_{args.num_perm}_thr{int(0.8*100):02d}.joblib"
    faiss_job = IND / "index_faiss.joblib"

    # 1) HERE, I AM running the preprocessing script (tokenization, stopword removal)
    run(f'python "{SRC}/preprocess.py" --csv "{args.csv}" --out "{processed_npz}" '
        f'--text_col "{args.text_col}" --max_docs {args.max_docs}')

    # 2) HERE, I AM generating LSI embeddings (dimensionality reduction) 
    run(f'python "{SRC}/embed.py" --processed_npz "{processed_npz}" '
        f'--method lsi --n_components {args.lsi_dims} --out "{emb_lsi}"')

    # 3) HERE, I AM performing Product Quantization (PQ) for compressed, approximate search 
    run(f'python "{SRC}/quantize.py" --emb "{emb_lsi}" '
        f'--out_prefix "{IND}/pq" --M 8 --Ks 256')
    produced = IND / "pq_pq.joblib"
    if produced.exists():
        produced.rename(pq_joblib)

    # 4) HERE, I AM creating MinHash sketches (Jaccard similarity approximation) and building the LSH index 
    run(f'python "{SRC}/embed.py" --processed_npz "{processed_npz}" '
        f'--method minhash --num_perm {args.num_perm} --shingle_n {args.shingle_n} --out "{sketches_file}"')
    run(f'python "{SRC}/lsh_index.py" --sketches "{sketches_file}" '
        f'--out "{lsh_job}" --num_perm {args.num_perm} --threshold 0.8')

    # 5) HERE, I AM building the FAISS index (exact cosine similarity)
    run(f'python "{SRC}/faiss_hnsw_index.py" --emb "{emb_lsi}" --out_prefix "{IND}/index" --index faiss')

    # 6) HERE, I AM running the benchmark script to evaluate speed and recall of all indices
    metrics_out = RES / "metrics.json"
    run(f'python "{SRC}/benchmark.py" --emb "{emb_lsi}" --processed_npz "{processed_npz}" --pq_joblib "{pq_joblib}" '
        f'--lsh_joblib "{lsh_job}" --faiss_joblib "{faiss_job}" --k {args.k} --n_queries 200 --out "{metrics_out}"')

    # 7) HERE, I AM generating the final plot for comparison
    import matplotlib.pyplot as plt
    with open(metrics_out) as f:
        metrics = json.load(f)

    methods = list(metrics.keys())
    times = [metrics[m]["avg_time"] for m in methods]
    recalls = [metrics[m]["recall"] for m in methods]

    fig, ax = plt.subplots()
    ax.scatter(times, recalls)
    # HERE, I AM annotating the points with the method name
    for i, txt in enumerate(methods):
        ax.annotate(txt, (times[i], recalls[i]))

    # HERE, I AM setting the axis labels and title for the plot
    ax.set_xlabel("avg_time (s)")
    ax.set_ylabel(f"recall@{args.k}")
    ax.set_title("Index comparison")

    # HERE, I AM saving the figure
    out_fig = FIG / "recall_vs_time.png"
    fig.savefig(out_fig, bbox_inches="tight")
    print("DONE. Metrics + figure generated. Figure saved to", out_fig)






