# scripts/param_sweep.py
import os, subprocess, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
FIG = ROOT / "figures" / "sweeps"
RES = ROOT / "results"
IND = ROOT / "indices"
os.makedirs(FIG, exist_ok=True)

# Better sweep choices for MinHash LSH
num_perms = [128, 256, 512]
thresholds = [0.45, 0.55, 0.65]   # lower thresholds produce more candidates
lsi_dims = 200
k = 10
shingle_n = 3

metrics = []
for num_perm in num_perms:
    for thr in thresholds:
        print("Running sweep:", num_perm, thr)
        proc_npz = DATA / "processed.npz"
        emb_lsi = DATA / f"emb_lsi_{lsi_dims}.npy"
        sketches_file = IND / f"sketches_{num_perm}.joblib"
        cmd1 = f'python "{SRC / "embed.py"}" --processed_npz "{proc_npz}" --method minhash --num_perm {num_perm} --shingle_n {shingle_n} --out "{sketches_file}"'
        subprocess.run(cmd1, shell=True, check=True)
        lsh_job = IND / f"minhash_lsh_{num_perm}_thr{int(thr*100)}.joblib"
        cmd2 = f'python "{SRC / "lsh_index.py"}" --sketches "{sketches_file}" --out "{lsh_job}" --num_perm {num_perm} --threshold {thr}'
        subprocess.run(cmd2, shell=True, check=True)
        pq_joblib = IND / "pq_M8_Ks256.joblib"
        faiss_job = IND / "index_faiss.joblib"
        metrics_out = RES / f"metrics_np{num_perm}_thr{int(thr*100)}.json"
        cmd3 = f'python "{SRC / "benchmark.py"}" --emb "{emb_lsi}" --processed_npz "{proc_npz}" --pq_joblib "{pq_joblib}" --lsh_joblib "{lsh_job}" --faiss_joblib "{faiss_job}" --k {k} --n_queries 500 --out "{metrics_out}"'
        subprocess.run(cmd3, shell=True, check=True)
        with open(metrics_out) as f:
            data = json.load(f)
        data['num_perm'] = num_perm
        data['threshold'] = thr
        metrics.append(data)

# convert to table and plot recall vs time for each config
rows = []
for m in metrics:
    for method in ['pq','lsh','faiss']:
        if method in m:
            rows.append({
                'method': method,
                'num_perm': m['num_perm'],
                'threshold': m['threshold'],
                'avg_time': m[method]['avg_time'],
                'recall': m[method]['recall']
            })
df = pd.DataFrame(rows)

# static seed for jitter reproducibility
np.random.seed(0)

# produce readable scatter with non-overlapping annotations via jitter + white box
for method in df['method'].unique():
    plt.figure(figsize=(10, 7))
    subset = df[df['method'] == method]
    plt.scatter(subset['avg_time'], subset['recall'], s=120)

    for i, r in subset.iterrows():
        label = f"np{int(r['num_perm'])}_thr{int(r['threshold']*100)}"
        # small deterministic jitter
        x_offset = ( (hash(label) % 10) - 5 ) * 1e-4
        y_offset = ( (hash(label[::-1]) % 10) - 5 ) * 1e-3
        plt.annotate(
            label,
            (r['avg_time'], r['recall']),
            xytext=(r['avg_time'] + x_offset, r['recall'] + y_offset),
            textcoords='data',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.6)
        )

    plt.xlabel("Average Query Time (s)")
    plt.ylabel("Recall")
    plt.title(f"{method.upper()} Parameter Sweep")
    plt.grid(True, linestyle="--", alpha=0.4)
    outfile = FIG / f"{method}_sweep.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved", outfile)

# Additional plots: recall vs num_perm, time vs num_perm, recall vs threshold, time vs threshold
for method in df['method'].unique():
    subset = df[df['method']==method]
    plt.figure(figsize=(8,5))
    for thr in sorted(subset['threshold'].unique()):
        s = subset[subset['threshold']==thr]
        plt.plot(s['num_perm'], s['recall'], marker='o', label=f"thr={thr}")
    plt.title(f"{method.upper()} - Recall vs num_perm")
    plt.xlabel("num_perm")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.legend()
    plt.savefig(FIG / f"{method}_recall_vs_numperm.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,5))
    for thr in sorted(subset['threshold'].unique()):
        s = subset[subset['threshold']==thr]
        plt.plot(s['num_perm'], s['avg_time'], marker='o', label=f"thr={thr}")
    plt.title(f"{method.upper()} - Time vs num_perm")
    plt.xlabel("num_perm")
    plt.ylabel("avg_time (s)")
    plt.grid(True)
    plt.legend()
    plt.savefig(FIG / f"{method}_time_vs_numperm.png", dpi=300, bbox_inches='tight')
    plt.close()

print("Additional comparison plots saved in:", FIG)






