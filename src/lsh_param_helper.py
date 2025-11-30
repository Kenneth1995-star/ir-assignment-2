# src/lsh_param_helper.py
import numpy as np
import json, random
from tqdm import tqdm

def compute_random_jaccard_samples(token_sets, n_samples=2000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    n = len(token_sets)
    samples = []
    for _ in range(n_samples):
        i = random.randrange(n)
        j = random.randrange(n)
        if i==j:
            continue
        A = token_sets[i]; B = token_sets[j]
        if len(A|B)==0:
            sim = 0.0
        else:
            sim = len(A & B) / len(A | B)
        samples.append(sim)
    return samples

if __name__ == "__main__":
    import argparse, numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_npz", required=True)
    parser.add_argument("--out_json", default="results/jaccard_samples.json")
    parser.add_argument("--n_samples", type=int, default=2000)
    args = parser.parse_args()

    data = np.load(args.processed_npz, allow_pickle=True)
    token_lists = list(data["token_lists"])
    token_sets = [set(t) for t in token_lists]
    samples = compute_random_jaccard_samples(token_sets, n_samples=args.n_samples)
    import statistics
    print("Jaccard percentiles: 10,25,50,75,90,95:", [np.percentile(samples,p) for p in (10,25,50,75,90,95)])
    with open(args.out_json,'w') as f:
        json.dump({"samples": samples}, f)
    print("Saved samples to", args.out_json)
