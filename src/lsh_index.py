# HERE I AM building a MinHash LSH index from sketches and saving it to joblib.
# I AM also saving the sketches with it for later diagnostics or benchmarking.

import argparse
import joblib
from datasketch import MinHashLSH

def build_lsh(sketches, num_perm, threshold):
    """
    HERE I AM building the LSH index.
    - `sketches` is a list of MinHash objects
    - `num_perm` is number of hash permutations
    - `threshold` is the Jaccard similarity threshold for candidate retrieval
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    print("Building LSH index (this can take a while for many docs)...")
    for i, sk in enumerate(sketches):
        lsh.insert(str(i), sk)  # HERE I AM inserting keys as strings for joblib safety
    return lsh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketches", required=True, help="joblib file with list of MinHash sketches")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_perm", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    sketches = joblib.load(args.sketches)
    lsh = build_lsh(sketches, num_perm=args.num_perm, threshold=args.threshold)

    # HERE I AM saving both the LSH index and sketches for later use
    out_obj = {"lsh": lsh, "sketches": sketches}
    joblib.dump(out_obj, args.out)
    print("Saved MinHash LSH + sketches to", args.out)





