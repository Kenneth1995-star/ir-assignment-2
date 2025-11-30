# src/lsh_index.py
import argparse
import joblib
from datasketch import MinHashLSH

def build_lsh(sketches, num_perm, threshold):
    """
    Build MinHash LSH. We store keys as strings "i" so joblib can save easily.
    """
    # MinHashLSH uses banding internally; threshold controls recall/precision trade-off
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    print("Building LSH index (this can take a while for many docs)...")
    for i, sk in enumerate(sketches):
        lsh.insert(str(i), sk)

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

    # Save both the LSH and the sketches; benchmark and diagnostics rely on both
    out_obj = {"lsh": lsh, "sketches": sketches}
    joblib.dump(out_obj, args.out)
    print("Saved MinHash LSH + sketches to", args.out)




