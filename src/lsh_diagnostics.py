# HERE I AM DOING diagnostics on my LSH joblib files. 
# I AM loading a saved LSH index and its sketches to analyze candidate statistics.

import joblib
import numpy as np
import argparse

def lsh_candidate_stats(lsh_joblib_path, show_first=20):
    """
    HERE I AM loading the joblib file containing the LSH index and MinHash sketches.
    I AM calculating statistics such as mean/median number of candidates and fraction
    of documents returning only themselves.
    """
    d = joblib.load(lsh_joblib_path)

    # check that the joblib file has the expected structure
    if not isinstance(d, dict) or 'lsh' not in d or 'sketches' not in d:
        raise RuntimeError("Expected joblib to contain dict with keys 'lsh' and 'sketches'")

    lsh = d['lsh']
    sketches = d['sketches']

    counts = []   # HERE I AM storing number of candidates per sketch
    self_only = 0 # HERE I AM counting sketches where only itself is returned
    examples = [] # HERE I AM storing first few example results for inspection

    for i, mh in enumerate(sketches):
        cand = lsh.query(mh)  # HERE I AM querying LSH for candidates
        c = cand if cand else []
        counts.append(len(c))

        # count if only candidate is itself (string format)
        if len(c) == 1 and str(i) in [str(x) for x in c]:
            self_only += 1

        # store first few examples
        if i < show_first:
            examples.append((i, c[:10]))

    counts = np.array(counts)
    print("LSH candidates: mean", counts.mean(), "median", np.median(counts), "min", counts.min(), "max", counts.max())
    print("Fraction with zero candidates:", float((counts==0).mean()))
    print("Fraction self-only (only self returned):", float(self_only / len(counts)))
    print("First", show_first, "examples (doc_id, candidates[:10]):")
    for e in examples:
        print(" ", e)

if __name__ == "__main__":
    # HERE I AM parsing command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lsh_joblib", required=True)
    parser.add_argument("--show_first", type=int, default=20)
    args = parser.parse_args()

    # HERE I AM running the diagnostics
    lsh_candidate_stats(args.lsh_joblib, show_first=args.show_first)







