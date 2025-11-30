# src/lsh_diagnostics.py
import joblib, numpy as np, argparse

def lsh_candidate_stats(lsh_joblib_path, show_first=20):
    d = joblib.load(lsh_joblib_path)
    if not isinstance(d, dict) or 'lsh' not in d or 'sketches' not in d:
        raise RuntimeError("Expected joblib to contain dict with keys 'lsh' and 'sketches'")
    lsh = d['lsh']
    sketches = d['sketches']
    counts = []
    self_only = 0
    examples = []
    for i, mh in enumerate(sketches):
        cand = lsh.query(mh)
        c = cand if cand else []
        counts.append(len(c))
        # count if only candidate is itself (string)
        if len(c) == 1 and str(i) in [str(x) for x in c]:
            self_only += 1
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--lsh_joblib", required=True)
    parser.add_argument("--show_first", type=int, default=20)
    args = parser.parse_args()
    lsh_candidate_stats(args.lsh_joblib, show_first=args.show_first)






