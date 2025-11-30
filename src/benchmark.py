import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time, joblib, json, os
import random

# Here, I am computing brute-force cosine Top-K for ground truth or comparison.
# I use cosine_similarity from sklearn on the entire matrix.
def brute_force_topk_cosine(X, qvec, k=10):
    sims = cosine_similarity(X, qvec.reshape(1, -1)).ravel()
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

# Here, I am computing brute-force Jaccard Top-K for set-based LSH ground truth.
def brute_force_topk_jaccard(token_sets, qset, k=10):
    n = len(token_sets)
    scores = np.zeros(n, dtype=float)

    # Here, I am looping through every set and computing true Jaccard similarity.
    for i, s in enumerate(token_sets):
        inter = len(qset & s)
        union = len(qset | s)
        scores[i] = 0.0 if union == 0 else inter / union

    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]


# Here, I am evaluating PQ, LSH, and FAISS on multiple random queries.
# I am computing recall@k and average query time for each method.
def evaluate_on_queries(X, queries_idx, indices, k=10, token_sets=None, debug_sample=5):
    results = {}

    # ---------------------------------------------------------
    # Here, I am computing exact COSINE ground truth for PQ & FAISS
    # ---------------------------------------------------------
    gt_cosine = {}
    if any(m in indices for m in ("pq", "faiss")):
        for q in queries_idx:
            idxs, _ = brute_force_topk_cosine(X, X[q], k=k+1)
            gt = [int(i) for i in idxs if int(i) != int(q)]
            gt_cosine[q] = gt[:k]

    # ---------------------------------------------------------
    # Here, I am computing exact JACCARD ground truth for LSH
    # ---------------------------------------------------------
    gt_jaccard = {}
    if "lsh" in indices and indices["lsh"] is not None:
        if token_sets is None:
            raise RuntimeError("Here, I am stopping because LSH requires token_sets (pass --processed_npz).")

        for q in queries_idx:
            idxs, _ = brute_force_topk_jaccard(token_sets, token_sets[q], k=k+1)
            gt = [int(i) for i in idxs if int(i) != int(q)]
            gt_jaccard[q] = gt[:k]

    # ---------------------------------------------------------
    # Here, I am evaluating PRODUCT QUANTIZATION (PQ)
    # ---------------------------------------------------------
    if "pq" in indices and indices["pq"] is not None:
        search_fn = indices["pq"]["search_fn"]
        pq_times, pq_recalls = [], []

        for q in queries_idx:
            t0 = time.time()
            idxs, _ = search_fn(X[q], top_k=k+1)
            pq_times.append(time.time() - t0)

            idxs = [int(i) for i in np.array(idxs).reshape(-1).tolist() if int(i) != q][:k]
            pq_recalls.append(len(set(idxs) & set(gt_cosine[q])) / float(k))

        results["pq"] = {
            "avg_time": float(np.mean(pq_times)),
            "recall": float(np.mean(pq_recalls))
        }

    # ---------------------------------------------------------
    # Here, I am evaluating LOCALITY SENSITIVE HASHING (LSH)
    # ---------------------------------------------------------
    if "lsh" in indices and indices["lsh"] is not None:
        lsh_struct = indices["lsh"]
        lsh = lsh_struct["lsh"]
        sketches = lsh_struct["sketches"]

        lsh_times, lsh_recalls = [], []

        for q in queries_idx:
            mh = sketches[q]
            t0 = time.time()
            cand = lsh.query(mh)
            lsh_times.append(time.time() - t0)

            # Here, I am re-ranking LSH candidates using true Jaccard.
            if cand:
                cand_ids = [int(x) for x in cand if int(x) != q]
                if len(cand_ids) > 0:
                    qset = token_sets[q]
                    scores = []
                    for cid in cand_ids:
                        inter = len(qset & token_sets[cid])
                        union = len(qset | token_sets[cid])
                        scores.append(0.0 if union == 0 else inter / union)
                    ords = np.argsort(-np.array(scores))[:k]
                    retrieved = [cand_ids[i] for i in ords]
                else:
                    retrieved = []
            else:
                retrieved = []

            lsh_recalls.append(len(set(retrieved) & set(gt_jaccard[q])) / float(k))

        results["lsh"] = {
            "avg_time": float(np.mean(lsh_times)),
            "recall": float(np.mean(lsh_recalls))
        }

    # ---------------------------------------------------------
    # Here, I am evaluating FAISS
    # ---------------------------------------------------------
    if "faiss" in indices and indices["faiss"] is not None:
        idx = indices["faiss"]["index"]
        search_fn = indices["faiss"]["search_fn"]
        faiss_times, faiss_recalls = [], []

        for q in queries_idx:
            t0 = time.time()
            I, D = search_fn(idx, X[q], k=k+1)
            faiss_times.append(time.time() - t0)

            I = [int(i) for i in np.array(I).reshape(-1).tolist() if int(i) != q][:k]
            faiss_recalls.append(len(set(I) & set(gt_cosine[q])) / float(k))

        results["faiss"] = {
            "avg_time": float(np.mean(faiss_times)),
            "recall": float(np.mean(faiss_recalls))
        }

    # ---------------------------------------------------------
    # Here, I am printing a small human-readable DEBUG SAMPLE
    # ---------------------------------------------------------
    if len(queries_idx) > 0:
        print("\n[DEBUG] Showing sample queries:")
        for q in queries_idx[:debug_sample]:
            print(f"\nQuery {q}:")
            if "faiss" in results:
                print("  Cosine GT:", gt_cosine[q][:5])
                I, _ = indices["faiss"]["search_fn"](indices["faiss"]["index"], X[q], k=k+1)
                I = [int(i) for i in np.array(I).reshape(-1).tolist() if int(i) != q][:k]
                print("  FAISS retrieved:", I[:5])

            if "lsh" in results:
                print("  Jaccard GT:", gt_jaccard[q][:5])
                cand = indices["lsh"]["lsh"].query(indices["lsh"]["sketches"][q])
                cand = [int(x) for x in cand if int(x) != q] if cand else []
                print("  LSH candidates:", len(cand))

            if "pq" in results:
                pq_idxs, _ = indices["pq"]["search_fn"](X[q], top_k=k+1)
                pq_idxs = [int(i) for i in np.array(pq_idxs).reshape(-1).tolist() if int(i) != q][:k]
                print("  PQ retrieved:", pq_idxs[:5])

    return results


# ----------------------------------------------------
# Here, I am defining the CLI execution block
# ----------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True)
    parser.add_argument("--processed_npz")
    parser.add_argument("--pq_joblib")
    parser.add_argument("--lsh_joblib")
    parser.add_argument("--faiss_joblib")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--out", default="results/metrics.json")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    X = np.load(args.emb)
    n = X.shape[0]
    queries_idx = random.sample(range(n), min(args.n_queries, n))
    indices = {}

    # Here, I am loading PQ index
    if args.pq_joblib:
        pq = joblib.load(args.pq_joblib)
        import quantize as qmod
        indices["pq"] = {
            "codebooks": pq["codebooks"],
            "codes": pq["codes"],
            "search_fn": lambda q, top_k: qmod.pq_approx_search(q, pq["codebooks"], pq["codes"], top_k)
        }

    # Here, I am loading LSH index
    token_sets = None
    if args.lsh_joblib:
        lsh_struct = joblib.load(args.lsh_joblib)
        if not ("lsh" in lsh_struct and "sketches" in lsh_struct):
            raise RuntimeError("Here, I am stopping because LSH joblib is malformed.")
        indices["lsh"] = lsh_struct

        if args.processed_npz is None:
            raise RuntimeError("Here, I need processed_npz for LSH ground truth evaluation.")

        pdata = np.load(args.processed_npz, allow_pickle=True)
        token_lists = list(pdata["token_lists"])
        token_sets = [set(t) for t in token_lists]

    # Here, I am loading FAISS/HNSW index
    if args.faiss_joblib:
        faiss_idx = joblib.load(args.faiss_joblib)
        from faiss_hnsw_index import search_faiss
        indices["faiss"] = {"index": faiss_idx, "search_fn": search_faiss}

    results = evaluate_on_queries(
        X, queries_idx, indices, k=args.k, token_sets=token_sets
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved metrics to", args.out)










