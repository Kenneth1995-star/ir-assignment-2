# src/lsh_cosine.py
import numpy as np
import joblib
from collections import defaultdict

class RandomHyperplaneLSH:
    def __init__(self, dim, n_bits=128, seed=42):
        np.random.seed(seed)
        self.n_bits = n_bits
        self.projections = np.random.randn(n_bits, dim).astype('float32')
        self.buckets = defaultdict(list)

    def _hash(self, vec):
        # vec: 1D numpy array
        vec = vec.astype('float32')
        proj = self.projections.dot(vec)
        bits = (proj > 0).astype(int)
        # pack into string key
        return ''.join(['1' if b else '0' for b in bits])

    def index(self, X):
        for i, v in enumerate(X):
            key = self._hash(v)
            self.buckets[key].append(i)

    def query(self, qvec):
        key = self._hash(qvec)
        return list(self.buckets.get(key, []))

if __name__ == "__main__":
    import argparse, numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_bits", type=int, default=128)
    args = parser.parse_args()
    X = np.load(args.emb).astype('float32')
    lsh = RandomHyperplaneLSH(dim=X.shape[1], n_bits=args.n_bits)
    lsh.index(X)
    joblib.dump(lsh, args.out)
    print("Saved Cosine LSH (random hyperplane) to", args.out)
