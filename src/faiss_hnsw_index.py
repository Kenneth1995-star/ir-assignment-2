import numpy as np
import joblib

# Here, I am writing a helper function that tries to import FAISS.
# If FAISS is not installed on the system, I am catching the exception
# and returning None so that the program does not crash.
def try_import_faiss():
    try:
        import faiss
        return faiss
    except Exception as e:
        print("FAISS not available:", e)
        return None


# Here, I am building a FAISS index using IndexFlatIP.
# I am using IP (inner product) after normalizing vectors,
# because cosine similarity equals the inner product of L2-normalized vectors.
def build_faiss_index(X):
    faiss = try_import_faiss()
    if faiss is None:
        raise RuntimeError("FAISS is not installed on this system.")

    # Here, I am converting all vectors to float32
    # because FAISS requires float32 inputs.
    X32 = X.astype("float32")

    # Here, I am normalizing all vectors to unit length.
    # This makes cosine similarity equivalent to FAISS inner product scoring.
    faiss.normalize_L2(X32)

    # Here, I am getting the dimensionality of the vectors.
    d = X32.shape[1]

    # Here, I am creating a simple brute-force FAISS index
    # that computes inner products between the query and all stored vectors.
    index = faiss.IndexFlatIP(d)

    # Here, I am adding all vectors to the index.
    index.add(X32)

    return index


# Here, I am defining a search function for querying FAISS.
# This function returns both the indices (I) and similarity scores (D).
def search_faiss(index, q, k=10):
    import faiss

    # Here, I am converting the query to float32 and reshaping it
    # so FAISS can process it correctly.
    q32 = q.astype("float32").reshape(1, -1)

    # Here, I am normalizing the query vector in exactly the same way
    # as the dataset vectors were normalized during indexing.
    faiss.normalize_L2(q32)

    # Here, I am performing the top-k similarity search.
    # FAISS returns:
    #  - D: similarity scores
    #  - I: indices of the nearest documents
    D, I = index.search(q32, k)

    # Here, I am returning the first row (since we only queried 1 vector).
    return I[0], D[0]


# Here, I am providing an alternative ANN index using the HNSW algorithm.
# HNSW works well for cosine similarity if vectors are normalized.
def build_hnswlib_index(X, space="cosine", ef_construction=200, M=16):
    import hnswlib

    # Here, I am converting to float32 for compatibility.
    Xf = X.astype("float32")
    n, d = Xf.shape

    # Here, I am initializing an HNSW index in the chosen metric space.
    p = hnswlib.Index(space=space, dim=d)

    # Here, I am creating the index structure with the chosen parameters.
    #  - max_elements: total capacity
    #  - ef_construction: accuracy-speed trade-off during building
    #  - M: number of bi-directional graph links
    p.init_index(max_elements=n, ef_construction=ef_construction, M=M)

    # Here, I am inserting all vectors into the HNSW graph.
    p.add_items(Xf)

    # Here, I am setting the ef parameter for queries.
    # Higher ef gives better recall but slower search.
    p.set_ef(50)

    return p


# Here, I am creating the CLI entry point of the script.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True,
                        help="Here, I am specifying the path to the .npy embeddings file.")
    parser.add_argument("--out_prefix", required=True,
                        help="Here, I am specifying the prefix for the saved index.")
    parser.add_argument("--index", choices=["faiss", "hnsw"], default="faiss",
                        help="Here, I am choosing which index type to build.")
    args = parser.parse_args()

    # Here, I am loading all high-dimensional document embeddings.
    X = np.load(args.emb)

    # Here, I am building and saving the requested ANN index type.
    if args.index == "faiss":
        idx = build_faiss_index(X)
        joblib.dump(idx, args.out_prefix + "_faiss.joblib")
        print("Saved FAISS index to", args.out_prefix + "_faiss.joblib")

    else:  # HNSW
        idx = build_hnswlib_index(X)
        joblib.dump(idx, args.out_prefix + "_hnsw.joblib")
        print("Saved HNSW index to", args.out_prefix + "_hnsw.joblib")












