import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib
import os

def product_quantize(X, M=8, Ks=256, seed=42):
    """
    Here, I am implementing the core logic for Product Quantization (PQ) .
    I am taking the high-dimensional embedding matrix X and compressing it.
    I am dividing the vector space into M subspaces (e.g., 8 blocks).
    For each subspace, I am clustering the data using K-Means to find Ks codebook centers (e.g., 256 centers).
    The original vector is then replaced by a short vector of M indices, one for the closest center in each subspace.
    """
    n, d = X.shape
    # I am checking that the dimension is cleanly divisible by the number of subspaces.
    assert d % M == 0, "d must be divisible by M"
    sub_d = d // M
    codebooks = [] # This will store the cluster centers for each subspace.
    codes = np.empty((n, M), dtype=np.int32) # This will store the index/code for each document in each subspace.
    
    for m in range(M):
        # I am extracting the m-th subspace block of the data.
        block = X[:, m*sub_d:(m+1)*sub_d]
        
        # I am running K-Means to find Ks cluster centers for this block. MiniBatchKMeans is faster.
        kmeans = MiniBatchKMeans(n_clusters=Ks, random_state=seed, n_init=3)
        
        # I am getting the cluster index (code) for every document's sub-vector.
        labels = kmeans.fit_predict(block)
        codes[:, m] = labels
        
        # I am saving the cluster centers (the codebook) for this subspace.
        codebooks.append(kmeans.cluster_centers_)
        
    # I am returning the codebooks (for decoding/querying) and the compressed codes.
    return codebooks, codes

def pq_approx_search(query, codebooks, codes, top_k=10):
    """
    Here, I am implementing the Product Quantization Approximate Search using the Asymmetric Distance Computation (ADC).
    Instead of calculating a true distance, I am estimating it very quickly by looking up
    the pre-computed distances between the query's sub-vectors and the codebook centers.
    """
    n = codes.shape[0]
    M = len(codebooks)
    sub_d = codebooks[0].shape[1]
    
    # This array will accumulate the distance contribution from each subspace for every document.
    dist = np.zeros(n)
    
    for m in range(M):
        # I am extracting the m-th sub-vector of the query.
        qsub = query[m*sub_d:(m+1)*sub_d]
        # I am getting the codebook (cluster centers) for the m-th subspace.
        cb = codebooks[m]
        
        # I am calculating the squared Euclidean distance between the query sub-vector (qsub)
        # and ALL the cluster centers (cb) in this subspace.
        dists_centers = ((cb - qsub)**2).sum(axis=1)
        
        # Here is the ADC magic: I am looking up the pre-calculated distance for each document's code.
        # codes[:, m] holds the index (0 to Ks-1) of the closest center for the m-th subspace for every document.
        # I am adding this looked-up distance to the total distance.
        dist += dists_centers[codes[:, m]]
        
    # I am sorting the total estimated distances and getting the indices of the top-k nearest neighbors.
    idx = np.argsort(dist)[:top_k]
    
    # I am returning the indices and their corresponding distances.
    return idx, dist[idx]

if __name__ == "__main__":
    # Here, I am setting up the CLI to load embeddings and build the PQ index.
    import argparse, numpy as np
    parser = argparse.ArgumentParser()
    # Input is the LSI embedding file.
    parser.add_argument("--emb", required=True, help="np file of embeddings")
    # Output path prefix for the saved index.
    parser.add_argument("--out_prefix", required=True)
    # Parameters for Product Quantization: number of subspaces (M) and codebook size (Ks).
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Ks", type=int, default=256)
    args = parser.parse_args()
    
    # I am loading the LSI embeddings.
    X = np.load(args.emb)
    
    # I am building the PQ index.
    codebooks, codes = product_quantize(X, M=args.M, Ks=args.Ks)
    
    # I am preparing the output file path.
    outpath = args.out_prefix + "_pq.joblib"
    
    # I am saving the codebooks and codes (the PQ index) using joblib.
    joblib.dump({"codebooks": codebooks, "codes": codes}, outpath)
    print("Saved PQ index:", outpath)



